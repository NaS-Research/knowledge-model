import json
import logging
import os
import sys

from knowledge_model.ingestion.fetch_pubmed import fetch_articles
from knowledge_model.ingestion.download_pdf import download_pmc_pdf
from knowledge_model.ingestion.parse_pdfs import parse_pdf
from knowledge_model.ingestion.upload_s3 import upload_pdf_to_s3
from knowledge_model.processing.text_cleaner import clean_text, chunk_text
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article, ArticleChunk

logger = logging.getLogger(__name__)

def run_pipeline(query="machine learning in cancer", max_results=5, chunk_size=1000):
    logger.info("Fetching articles for '%s'", query)
    articles = fetch_articles(query, max_results)
    db = SessionLocal()

    # Ensure output directory for JSONL dataset exists
    output_file = "data/science_articles/train.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        for art in articles:
            pmid = art.get("pmid")
            pmcid_original = art.get("pmcid")
            doi = art.get("doi")

            pmcid_clean = None
            if pmcid_original:
                pmcid_clean = pmcid_original.replace("pmc-id:", "").replace(";", "").strip()

            existing_article = db.query(Article).filter(Article.pmid == pmid).first()

            title = clean_text(art.get("title", "Untitled"))
            authors = ", ".join(art.get("authors", []))
            journal = clean_text(art.get("journal", "")) or None
            pubdate = clean_text(art.get("pubdate", "")) or None
            abstract = clean_text(art.get("abstract", "")) or None

            pdf_s3_url = None
            pdf_chunks = []
            pdf_downloaded = False

            if pmcid_clean:
                skip_download = False
                if existing_article and existing_article.pdf_downloaded:
                    logger.info("Already have PDF for PMID %s; skipping download.", pmid)
                    skip_download = True

                if not skip_download:
                    try:
                        pdf_path = download_pmc_pdf(pmcid_clean)
                        parsed_pdf = parse_pdf(pdf_path)
                        cleaned_pdf_text = clean_text(parsed_pdf["text"])
                        pdf_chunks = chunk_text(cleaned_pdf_text, chunk_size=chunk_size)

                        pdf_s3_url = upload_pdf_to_s3(pdf_path)
                        os.remove(pdf_path)

                        pdf_downloaded = True
                    except Exception as e:
                        logger.warning(
                            "PDF download/parse/upload failed for %s (PMCID: %s): %s",
                            pmid, pmcid_clean, e
                        )
                        pdf_downloaded = False

            if not existing_article:
                db_article = Article(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    journal=journal,
                    pubdate=pubdate,
                    abstract=abstract,
                    pdf_s3_url=pdf_s3_url,
                    doi=doi,
                    content=None,
                    pdf_downloaded=pdf_downloaded
                )
                db.add(db_article)
                db.commit()

                if pdf_downloaded and pdf_chunks:
                    for i, chunk_str in enumerate(pdf_chunks):
                        db_chunk = ArticleChunk(
                            article_id=db_article.id,
                            chunk_index=i,
                            chunk_text=chunk_str
                        )
                        db.add(db_chunk)
                    db.commit()

                    # Append each chunk to the JSONL dataset file
                    with open(output_file, "a", encoding="utf-8") as f:
                        for chunk_str in pdf_chunks:
                            record = {
                                "pmid": pmid,
                                "title": title,
                                "text": chunk_str
                            }
                            f.write(json.dumps(record) + "\n")
            else:
                logger.info("Found existing article for PMID %s", pmid)
                existing_article.title = title
                existing_article.authors = authors
                existing_article.journal = journal
                existing_article.pubdate = pubdate
                existing_article.abstract = abstract
                existing_article.doi = doi

                if pdf_downloaded and pdf_chunks:
                    existing_article.pdf_s3_url = pdf_s3_url
                    existing_article.pdf_downloaded = True

                    for i, chunk_str in enumerate(pdf_chunks):
                        db_chunk = ArticleChunk(
                            article_id=existing_article.id,
                            chunk_index=i,
                            chunk_text=chunk_str
                        )
                        db.add(db_chunk)
                    db.commit()
                    # Append new chunks to the JSONL dataset file
                    with open(output_file, "a", encoding="utf-8") as f:
                        for chunk_str in pdf_chunks:
                            record = {
                                "pmid": pmid,
                                "title": title,
                                "text": chunk_str
                            }
                            f.write(json.dumps(record) + "\n")
        logger.info("Inserted/Updated %d articles", len(articles))

        from knowledge_model.ingestion.upload_s3 import upload_dataset_to_s3
        dataset_url = upload_dataset_to_s3(output_file)
        logger.info("Dataset uploaded to: %s", dataset_url)

    except Exception as e:
        logger.exception("Error during pipeline: %s", e)
        db.rollback()
    finally:
        db.close()

def test_open_access(chunk_size=1000):
    """
    Force-test a single known open-access PMC article to verify PDF download and S3 upload.
    """
    db = SessionLocal()
    try:
        forced_article = {
            "pmid": "TEST-12345",
            "pmcid": "PMC7327471",
            "doi": "10.1016/j.cell.2020.06.023",
            "title": "An example open access article",
            "authors": ["Smith J", "Doe A"],
            "journal": "Cell",
            "pubdate": "2020",
            "abstract": "Test abstract for an open access article."
        }

        pmid = forced_article["pmid"]
        pmcid_clean = forced_article["pmcid"]
        existing_article = db.query(Article).filter(Article.pmid == pmid).first()

        pdf_s3_url = None
        pdf_chunks = []
        pdf_downloaded = False

        if existing_article and existing_article.pdf_downloaded:
            logger.info("Already have PDF for PMID %s; skipping download.", pmid)
        else:
            try:
                pdf_path = download_pmc_pdf(pmcid_clean)
                parsed_pdf = parse_pdf(pdf_path)
                cleaned_pdf_text = clean_text(parsed_pdf["text"])
                pdf_chunks = chunk_text(cleaned_pdf_text, chunk_size=chunk_size)

                pdf_s3_url = upload_pdf_to_s3(pdf_path)
                os.remove(pdf_path)

                pdf_downloaded = True
            except Exception as e:
                logger.warning(
                    "PDF download/parse/upload failed for %s (PMCID: %s): %s",
                    pmid, pmcid_clean, e
                )
                pdf_downloaded = False

        if not existing_article:
            db_article = Article(
                pmid=pmid,
                title=forced_article["title"],
                authors=", ".join(forced_article["authors"]),
                journal=forced_article["journal"],
                pubdate=forced_article["pubdate"],
                abstract=forced_article["abstract"],
                pdf_s3_url=pdf_s3_url,
                doi=forced_article["doi"],
                pdf_downloaded=pdf_downloaded,
                content=None
            )
            db.add(db_article)
            db.commit()

            if pdf_downloaded and pdf_chunks:
                for i, chunk_str in enumerate(pdf_chunks):
                    db_chunk = ArticleChunk(
                        article_id=db_article.id,
                        chunk_index=i,
                        chunk_text=chunk_str
                    )
                    db.add(db_chunk)
                db.commit()
                with open("data/science_articles/train.jsonl", "a", encoding="utf-8") as f:
                    for chunk_str in pdf_chunks:
                        record = {
                            "pmid": pmid,
                            "title": forced_article["title"],
                            "text": chunk_str
                        }
                        f.write(json.dumps(record) + "\n")
        else:
            existing_article.pdf_downloaded = existing_article.pdf_downloaded or pdf_downloaded
            existing_article.pdf_s3_url = existing_article.pdf_s3_url or pdf_s3_url

            if pdf_downloaded and pdf_chunks:
                for i, chunk_str in enumerate(pdf_chunks):
                    db_chunk = ArticleChunk(
                        article_id=existing_article.id,
                        chunk_index=i,
                        chunk_text=chunk_str
                    )
                    db.add(db_chunk)
                db.commit()
                with open("data/science_articles/train.jsonl", "a", encoding="utf-8") as f:
                    for chunk_str in pdf_chunks:
                        record = {
                            "pmid": pmid,
                            "title": forced_article["title"],
                            "text": chunk_str
                        }
                        f.write(json.dumps(record) + "\n")
        logger.info("Test open-access ingestion complete.")
    except Exception as e:
        logger.exception("Error during test_open_access: %s", e)
        db.rollback()
    finally:
        db.close()

def main():
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    if len(args) == 1 and args[0].lower() == "test_oa":
        test_open_access()
    else:
        run_pipeline(query="SARS-CoV-2 open access", max_results=5)

if __name__ == "__main__":
    main()