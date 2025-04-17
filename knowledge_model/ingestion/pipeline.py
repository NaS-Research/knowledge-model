"""
Ingestion pipeline to fetch articles, download PDFs, extract content, and store clean chunks.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article, ArticleChunk
from knowledge_model.ingestion.download_pdf import download_pmc_pdf
from knowledge_model.ingestion.fetch_pubmed import fetch_articles
from knowledge_model.ingestion.parse_pdfs import parse_pdf
from knowledge_model.ingestion.upload_s3 import upload_dataset_to_s3
from knowledge_model.processing.text_cleaner import clean_text, chunk_text

logger = logging.getLogger(__name__)


def run_pipeline(query: str, max_results: int = 5, chunk_size: int = 1000) -> None:
    logger.info("Fetching articles for '%s'", query)
    articles = fetch_articles(query, max_results)
    logger.info("Fetched %d articles from PubMed.", len(articles))
    db = SessionLocal()

    match = re.search(r'"(\d{4})/(\d{2})/01"\[PDAT\]', query)
    if not match:
        raise ValueError("Query must contain a start date in the format 'YYYY/MM/01[PDAT]'")
    year, month = match.group(1), match.group(2)
    batch_label = f"{year}-{month}"
    output_file = Path(f"data/science_articles/{batch_label}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for art in articles:
            pmid = art.get("pmid")
            pmcid_raw = art.get("pmcid")
            pmcid = (pmcid_raw or "").replace("pmc-id:", "").replace(";", "").strip()
            doi = art.get("doi")
            pubdate = clean_text(art.get("pubdate") or "") or None

            existing = db.query(Article).filter(Article.pmid == pmid).first()

            title = clean_text(art.get("title") or "Untitled")
            authors = ", ".join(art.get("authors") or [])
            journal = clean_text(art.get("journal") or "") or None
            abstract = clean_text(art.get("abstract") or "") or None

            pdf_s3_url = None
            pdf_chunks: list[str] = []
            pdf_downloaded = False

            if pmcid and not (existing and existing.pdf_downloaded):
                try:
                    path = download_pmc_pdf(pmcid)
                    parsed = parse_pdf(path)
                    cleaned_text = clean_text(parsed["text"])
                    pdf_chunks = chunk_text(cleaned_text, chunk_size=chunk_size)
                    pdf_s3_url = upload_dataset_to_s3(path)
                    os.remove(path)
                    pdf_downloaded = True
                    logger.info("Parsed and chunked PDF for PMCID %s (%d chunks)", pmcid, len(pdf_chunks))
                except Exception as e:
                    logger.warning("PDF failed for %s (%s): %s", pmid, pmcid, e)
            else:
                logger.warning("No PMCID available or already downloaded for article %s", pmid)

            article = existing or Article(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                pubdate=pubdate,
                abstract=abstract,
                pdf_s3_url=pdf_s3_url,
                doi=doi,
                content=None,
                pdf_downloaded=pdf_downloaded,
            )
            db.add(article)
            db.commit()
            logger.info("Inserted new article %s into database.", pmid)

            if pdf_downloaded and pdf_chunks:
                for i, chunk in enumerate(pdf_chunks):
                    db_chunk = ArticleChunk(article_id=article.id, chunk_index=i, chunk_text=chunk)
                    db.add(db_chunk)
                db.commit()
                logger.info("Inserted %d chunks for article %s", len(pdf_chunks), pmid)

                clean_dir = Path("data/clean") / year / month
                clean_dir.mkdir(parents=True, exist_ok=True)
                base_name = f"{pmid}_{article.id}"
                existing = any(p.name.startswith(base_name) for p in clean_dir.glob("*.jsonl"))
                if existing:
                    logger.info("Skipping duplicate chunk write for %s", base_name)
                    continue

                with open(output_file, "a", encoding="utf-8") as train_f, \
                     open(clean_dir / f"{pmid}_{article.id}.jsonl", "a", encoding="utf-8") as clean_f:
                    for chunk in pdf_chunks:
                        record = {"pmid": pmid, "title": title, "text": chunk}
                        line = json.dumps(record) + "\n"
                        train_f.write(line)
                        clean_f.write(line)

                logger.info("Wrote cleaned chunks for %s to %s", pmid, clean_dir)

        logger.info("Inserted/Updated %d articles", len(articles))
        if output_file.exists():
            logger.info("Uploading training dataset %s to S3...", output_file.name)
            dataset_url = upload_dataset_to_s3(output_file)
            logger.info("Training dataset uploaded to: %s", dataset_url)
        else:
            logger.warning("No dataset file created at %s — skipping upload.", output_file)

    except Exception as e:
        logger.exception("Error during pipeline: %s", e)
        db.rollback()
    finally:
        db.close()


def test_open_access(chunk_size: int = 1000) -> None:
    db = SessionLocal()
    try:
        forced = {
            "pmid": "TEST-12345",
            "pmcid": "PMC7327471",
            "doi": "10.1016/j.cell.2020.06.023",
            "title": "An example open access article",
            "authors": ["Smith J", "Doe A"],
            "journal": "Cell",
            "pubdate": "2020",
            "abstract": "Test abstract for an open access article.",
        }

        pmid = forced["pmid"]
        pmcid = forced["pmcid"]
        existing = db.query(Article).filter(Article.pmid == pmid).first()

        pdf_s3_url = None
        pdf_chunks = []
        pdf_downloaded = False

        now = datetime.utcnow()
        clean_dir = Path("data/clean") / f"{now.year:04d}" / f"{now.month:02d}"
        clean_dir.mkdir(parents=True, exist_ok=True)

        if not (existing and existing.pdf_downloaded):
            try:
                path = download_pmc_pdf(pmcid)
                parsed = parse_pdf(path)
                cleaned = clean_text(parsed["text"])
                pdf_chunks = chunk_text(cleaned, chunk_size=chunk_size)
                pdf_s3_url = upload_dataset_to_s3(path)
                os.remove(path)
                pdf_downloaded = True
            except Exception as e:
                logger.warning("PDF failed for %s (%s): %s", pmid, pmcid, e)

        article = existing or Article(
            pmid=pmid,
            title=forced["title"],
            authors=", ".join(forced["authors"]),
            journal=forced["journal"],
            pubdate=forced["pubdate"],
            abstract=forced["abstract"],
            pdf_s3_url=pdf_s3_url,
            doi=forced["doi"],
            pdf_downloaded=pdf_downloaded,
            content=None,
        )
        db.add(article)
        db.commit()

        if pdf_downloaded and pdf_chunks:
            for i, chunk in enumerate(pdf_chunks):
                db_chunk = ArticleChunk(article_id=article.id, chunk_index=i, chunk_text=chunk)
                db.add(db_chunk)
            db.commit()

            with open("data/science_articles/train.jsonl", "a", encoding="utf-8") as train_f, \
                 open(clean_dir / f"{pmid}_{article.id}.jsonl", "a", encoding="utf-8") as clean_f:
                for chunk in pdf_chunks:
                    record = {"pmid": pmid, "title": forced["title"], "text": chunk}
                    line = json.dumps(record) + "\n"
                    train_f.write(line)
                    clean_f.write(line)

        logger.info("Test open-access ingestion complete.")

    except Exception as e:
        logger.exception("Error during test_open_access: %s", e)
        db.rollback()
    finally:
        db.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    if len(args) == 1 and args[0].lower() == "test_oa":
        test_open_access()
    else:
        run_pipeline(
            query='("2023/01/01"[PDAT] : "2023/01/31"[PDAT])',
            max_results=100,
        )


if __name__ == "__main__":
    main()
