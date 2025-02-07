"""
pipeline.py
Orchestrates ingestion: fetches articles from PubMed, downloads PDFs if possible,
parses and cleans text, then stores everything in the database.
"""

import logging
from knowledge_model.ingestion.fetch_pubmed import fetch_articles
from knowledge_model.ingestion.download_pdf import download_pmc_pdf
from knowledge_model.ingestion.parse_pdfs import parse_pdf
from knowledge_model.processing.text_cleaner import clean_text
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article

logger = logging.getLogger(__name__)

def run_pipeline(query="machine learning in cancer", max_results=5):
    logger.info("Fetching articles for '%s'", query)
    articles = fetch_articles(query, max_results)

    db = SessionLocal()
    try:
        for art in articles:
            pmid = art.get("pmid")
            pmcid = art.get("pmcid")

            # Skip if this PMID already exists
            existing = db.query(Article).filter(Article.pmid == pmid).first()
            if existing:
                logger.info("Skipping duplicate PMID: %s", pmid)
                continue

            # Clean text fields
            title = clean_text(art.get("title", "Untitled"))
            authors = ", ".join(art.get("authors", []))
            journal = clean_text(art.get("journal", "")) or None
            pubdate = clean_text(art.get("pubdate", "")) or None
            abstract = clean_text(art.get("abstract", "")) or None

            # Attempt PDF download & parsing
            pdf_text = None
            if pmcid:
                try:
                    pdf_path = download_pmc_pdf(pmcid)
                    parsed_pdf = parse_pdf(pdf_path)
                    pdf_text = clean_text(parsed_pdf["text"])
                except Exception as e:
                    logger.warning("PDF download/parse failed for %s: %s", pmcid, e)

            # Create DB record
            db_article = Article(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                pubdate=pubdate,
                abstract=abstract,
                content=pdf_text
            )
            db.add(db_article)

        db.commit()
        logger.info("Inserted %d articles", len(articles))
    except Exception as e:
        logger.exception("Error inserting articles: %s", e)
        db.rollback()
    finally:
        db.close()

def main():
    logging.basicConfig(level=logging.INFO)
    run_pipeline()

if __name__ == "__main__":
    main()
