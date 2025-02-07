"""
pipeline.py
Orchestrates ingestion: fetches articles from PubMed, stores them in the DB.
"""

import logging
from knowledge_model.ingestion.fetch_pubmed import fetch_articles
from knowledge_model.ingestion.parse_pdfs import parse_pdf
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article

logger = logging.getLogger(__name__)

def run_pipeline(query="machine learning in cancer", max_results=5):
    logger.info("Fetching articles for '%s'", query)
    articles = fetch_articles(query, max_results)
    db = SessionLocal()
    try:
        for art in articles:
            db_article = Article(
                pmid=art.get("pmid"),
                title=art.get("title", "Untitled"),
                authors=", ".join(art.get("authors", [])),
                journal=art.get("journal"),
                pubdate=art.get("pubdate"),
                content=None  # optional PDF content goes here
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
