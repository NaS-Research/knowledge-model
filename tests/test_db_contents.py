"""
test_s3_links.py
Queries the database to print each article's S3 PDF URL.
"""

import logging
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_s3_links():
    db = SessionLocal()
    try:
        articles = db.query(Article).all()
        logger.info("Found %d articles in the DB", len(articles))
        for i, art in enumerate(articles, start=1):
            logger.info("%d) PMID: %s | S3 URL: %s", i, art.pmid, art.pdf_s3_url)
    finally:
        db.close()

if __name__ == "__main__":
    test_s3_links()
