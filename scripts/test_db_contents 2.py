"""
test_db_contents.py
-------------------
Quick script to confirm articles are in the database and print them.
"""

import logging
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_db_contents():
    db = SessionLocal()
    try:
        articles = db.query(Article).all()
        logger.info("Found %d articles in the database", len(articles))
        for idx, art in enumerate(articles, start=1):
            logger.info(
                "%d) PMID: %s | Title: %s | Authors: %s",
                idx,
                art.pmid,
                art.title,
                art.authors
            )
    finally:
        db.close()

if __name__ == "__main__":
    print_db_contents()
