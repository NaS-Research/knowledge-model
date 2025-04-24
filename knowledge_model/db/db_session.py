"""
db_session.py
-------------
Creates the database engine and session, and provides init_db for table creation.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from knowledge_model.db.sql_models import Base

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///knowledge_model.db")

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Create all tables in the database if they do not exist.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

def get_db():
    """
    Yields a database session, ensuring it's closed after use.
    Useful for FastAPI or other frameworks that expect a session per request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
