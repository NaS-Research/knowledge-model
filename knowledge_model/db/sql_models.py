"""
sql_models.py
-------------
Defines SQLAlchemy models for storing article metadata.
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text

Base = declarative_base()

class Article(Base):
    """
    Represents an article with basic metadata and content.
    """
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    pmid = Column(String(32), unique=True, nullable=True)
    title = Column(String(500), nullable=False)
    authors = Column(String(500))
    journal = Column(String(300))
    pubdate = Column(String(50))
    content = Column(Text)
