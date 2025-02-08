from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    pmid = Column(String(32), unique=True, nullable=True)
    title = Column(String(500), nullable=False)
    authors = Column(String(500))
    journal = Column(String(300))
    pubdate = Column(String(50))
    abstract = Column(Text)
    content = Column(Text)
    pdf_s3_url = Column(String(500))
    doi = Column(String(128), nullable=True)
    pdf_downloaded = Column(Boolean, default=False)  # <--- NEW

    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")

class ArticleChunk(Base):
    __tablename__ = "article_chunks"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)

    article = relationship("Article", back_populates="chunks")

class ArticleChunkEmbedding(Base):
    __tablename__ = "article_chunk_embeddings"

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey("article_chunks.id"), unique=True)
    embedding = Column(LargeBinary)
