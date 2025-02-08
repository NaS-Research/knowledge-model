"""
embed_pipeline.py
Generates embeddings for each article chunk using DeepSeek-VL2
and stores them in the database or a vector store.
"""

import logging
import numpy as np
import pickle  # or struct/base64, etc., if you store embeddings in DB
from sqlalchemy.orm import Session

from knowledge_model.embeddings.embedder_vl2 import DeepSeekVL2Embedder
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import ArticleChunk
# If storing embeddings in DB, you might also import:
# from knowledge_model.db.sql_models import ArticleChunkEmbedding

logger = logging.getLogger(__name__)

def run_embedding_pipeline(batch_size=100):
    """
    1. Load the embedder
    2. Iterate over chunks in 'article_chunks'
    3. Generate embeddings & store them in DB or a vector store
    :param batch_size: Number of chunks processed per batch
    """
    logger.info("Initializing DeepSeek-VL2 embedder...")
    embedder = DeepSeekVL2Embedder(
        model_path="deepseek-ai/deepseek-vl2-small",  # or local path
        dtype=None,        # defaults to bfloat16 if available
        device=None        # auto-detect 'cuda' if available
    )

    db = SessionLocal()
    try:
        # For demonstration, fetch all chunks:
        chunks = db.query(ArticleChunk).all()
        total = len(chunks)
        logger.info("Found %d chunks to embed", total)

        for i, chunk in enumerate(chunks, start=1):
            text_content = chunk.chunk_text

            # If you have image paths associated with each chunk, you'd do:
            # image_paths = chunk.image_paths or []
            # embedding_vector = embedder.embed_multimodal(text_content, image_paths=image_paths)

            embedding_vector = embedder.embed_text(text_content)
            # embedding_vector is a NumPy array (float32 or float16)

            # (Optional) Store in DB as pickle:
            # embedding_bytes = pickle.dumps(embedding_vector, protocol=pickle.HIGHEST_PROTOCOL)
            # new_embedding = ArticleChunkEmbedding(chunk_id=chunk.id, embedding=embedding_bytes)
            # db.add(new_embedding)

            # Or push to a vector DB (Faiss, Pinecone, etc.)

            if i % batch_size == 0:
                logger.info("Processed %d/%d chunks", i, total)
                db.commit()

        db.commit()
        logger.info("Finished embedding pipeline.")

    except Exception as e:
        logger.exception("Error in embedding pipeline: %s", e)
        db.rollback()
    finally:
        db.close()

def main():
    logging.basicConfig(level=logging.INFO)
    run_embedding_pipeline()

if __name__ == "__main__":
    main()
