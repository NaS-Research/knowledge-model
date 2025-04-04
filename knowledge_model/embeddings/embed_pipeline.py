"""
embed_pipeline.py
Generates embeddings for each article chunk using DeepSeek-VL2
and stores them in the database or a vector store.
"""

import logging
import numpy as np
import pickle 
from sqlalchemy.orm import Session

from knowledge_model.embeddings.embedder_vl2 import DeepSeekVL2Embedder
from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import ArticleChunk

logger = logging.getLogger(__name__)

def run_embedding_pipeline(batch_size=100):
    """
    1. Load the embedder
    2. Iterate over chunks in 'article_chunks'
    3. Generate embeddings & store them in DB or a vector store
    :param batch_size: Number of chunks processed per batch
    """
    logger.info("Initializing DeepSeek embedder (1.3b chat)...")
    embedder = DeepSeekVL2Embedder(
        model_path="deepseek-ai/deepseek-vl2-tiny", 
        dtype=None,    # or torch.float16
        device=None
    )

    logger.info("Testing a short embed (or generation) ...")
    test_text = "Hello from the new DeepSeek 1.3b chat model!"
    embedding = embedder.embed_text(test_text)
    logger.info(f"Embedding shape: {embedding.shape}")

    db = SessionLocal()
    try:
        chunks = db.query(ArticleChunk).all()
        total = len(chunks)
        logger.info("Found %d chunks to embed", total)

        for i, chunk in enumerate(chunks, start=1):
            text_content = chunk.chunk_text


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
