# File: tests/test_embedding.py

import pytest
import numpy as np

from knowledge_model.embeddings.embedder_vl2 import DeepSeekVL2Embedder
from numpy.linalg import norm
from numpy import dot

@pytest.fixture(scope="module")
def embedder():
    # Initialize the embedder once for all tests.
    return DeepSeekVL2Embedder()

def cosine_sim(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

def test_basic_text_embedding(embedder):
    text = "Hello world!"
    emb = embedder.embed_text(text)
    assert emb.shape == (1280,), f"Expected shape (1280,), got {emb.shape}"
    assert not np.isnan(emb).any(), "Embedding contains NaNs"

def test_similar_texts(embedder):
    text1 = "Hello world!"
    text2 = "Greetings planet!"

    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)

    sim = cosine_sim(emb1, emb2)
    print(f"Cosine similarity between '{text1}' and '{text2}': {sim}")
    # Typically you'd expect a moderately high similarity
    assert sim > 0.2, "Expected some similarity between related texts"

def test_different_texts(embedder):
    text1 = "Hello world!"
    text2 = "I enjoy pizza."

    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)

    sim = cosine_sim(emb1, emb2)
    print(f"Cosine similarity between '{text1}' and '{text2}': {sim}")
    # Typically you'd expect a fairly low similarity
    assert sim < 0.9, "Expected a lower similarity between unrelated texts"

def test_multimodal_embedding(embedder):
    # Provide a real image path if you have one
    text = "A cute cat on a couch."
    image_paths = ["tests/test_api/sample_cat.jpg"]  # Example path
    
    emb = embedder.embed_multimodal(text, image_paths)
    assert emb.shape == (1280,), f"Expected shape (1280,), got {emb.shape}"
    assert not np.isnan(emb).any(), "Multimodal embedding contains NaNs"
