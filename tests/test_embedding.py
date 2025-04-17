import numpy as np
import pytest
from numpy import dot
from numpy.linalg import norm

from knowledge_model.embeddings.embedder_vl2 import DeepSeekVL2Embedder


@pytest.fixture(scope="module")
def embedder() -> DeepSeekVL2Embedder:
    """Fixture to initialize the DeepSeekVL2Embedder."""
    return DeepSeekVL2Embedder()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    return float(dot(v1, v2) / (norm(v1) * norm(v2)))


def test_basic_text_embedding(embedder: DeepSeekVL2Embedder) -> None:
    """Test embedding for basic text."""
    text = "Hello world!"
    emb = embedder.embed_text(text)
    assert emb.shape == (1280,), f"Expected shape (1280,), got {emb.shape}"
    assert not np.isnan(emb).any(), "Embedding contains NaNs"


def test_similar_texts(embedder: DeepSeekVL2Embedder) -> None:
    """Test embedding for similar texts."""
    text1 = "Hello world!"
    text2 = "Greetings planet!"
    sim = cosine_similarity(embedder.embed_text(text1), embedder.embed_text(text2))
    assert sim > 0.2, "Expected some similarity between related texts"


def test_different_texts(embedder: DeepSeekVL2Embedder) -> None:
    """Test embedding for different texts."""
    text1 = "Hello world!"
    text2 = "I enjoy pizza."
    sim = cosine_similarity(embedder.embed_text(text1), embedder.embed_text(text2))
    assert sim < 0.9, "Expected a lower similarity between unrelated texts"


def test_multimodal_embedding(embedder: DeepSeekVL2Embedder) -> None:
    """Test multimodal embedding with text and image."""
    text = "A cute cat on a couch."
    image_paths = ["tests/test_api/sample_cat.jpg"]
    emb = embedder.embed_multimodal(text, image_paths)
    assert emb.shape == (1280,), f"Expected shape (1280,), got {emb.shape}"
    assert not np.isnan(emb).any(), "Multimodal embedding contains NaNs"
