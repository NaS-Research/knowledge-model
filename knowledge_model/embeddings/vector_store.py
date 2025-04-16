"""
Builds and queries a local Faiss index of article chunks.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDER_ID = "all-MiniLM-L6-v2"
INDEX_PATH = Path("data/faiss.idx")
META_PATH = Path("data/faiss.idx.meta")
CLEAN_DIR = Path("data/clean")


class LocalFaiss:
    """Wrapper around a local FAISS index with chunk metadata."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict[str, Any]] = []

    def add(self, vecs: np.ndarray, metas: list[dict[str, Any]]) -> None:
        self.index.add(vecs.astype("float32"))
        self.meta.extend(metas)

    def save(self) -> None:
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_bytes(pickle.dumps(self.meta))

    @classmethod
    def load(cls) -> LocalFaiss:
        if not INDEX_PATH.exists():
            raise FileNotFoundError("Index not built yet.")
        idx = faiss.read_index(str(INDEX_PATH))
        meta = pickle.loads(META_PATH.read_bytes())
        store = cls(idx.d)
        store.index = idx
        store.meta = meta
        return store

    def search(self, vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        D, I = self.index.search(vec.astype("float32"), k)
        return [
            self.meta[i] | {"score": float(D[0][j])}
            for j, i in enumerate(I[0])
        ]


def build_store(clean_dir: Path = CLEAN_DIR) -> None:
    embedder = SentenceTransformer(
        EMBEDDER_ID,
        device="mps" if hasattr(__import__('torch'), 'backends') and __import__('torch').backends.mps.is_available() else "cpu",
    )
    store: LocalFaiss | None = None

    for jf in clean_dir.rglob("*.json"):
        rec = json.loads(jf.read_text())
        chunks = rec["chunks"]
        if not chunks:
            continue
        vecs = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        metas = [{"article_id": rec["id"], "chunk_id": i, "text": t} for i, t in enumerate(chunks)]

        if store is None:
            store = LocalFaiss(vecs.shape[1])
        store.add(np.array(vecs), metas)

    if store:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        store.save()
        print(f"Indexed {len(store.meta)} chunks → {INDEX_PATH}")
    else:
        print("No chunks found. Did you run text_cleaner?")


if __name__ == "__main__":
    build_store()