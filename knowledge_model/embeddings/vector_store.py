"""
Builds and queries a local Faiss index of article chunks.

Run:  python -m knowledge_model.embeddings.vector_store
"""

import os, json, pickle, pathlib, numpy as np, faiss
from sentence_transformers import SentenceTransformer

EMBEDDER_ID = "all-MiniLM-L6-v2"      # small, fits on CPU/MPS
INDEX_PATH  = "data/faiss.idx"        # binary index file
META_PATH   = "data/faiss.idx.meta"   # pickled metadata list
CLEAN_DIR   = "data/clean"            # cleaned article JSON files

# ---------- Helper class -------------------------------------------------

class LocalFaiss:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta  = []               # parallel list with chunk metadata

    def add(self, vecs: np.ndarray, metas: list[dict]):
        self.index.add(vecs.astype("float32"))
        self.meta.extend(metas)

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        pathlib.Path(META_PATH).write_bytes(pickle.dumps(self.meta))

    @classmethod
    def load(cls):
        if not pathlib.Path(INDEX_PATH).exists():
            raise FileNotFoundError("Index not built yet.")
        idx  = faiss.read_index(INDEX_PATH)
        meta = pickle.loads(pathlib.Path(META_PATH).read_bytes())
        store = cls(idx.d)
        store.index, store.meta = idx, meta
        return store

    def search(self, vec: np.ndarray, k: int = 5):
        D, I = self.index.search(vec.astype("float32"), k)
        return [self.meta[i] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]

# ---------- Build function ----------------------------------------------

def build_store(clean_dir: str = CLEAN_DIR):
    embedder = SentenceTransformer(EMBEDDER_ID, device="mps" if hasattr(__import__('torch'), 'backends') and __import__('torch').backends.mps.is_available() else "cpu")
    store = None

    for jf in pathlib.Path(clean_dir).glob("*.json"):
        rec = json.loads(jf.read_text())
        chunks = rec.get("chunks") or []
        if not chunks:
            continue
        vecs = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        metas = [
            {"article_id": rec["id"], "chunk_id": i, "text": t}
            for i, t in enumerate(chunks)
        ]
        if store is None:
            store = LocalFaiss(vecs.shape[1])
        store.add(np.array(vecs), metas)

    if store:
        pathlib.Path("data").mkdir(exist_ok=True)
        store.save()
        print(f"Indexed {len(store.meta)} chunks → {INDEX_PATH}")
    else:
        print("No chunks found. Did you run text_cleaner?")

# ---------- CLI entry ----------------------------------------------------

if __name__ == "__main__":
    build_store()