"""Builds and queries a local Faiss index of article chunks.

Example:
    store = LocalFaiss.load()
    results = store.search(query_vector)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDER_ID = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 12
INDEX_PATH = Path("data/faiss.idx")
META_PATH = Path("data/faiss.idx.meta")
CLEAN_DIR = Path("data/clean")


def _device() -> str:
    """Determine the device to use for embedding.

    Returns:
        str: 'mps' if available, otherwise 'cpu'.
    """
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalFaiss:
    """Wrapper around a local FAISS index with chunk metadata.

    Args:
        dim (int): Dimension of the vectors to be indexed.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        """Add vectors and their metadata to the index.

        Args:
            vecs (np.ndarray): Array of vectors to add.
            metas (List[Dict[str, Any]]): Corresponding metadata for each vector.
        """
        self.index.add(vecs.astype("float32"))
        self.meta.extend(metas)

    def save(self, outdir: Path | str | None = None) -> None:
        """Persist the FAISS index and metadata.

        Args:
            outdir: Directory (or file stem) to write to.  
                    • If *None*, falls back to the default INDEX_PATH.  
                    • If a directory, writes `<outdir>/faiss.idx` and
                      `<outdir>/faiss.idx.meta`.  
                    • If a path ending with “.idx”, uses that exact file name.
        """
        # Resolve target paths -------------------------------------------------
        if outdir is None:
            idx_path = INDEX_PATH
        else:
            out = Path(outdir)
            if out.is_dir():
                idx_path = out / "faiss.idx"
            elif out.suffix == ".idx":
                idx_path = out
            else:
                idx_path = out.with_suffix(".idx")
            idx_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = idx_path.with_suffix(".idx.meta")

        # ---------------------------------------------------------------------
        try:
            faiss.write_index(self.index, str(idx_path))
            meta_path.write_bytes(pickle.dumps(self.meta))
            logging.info(
                "Saved FAISS index (%d vectors) → %s",
                len(self.meta),
                idx_path.parent,
            )
        except Exception as exc:  # pragma: no cover
            logging.error("Failed to save FAISS artefacts: %s", exc)
            raise

    @classmethod
    def load(cls, path: Path | str | None = None) -> "LocalFaiss":
        """Load the FAISS index and metadata from disk.

        Optionally specify a custom path for the index file; defaults to INDEX_PATH for backward compatibility.

        Args:
            path (Path | str | None, optional): Custom index file location.

        Returns:
            LocalFaiss: Loaded LocalFaiss instance.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        # Path resolution logic for index file
        idx_path = Path(path) if path is not None else INDEX_PATH
        if idx_path.is_dir():
            idx_path = idx_path / "faiss.idx"
        elif idx_path.suffix == "":
            idx_path = idx_path.with_suffix(".idx")
        meta_path = idx_path.with_suffix(".idx.meta")
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {idx_path}")
        idx = faiss.read_index(str(idx_path))
        meta = pickle.loads(meta_path.read_bytes())
        store = cls(idx.d)
        store.index = idx
        store.meta = meta
        logging.info(f"Loaded FAISS index with {len(store.meta)} chunks from {idx_path}")
        return store

    def search(
        self,
        vec: np.ndarray,
        *,
        k: int = DEFAULT_TOP_K,
        min_score: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Return up to *k* passages with cosine‑similarity ≥ *min_score*.

        Args:
            vec: Single query vector (already L2‑normalised).
            k:   Max number of results to return **after** filtering.
                 Defaults to ``DEFAULT_TOP_K`` (12).
            min_score: Similarity cut‑off (0–1). Anything below
                       this threshold is ignored (default 0.75).

        Returns:
            List of metadata dictionaries ordered by descending score.
        """
        # Ask FAISS for a generous pool (2× target) to survive filtering
        raw_k = max(k * 2, k + 4)
        D, I = self.index.search(vec.astype("float32"), raw_k)

        hits: list[dict[str, Any]] = []
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            score = float(D[0][rank])  # inner‑product similarity (0‑1)
            if score < min_score:
                continue
            hits.append(self.meta[idx] | {"score": score})
            if len(hits) == k:
                break
        return hits


def build_store(clean_dir: Path = CLEAN_DIR) -> None:
    """Build and save a FAISS index from cleaned article chunks.

    Args:
        clean_dir (Path, optional): Directory containing cleaned JSON files. Defaults to CLEAN_DIR.
    """
    logging.basicConfig(level=logging.INFO)
    embedder = SentenceTransformer(
        DEFAULT_EMBEDDER_ID,
        device=_device(),
    )
    store: Optional[LocalFaiss] = None

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
        logging.info(f"Indexed {len(store.meta)} chunks → {INDEX_PATH}")
    else:
        logging.warning("No chunks found. Did you run text_cleaner?")


if __name__ == "__main__":
    build_store()