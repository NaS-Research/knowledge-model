#!/usr/bin/env python3
"""
build_faiss.py

Utility script to create a FAISS vector index (plus ``meta.json``) from a
JSON-Lines corpus. Only lines that contain the chosen *field* are indexed.

Example
-------
python -m knowledge_model.ingestion.build_faiss \
    --jsonl  data/combined/combined_v2.jsonl \
    --outdir data/faiss \
    --model  all-MiniLM-L6-v2 \
    --field  text
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from knowledge_model.embeddings.vector_store import LocalFaiss

__all__ = ["build_faiss_index"]

# ───────────────────────────── configuration ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ───────────────────────────── helpers ───────────────────────────────────────


def _stream_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield one JSON object per line (constant-memory iterator)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line")
                continue

# ---------------------------------------------------------------------
# Stream utility that accepts *either* a single file or a directory
# ---------------------------------------------------------------------
def _iter_jsonl(src: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON objects from *src*, which may be:
      • a single ``*.jsonl`` file, or
      • a directory – we will stream **all** ``*.jsonl`` files found
        recursively inside it.

    This keeps memory usage constant while allowing per‑article files.
    """
    if src.is_dir():
        files = sorted(src.rglob("*.jsonl"))
        if not files:
            logger.warning("No .jsonl files found under %s", src)
        for file in files:
            yield from _stream_records(file)
    else:
        yield from _stream_records(src)


def _encode_texts(
    texts: List[str],
    model_id: str,
    batch: int = 64,
) -> np.ndarray:
    """Return L2-normalised embeddings for every text snippet."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedder = SentenceTransformer(model_id, device=device)

    logger.info("Embedding %d passages on %s …", len(texts), device)
    vecs = embedder.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


# ───────────────────────────── public API ────────────────────────────────────
def build_faiss_index(
    jsonl: Path,
    outdir: Path,
    model_id: str = "all-MiniLM-L6-v2",
    field: str = "text",
) -> None:
    """
    Programmatic entry‑point for building a FAISS index.

    Parameters
    ----------
    jsonl : Path
        Path to a JSON‑Lines corpus.
    outdir : Path
        Output directory where the FAISS files + ``meta.json`` are written.
    model_id : str, optional
        Sentence‑BERT model to embed passages with.
    field : str, optional
        JSON key whose value is embedded (default ``"text"``).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- load & filter --------
    logger.info("Loading %s …", jsonl)
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for rec in _iter_jsonl(jsonl):
        passage = rec.get(field)
        if passage:
            texts.append(passage)
            meta.append(rec)

    if not texts:
        raise ValueError(
            f"No lines in {jsonl} contained a '{field}' field – cannot build index."
        )
    logger.info("Loaded %d passages (kept)", len(texts))

    # -------- embed + write --------
    vecs = _encode_texts(texts, model_id=model_id)

    store = LocalFaiss(vecs.shape[1])
    store.add(vecs, meta)
    store.save(outdir)

    logger.info(
        "Index built → %s  |  dim=%d  |  rows=%d",
        outdir,
        vecs.shape[1],
        vecs.shape[0],
    )


# ───────────────────────────── main ──────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index + meta.json")
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to JSONL corpus")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-BERT model ID")
    parser.add_argument(
        "--field",
        default="text",
        help="JSON key whose value should be embedded (default: 'text')",
    )
    args = parser.parse_args()

    build_faiss_index(
        jsonl=args.jsonl,
        outdir=args.outdir,
        model_id=args.model,
        field=args.field,
    )


if __name__ == "__main__":
    main()
