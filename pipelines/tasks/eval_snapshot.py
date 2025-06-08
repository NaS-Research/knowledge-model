

"""
Task: eval_snapshot
-------------------
Evaluate the most recently‑built FAISS index against a fixed query set and
return recall@10.

A small JSON‑Lines file at ``tests/eval_queries.jsonl`` must exist with objects
like::

    {"query": "metformin renal clearance", "pmid": "24717411"}

The task:

1. Locates the newest index folder under ``data/index/YYYY/MM``.
2. Loads ``faiss.index`` and ``meta.npy`` from that folder.
3. Encodes each query with *all‑MiniLM‑L6‑v2*.
4. Computes recall@10 and logs the result.

Returns
-------
float
    The recall@10 score (0.0 – 1.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from knowledge_model.config.settings import settings
from typing import List, Tuple

import faiss
import numpy as np
from prefect import task, get_run_logger
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
INDEX_ROOT = settings.DATA_DIR / "index"  # where YYYY/MM/ directories live
EVAL_PATH = settings.REPO_ROOT / "tests" / "eval_queries.jsonl"  # fixed evaluation set
K = 10                                        # recall@K


# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------
def _latest_index_dir() -> Path:
    """
    Return the directory of the most recent FAISS index under INDEX_ROOT.

    Assumes the structure data/index/YYYY/MM/faiss.index
    """
    index_files = sorted(INDEX_ROOT.glob("*/**/faiss.index"))
    if not index_files:
        raise RuntimeError(f"No FAISS indexes found under {INDEX_ROOT}")
    return index_files[-1].parent


def _load_eval_set() -> List[Tuple[str, str]]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {EVAL_PATH}. "
            "Add a JSONL file with {'query': str, 'pmid': str} rows."
        )
    rows = []
    for line in EVAL_PATH.read_text().splitlines():
        obj = json.loads(line)
        rows.append((obj["query"], obj["pmid"]))
    return rows


# --------------------------------------------------------------------------------------
# Prefect task
# --------------------------------------------------------------------------------------
@task
def eval_snapshot() -> float:
    """
    Run the fixed evaluation set against the newest index and return recall@10.
    """
    logger = get_run_logger()

    # Locate resources ----------------------------------------------------------
    idx_dir = _latest_index_dir()
    logger.info("Evaluating index in %s", idx_dir)

    index = faiss.read_index(str(idx_dir / "faiss.index"))
    meta = np.load(idx_dir / "meta.npy", allow_pickle=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Evaluate ------------------------------------------------------------------
    eval_rows = _load_eval_set()
    hits = 0
    for query, expected_pmid in eval_rows:
        vec = model.encode([query]).astype(np.float32)
        _, indices = index.search(vec, K)
        retrieved = [
            meta[i]["pmid"] for i in indices[0] if i != -1
        ]  # guard against padding -1
        if expected_pmid in retrieved:
            hits += 1

    recall = hits / len(eval_rows)
    logger.info(
        "Evaluation complete — recall@%d = %.3f  (%d/%d)",
        K,
        recall,
        hits,
        len(eval_rows),
    )
    return recall


# Allow star‑import from pipelines.tasks
__all__ = ["eval_snapshot"]