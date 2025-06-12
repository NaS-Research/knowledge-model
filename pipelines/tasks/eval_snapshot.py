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
from knowledge_model.config.settings import DATA_ROOT
from typing import List, Tuple

import faiss
import numpy as np
from prefect import task, get_run_logger
from sentence_transformers import SentenceTransformer

DEFAULT_INDEX_ROOT = DATA_ROOT / "index"
EVAL_PATH  = DATA_ROOT.parent / "tests" / "eval_queries.jsonl"
K = 10


# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------
def _latest_index_dir(idx_root: Path) -> Path:
    """
    Return the directory of the most recent FAISS index under *idx_root*.

    Assumes the structure idx_root/YYYY/MM/faiss.index
    """
    # walk recursively; handles YYYY/MM/ or deeper nests
    index_files = sorted(idx_root.rglob("faiss.index"))
    if not index_files:
        raise RuntimeError(f"No FAISS indexes found under {idx_root}")
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
def eval_snapshot(idx_root: str | Path | None = None) -> float:
    """
    Run the fixed evaluation set against the newest index inside *idx_root*
    (defaults to DATA_ROOT/index) and return recall@10.
    """
    logger = get_run_logger()

    idx_root_path = Path(idx_root) if idx_root else DEFAULT_INDEX_ROOT
    idx_dir = _latest_index_dir(idx_root_path)
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
        retrieved = [meta[i]["pmid"] for i in indices[0] if i != -1]
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