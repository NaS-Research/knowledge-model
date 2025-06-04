

"""
Prefect task wrapper around the library‑level `build_faiss_index`.

Usage
-----
>>> from pipelines.tasks import build_faiss
>>> build_faiss(jsonl="data/clean/2025/05/chunks.jsonl",
...             outdir="data/index/2025/05")

The heavy lifting (loading, embedding, indexing) lives in
`knowledge_model.ingestion.build_faiss.build_faiss_index`.  Wrapping it in a
Prefect task gives you retries, logging, and DAG‑level visibility.
"""

from __future__ import annotations

import os
from pathlib import Path

from prefect import task, get_run_logger
from knowledge_model.ingestion.build_faiss import build_faiss_index


@task(retries=0, log_prints=True)
def build_faiss(jsonl: str | os.PathLike, outdir: str | os.PathLike) -> None:
    """
    Build a FAISS index from *jsonl* and write it to *outdir*.

    Parameters
    ----------
    jsonl : str | Path
        Path to the cleaned JSONL file produced by `fetch_clean_month`.
    outdir : str | Path
        Directory where `faiss.index` + `meta.npy` should be written.
    """
    logger = get_run_logger()
    jsonl = Path(jsonl)
    outdir = Path(outdir)

    logger.info("Building FAISS index for %s → %s", jsonl, outdir)
    build_faiss_index(jsonl=jsonl, outdir=outdir)
    logger.info("FAISS index written to %s", outdir)


# allow star‑import from pipelines.tasks
__all__ = ["build_faiss"]