
"""
Prefect task: build_faiss
------------------------
Given a directory that already contains one or many `*.jsonl` files of
cleaned / chunked articles, create a FAISS vector index + metadata array
and save them to the target output directory.

The heavy lifting lives in `knowledge_model.ingestion.build_faiss
.build_faiss_index`, which is a pure‑Python helper wrapping the
sentence‑transformer encoder and FAISS itself.  This task is just a
cookie‑thin Prefect wrapper so we can orchestrate it inside a flow,
retry if needed, and see nice logs in the Prefect UI.
"""
from __future__ import annotations

import os
from pathlib import Path

from prefect import task, get_run_logger

from knowledge_model.ingestion.build_faiss import build_faiss_index

# --------------------------------------------------------------------------------------
# Prefect task
# --------------------------------------------------------------------------------------
@task(retries=0, log_prints=True, name="build_faiss")
def build_faiss(src_dir: str | os.PathLike, outdir: str | os.PathLike) -> None:
    """Build a FAISS index from *src_dir* and write it to *outdir*.

    Parameters
    ----------
    src_dir : str | Path
        Directory containing one or more `*.jsonl` files (cleaned chunks).
    outdir : str | Path
        Destination directory – the task will create it if needed – where
        `faiss.index` and `meta.npy` will be written.
    """
    logger = get_run_logger()

    src_dir = Path(src_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Building FAISS index for %s → %s", src_dir, outdir)
    build_faiss_index(src_dir=src_dir, outdir=outdir)
    logger.info("FAISS index written to %s", outdir)


# allow star‑import from pipelines.tasks
__all__ = ["build_faiss"]
