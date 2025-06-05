
"""
pipelines.tasks.fetch_clean
===========================

Prefect task that identifies the next missing PubMed month, runs the ingestion
pipeline for that month, and returns the path to the cleaned chunks file.

The month is considered "processed" if ``data/clean/YYYY/MM/chunks.jsonl``
exists.  On each call the task scans chronologically—from 2013‑01 upward—until
it finds the first month whose directory is missing, then ingests that month.

Returns
-------
str
    Absolute or relative path to ``data/clean/YYYY/MM/chunks.jsonl`` for the
    month just ingested.  This is consumed by the ``build_faiss`` task.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from prefect import task, get_run_logger

from knowledge_model.ingestion.pipeline import run_pipeline, _month_query
from datetime import date

 # Root where cleaned text is written by the ingestion pipeline
CLEAN_DIR = Path("data/clean")


# --------------------------------------------------------------------------- #
# Helper: detect first missing YYYY/MM                                        #
# --------------------------------------------------------------------------- #
def _first_missing_month() -> tuple[str, str]:
    """
    Return (YYYY, MM) for the earliest month that has not yet been ingested.
    Scans from 2013-01 up to the current month, looking for the first
    data/clean/YYYY/MM directory that does not exist.
    """
    start_year = 2013
    today = date.today()

    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            # stop scanning beyond the current month
            if year == today.year and month > today.month:
                break
            y, m = str(year), f"{month:02d}"
            if not (CLEAN_DIR / y / m).exists():
                return y, m

    raise RuntimeError("All months up to the present have been processed.")



def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Prefect task
# --------------------------------------------------------------------------- #
@task(retries=0, log_prints=True, name="Fetch-Clean-Month")
def fetch_clean_month(year: str | None = None, month: str | None = None) -> str:
    """
    Ingest the next missing PubMed month (or a specific one).

    Parameters
    ----------
    year, month : Optional[str]
        If both provided, ingest that specific month; otherwise automatically
        detect the first missing YYYY/MM under ``data/clean/``.

    Returns
    -------
    str
        Path to the cleaned ``chunks.jsonl`` file for the ingested month.
    """
    logger = get_run_logger()

    # Decide which month to process
    y, m = (year, month) if year and month else _first_missing_month()
    logger.info("Ingesting PubMed data for %s-%s", y, m)

    # Ensure output directory exists before ingestion writes files
    month_dir: Path = CLEAN_DIR / y / m
    _ensure_dir(month_dir)

    # Run the ingestion pipeline (ESearch → EFetch → PDF → clean → chunk)
    run_pipeline(_month_query(y, m))

    # Path expected by downstream tasks
    jsonl_path = month_dir / "chunks.jsonl"
    logger.info("Finished ingestion for %s-%s → %s", y, m, jsonl_path)

    return str(jsonl_path)


# Export symbol for star-imports
__all__: list[str] = ["fetch_clean_month"]