"""
Prefect task that fetches and cleans one month of PubMed data.

Usage
-----
>>> from pipelines.tasks import fetch_clean
>>> fetch_clean(
...     year=2013,
...     month=1,
... )
"""

from __future__ import annotations

import os
from pathlib import Path
from knowledge_model.config.settings import DATA_ROOT
from prefect import task, get_run_logger

from knowledge_model.ingestion.fetch_clean import fetch_clean_month as _fetch_clean_month

CLEAN_DIR = DATA_ROOT / "clean"

def _ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

@task(retries=0, log_prints=True, name="fetch_clean")
def fetch_clean_month(year: int | None = None, month: int | None = None) -> str:
    """Fetch and clean PubMed data for one month.

    Parameters
    ----------
    year : int, optional
        Year of data to fetch. If None, finds first missing month.
    month : int, optional
        Month of data to fetch. If None, finds first missing month.

    Returns
    -------
    str
        Path to the cleaned chunks.jsonl file for the ingested month.
    """

    logger = get_run_logger()

    if year is None or month is None:
        year, month = _fetch_clean_month.find_first_missing_month(CLEAN_DIR)
    outdir = CLEAN_DIR / f"{year:04d}" / f"{month:02d}"

    _ensure_dir(outdir)

    logger.info(f"Fetching and cleaning data for {year:04d}/{month:02d} → {outdir}")
    _fetch_clean_month(year=year, month=month, outdir=outdir)
    chunks_path = outdir / "chunks.jsonl"
    return str(chunks_path)
