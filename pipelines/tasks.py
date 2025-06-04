"""
Prefect task wrappers around ingestion / training utilities.
"""

from prefect import task
from datetime import datetime, UTC
from pathlib import Path
from typing import Tuple

from knowledge_model.ingestion.pipeline import run_pipeline, _month_query

from pipelines.tasks.build_faiss import build_faiss

# Root of monthly cleaned data
CLEAN_DIR = Path("data/clean")
START_YEAR = 2013
START_MONTH = 1

def _next_month(year: int, month: int) -> Tuple[int, int]:
    """Return the (year, month) tuple for the following calendar month."""
    return (year + 1, 1) if month == 12 else (year, month + 1)

def _first_missing_month() -> tuple[str, str]:
    """
    Scan CLEAN_DIR chronologically (starting 2013‑01) and return the first
    (YYYY, MM) that *does not* have a folder yet.
    """
    y, m = START_YEAR, START_MONTH
    while (CLEAN_DIR / f"{y}" / f"{m:02d}").exists():
        y, m = _next_month(y, m)
    return f"{y:04d}", f"{m:02d}"

@task(name="Fetch-Clean-Month", retries=2, retry_delay_seconds=300)
def fetch_clean_month(year: str | None = None, month: str | None = None):
    logger = fetch_clean_month.get_run_logger()  # Prefect‑provided logger

    # Determine which month to process
    y, m = (year, month) if year and month else _first_missing_month()
    logger.info("Fetching + cleaning PubMed data for %s-%s", y, m)

    run_pipeline(_month_query(y, m))
