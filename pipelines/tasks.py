"""
Prefect task wrappers around ingestion / training utilities.
"""

from prefect import task
from datetime import datetime, UTC

from knowledge_model.ingestion.pipeline import run_pipeline, _month_query
from knowledge_model.ingestion.build_faiss import build_faiss_index
# … import other helpers as you refactor …

@task(name="Fetch-Clean-Month", retries=2, retry_delay_seconds=300)
def fetch_clean_month(year: str | None = None, month: str | None = None):
    """Download & clean one PubMed month, write to DB + JSONL."""
    now = datetime.now(UTC)
    year = year or f"{now.year:04d}"
    month = month or f"{now.month:02d}"
    run_pipeline(_month_query(year, month))

@task(name="Build-FAISS", retries=1)
def build_faiss():
    build_faiss_index()         # whatever your current script entry-point is