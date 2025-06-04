"""
pipelines.tasks package

Re‑export the public Prefect tasks so callers can simply write

    from pipelines.tasks import fetch_clean_month, build_faiss

without knowing the sub‑module layout.
"""

from .tasks import fetch_clean_month
from .build_faiss import build_faiss

__all__: list[str] = [
    "fetch_clean_month",
    "build_faiss",
]
