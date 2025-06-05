"""
pipelines.tasks package

Re‑export the public Prefect tasks so callers can simply write

    from pipelines.tasks import fetch_clean_month, build_faiss

without knowing the sub‑module layout.
"""

from .fetch_clean import fetch_clean_month
from .build_faiss import build_faiss
from .eval_snapshot import eval_snapshot

__all__: list[str] = [
    "fetch_clean_month",
    "build_faiss",
    "eval_snapshot",
]
