"""
Prefect task that builds or updates a FAISS index for **one month** of
cleaned PubMed data.

Usage
-----
>>> from pipelines.tasks import build_faiss
>>> build_faiss(
...     src_dir="data/clean/2013/01",  # directory returned by fetch_clean_month
... )

The heavy lifting (loading, embedding, indexing) is delegated to
`knowledge_model.ingestion.build_faiss.build_faiss_index`.  Wrapping it in a
Prefect task provides retries, structured logs and visibility within a Prefect
DAG.
"""
from __future__ import annotations

import os
from pathlib import Path

from prefect import task, get_run_logger

from knowledge_model.ingestion.build_faiss import build_faiss_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CLEAN_ROOT = Path("data/clean")
_INDEX_ROOT = Path("data/index")


def _mirror_index_dir(src_dir: Path) -> Path:
    """Return `data/index/YYYY/MM` for a given `data/clean/YYYY/MM`."""
    try:
        rel = src_dir.relative_to(_CLEAN_ROOT)
    except ValueError as exc:  # pragma: no cover – defensive
        raise ValueError(
            f"Expected src_dir under {_CLEAN_ROOT}, got {src_dir}") from exc
    return _INDEX_ROOT / rel


# ---------------------------------------------------------------------------
# Prefect task
# ---------------------------------------------------------------------------
@task(retries=0, log_prints=True, name="build_faiss")
def build_faiss(src_dir: str | os.PathLike) -> None:  # noqa: D401 – imperative
    """Embed *all* JSONL files in *src_dir* and build a FAISS index.

    Parameters
    ----------
    src_dir : str | Path
        Directory produced by `fetch_clean_month`, e.g. ``data/clean/2013/01``.
        The task automatically mirrors this under ``data/index/YYYY/MM``.
    """

    logger = get_run_logger()
    src_dir = Path(src_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Clean directory not found: {src_dir}")

    outdir = _mirror_index_dir(src_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Building FAISS index for %%s → %%s", src_dir, outdir)
    build_faiss_index(jsonl=src_dir, outdir=outdir)  # accepts dir or file
    logger.info("FAISS index written to %s", outdir)


__all__ = ["build_faiss"]