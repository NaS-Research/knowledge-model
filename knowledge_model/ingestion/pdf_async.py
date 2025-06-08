"""
Asynchronous PDF downloader for PMC articles.

Usage
-----
>>> from knowledge_model.ingestion.pdf_async import fetch_pdfs_async
>>> results = asyncio.run(fetch_pdfs_async(["PMC1234567", "PMC7654321"]))
>>> for pmcid, path in results.items():
...     print(pmcid, "→", path or "no‑pdf")

Design
------
* Keeps **≤ 10 requests / second / IP** (NCBI limit when you provide an
  API‑key).
* Caps in‑flight coroutines to 10 (to avoid unbounded memory use).
* Tries up to three URL patterns for each PMCID.
* Exponential back‑off with jitter on HTTP 5xx / timeouts.
* Writes PDFs to a temp directory and returns their paths; returns *None*
  when no PDF is available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from httpx import RemoteProtocolError
from aiolimiter import AsyncLimiter

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MAX_RPS = int(os.getenv("PMC_PDF_MAX_RPS", "10"))     # default 10 req / sec
MAX_CONCURRENCY = int(os.getenv("PMC_PDF_MAX_PAR", "10"))

limiter = AsyncLimiter(MAX_RPS, time_period=1.0)
sem = asyncio.Semaphore(MAX_CONCURRENCY)

API_KEY = os.getenv("NCBI_API_KEY")  # optional – appended to every request

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _candidate_urls(pmcid: str) -> List[str]:
    """
    Generate possible download URLs for a given PMCID.

    The first form works for most modern deposits; the second and third catch
    older edge‑cases.
    """
    return [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf?download=1",
    ]


async def _request_pdf(url: str, client: httpx.AsyncClient, attempt: int) -> bytes | None:
    """
    Perform a GET against *url* with retries and back‑off.  Returns bytes or
    None when the server's Content‑Type is not PDF.
    """
    # Back‑off: 0.5, 1.0, 2.0 …  + jitter(0–0.3)
    if attempt > 1:
        base = 0.5 * (2 ** (attempt - 2))
        await asyncio.sleep(base + random.uniform(0.0, 0.3))

    # HEAD first – cheap reject on HTML
    try:
        head = await client.head(url, follow_redirects=True)
        ctype = head.headers.get("Content-Type", "")
        if "application/pdf" not in ctype.lower():
            return None

        resp = await client.get(url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        return resp.content
    except (RemoteProtocolError, httpx.TimeoutException) as exc:
        # bubble up so caller can retry
        raise exc


async def _fetch_single_pmc(
    pmcid: str, client: httpx.AsyncClient, outdir: Path
) -> Optional[Path]:
    """
    Download the PDF for *pmcid* or return None.

    We try up to three URL permutations × three attempts each.
    """
    async with sem, limiter:
        for url in _candidate_urls(pmcid):
            for attempt in range(1, 4):
                try:
                    data = await _request_pdf(url, client, attempt)
                    if not data:
                        # Not a PDF -> stop trying other attempts, but move on
                        # to alternative URL pattern.
                        break
                    out_path = outdir / f"{pmcid}.pdf"
                    out_path.write_bytes(data)
                    logger.debug("Downloaded PDF for %s (%d bytes)", pmcid, len(data))
                    return out_path
                except (httpx.HTTPStatusError, httpx.TimeoutException, RemoteProtocolError) as exc:
                    # Retry on HTTP 5xx or timeout
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and 500 <= exc.response.status_code < 600
                        and attempt < 3
                    ):
                        logger.debug(
                            "Retry %d/3 for %s – HTTP %s",
                            attempt,
                            url,
                            exc.response.status_code,
                        )
                        continue
                    logger.debug("PDF download failed for %s: %s", url, exc)
                    break  # move to next URL pattern
        # All patterns exhausted
        logger.debug("No PDF available for %s", pmcid)
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def fetch_pdfs_async(pmcids: List[str]) -> Dict[str, Optional[Path]]:
    """
    Concurrently download PDFs for a list of *pmcids*.

    Returns a dict {pmcid: Path | None}.  Uses a shared httpx.AsyncClient and
    respects global rate / concurrency limits.
    """
    if not pmcids:
        return {}

    # Ensure tmp dir exists per‑call
    tmp_root = Path(tempfile.mkdtemp(prefix="pmc_pdf_"))

    params = {"api_key": API_KEY} if API_KEY else None

    async with httpx.AsyncClient(params=params, timeout=60) as client:
        tasks = {pmcid: _fetch_single_pmc(pmcid, client, tmp_root) for pmcid in pmcids}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))