"""
Downloads an open-access PDF from PubMed Central given a PMCID.
"""

import logging
 
from pathlib import Path

import requests
import random
import time
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
from typing import Optional

_SESSION: Optional[requests.Session] = None


def download_pmc_pdf(pmcid: str, download_dir: str | Path = "/tmp", retries: int = 3) -> str | None:
    """
    Best‑effort download of an open‑access PMC PDF.

    Strategy
    --------
    1. Try canonical PMC URLs (…/pdf/{PMCID}.pdf then …/pdf).
       • If a 30x redirect ends in a PDF, accept it.
    2. If we still have HTML, parse the landing page and follow the first
       link that looks like a PDF on the publisher site.
    3. Give up after *retries* attempts.

    Returns the local file path on success or **None** if no PDF is available.
    Never raises for “legitimately no PDF” – only on network exceptions.
    """
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    download_dir = Path(download_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / f"{pmcid}.pdf"

    if path.exists() and path.stat().st_size > 25_000:
        logger.debug("PDF already cached – %s", path)
        return str(path)

    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(
            {"User-Agent": "Mozilla/5.0 (PMC‑DL/1.1; +https://example.org)"}
        )

    def _backoff(attempt: int) -> None:
        time.sleep(0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.3))

    # 1️⃣ canonical PMC location --------------------------------------------------
    canonical = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf",
    ]

    last_html = None
    for url in canonical:
        try:
            r = _SESSION.get(url, allow_redirects=True, timeout=(5, 60))
            ctype = r.headers.get("content-type", "")
            if "application/pdf" in ctype:
                path.write_bytes(r.content)
                return str(path)
            # follow redirect if target is a PDF
            if r.is_redirect:
                target = r.headers.get("location")
                if target:
                    r2 = _SESSION.get(target, allow_redirects=True, timeout=(5, 60))
                    if "application/pdf" in r2.headers.get("content-type", ""):
                        path.write_bytes(r2.content)
                        return str(path)
            last_html = r.text  # remember HTML for publisher parsing
        except requests.RequestException as exc:
            logger.debug("Network error %s while fetching %s", exc, url)

    # 2️⃣ scrape landing page for publisher PDF -----------------------------------
    if last_html:
        soup = BeautifulSoup(last_html, "lxml")
        link = None
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").lower()
            href = a["href"]
            if "pdf" in text or href.lower().endswith(".pdf"):
                link = href.strip()
                break

        if link:
            if link.startswith("/"):
                link = "https:" + link
            for attempt in range(1, retries + 1):
                try:
                    rp = _SESSION.get(link, allow_redirects=True, timeout=(5, 60))
                    if "application/pdf" in rp.headers.get("content-type", ""):
                        path.write_bytes(rp.content)
                        return str(path)
                except requests.RequestException as exc:
                    logger.debug("Publisher PDF error %s (try %d/%d)", exc, attempt, retries)
                _backoff(attempt)

    logger.debug("No PDF available for %s (PMC + publisher tried)", pmcid)
    return None



def download_pdf(pmcid: str, download_dir: str | Path = "/tmp", retries: int = 3) -> str | None:
    """
    Back‑compat wrapper: older pipeline code expects `download_pdf`, which is
    now renamed to `download_pmc_pdf`.  Delegates directly.
    """
    return download_pmc_pdf(pmcid, download_dir=download_dir, retries=retries)

__all__ = ["download_pmc_pdf", "download_pdf"]
