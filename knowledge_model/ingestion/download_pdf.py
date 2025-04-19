"""
Downloads an open-access PDF from PubMed Central given a PMCID.
"""

import logging
 
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
from time import sleep
from typing import Optional

_SESSION: Optional[requests.Session] = None
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds


def download_pmc_pdf(pmcid: str, download_dir: str = "/tmp") -> str:
    """
    Download an open‑access PDF from PubMed Central.
    Skips download if the file already exists.
    
    Parameters
    ----------
    pmcid : str
        The PMC identifier (e.g. "PMC123456").
    download_dir : str | Path
        Destination directory (will be created).
    
    Returns
    -------
    str
        Local file path of the PDF.
    
    Raises
    ------
    ValueError
        If the PDF could not be downloaded after retries.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    download_dir = Path(download_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / f"{pmcid}.pdf"
    
    # Skip if already on disk and >25 kB (guards against earlier bad fetch)
    if path.exists() and path.stat().st_size > 25_000:
        logger.info("PDF already cached – %s", path)
        return str(path)
    
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PMC-DL/1.0; +https://example.org)"
                )
            }
        )
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", pmcid, attempt, MAX_RETRIES)
            r = _SESSION.get(url, stream=True, timeout=(5, 60))
            r.raise_for_status()
    
            if "application/pdf" not in r.headers.get("Content-Type", ""):
                raise ValueError("Content-Type is not PDF")
    
            # Stream to disk in 1 MiB chunks
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1_048_576):
                    if chunk:
                        f.write(chunk)
    
            if path.stat().st_size < 12_000:
                raise ValueError("Downloaded PDF is suspiciously small")
    
            return str(path)
    
        except Exception as exc:
            logger.warning("Download failed: %s", exc)
            if attempt < MAX_RETRIES:
                sleep(RETRY_BACKOFF * attempt)
            else:
                if path.exists():
                    path.unlink(missing_ok=True)
                raise ValueError(f"Failed to download PDF for {pmcid}") from exc
