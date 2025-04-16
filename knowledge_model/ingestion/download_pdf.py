"""
Downloads an open-access PDF from PubMed Central given a PMCID.
"""

import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def download_pmc_pdf(pmcid: str, download_dir: str = "/tmp") -> str:
    """
    Download a PDF from PMC for an open-access article.

    Args:
        pmcid: The PMC identifier (e.g. "PMC123456").
        download_dir: The local directory to save the PDF.

    Returns:
        The local file path of the downloaded PDF.

    Raises:
        ValueError: If the PDF cannot be downloaded successfully.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    path = Path(download_dir) / f"{pmcid}.pdf"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    logger.info("Downloading PDF from %s", url)
    response = requests.get(url, headers=headers)

    if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
        path.write_bytes(response.content)
        return str(path)

    msg = f"Failed to download PDF for {pmcid} (status {response.status_code})."
    logger.warning(msg)
    raise ValueError(msg)
