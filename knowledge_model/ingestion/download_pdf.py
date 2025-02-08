"""
download_pdf.py
Downloads an open-access PDF from PubMed Central if a PMCID is available.
"""

import os
import requests
import logging

logger = logging.getLogger(__name__)

def download_pmc_pdf(pmcid: str, download_dir: str = "/tmp") -> str:
    """
    Attempt to download the PDF from PMC for an open-access article.
    Returns the local file path or raises an exception on failure.
    """

    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    local_path = os.path.join(download_dir, f"{pmcid}.pdf")

    logger.info("Downloading PDF from %s", pdf_url)

    # Add a User-Agent header to avoid potential bot blocking.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    resp = requests.get(pdf_url, headers=headers)
    if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    else:
        msg = f"Failed to download PDF for {pmcid} (status {resp.status_code})."
        logger.warning(msg)
        raise ValueError(msg)
