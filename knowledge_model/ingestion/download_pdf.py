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
    resp = requests.get(pdf_url)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/pdf"):
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    else:
        msg = f"Failed to download PDF for {pmcid} (status {resp.status_code})."
        logger.warning(msg)
        raise ValueError(msg)
