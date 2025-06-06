from __future__ import annotations

import logging
import os
import random
import time
from typing import Final, Optional

import requests

logger = logging.getLogger(__name__)

_API_BASE: Final[str] = "https://api.unpaywall.org/v2"
_EMAIL: Final[str] = os.getenv("UNPAYWALL_EMAIL", "anonymous@example.com")

def pdf_url_from_doi(doi: str, retries: int = 3, session: Optional[requests.Session] = None) -> str | None:
    """
    Return the *best* open‑access PDF URL for *doi* if one exists.

    Parameters
    ----------
    doi : str
        Digital Object Identifier, e.g. ``10.1155/2013/485082``.
    retries : int, default 3
        Number of HTTP retries with exponential back‑off.
    session : requests.Session, optional
        Re‑use an existing session if you are calling in a loop.

    Returns
    -------
    str | None
        Direct URL of the PDF, or *None* if no OA copy is available.
    """
    if not doi:
        return None

    ses = session or requests.Session()
    url = f"{_API_BASE}/{doi}?email={_EMAIL}"

    for attempt in range(1, retries + 1):
        try:
            r = ses.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf")
            return pdf_url or None
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                logger.debug("Unpaywall lookup failed for DOI %s: %s", doi, exc)
                return None

            # Exponential back‑off with a bit of jitter (0.3 s max)
            delay = 0.5 * 2 ** (attempt - 1) + random.uniform(0, 0.3)
            time.sleep(delay)

    return None


__all__ = ["pdf_url_from_doi"]