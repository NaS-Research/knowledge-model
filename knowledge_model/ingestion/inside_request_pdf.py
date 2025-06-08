from httpx import RemoteProtocolError, HTTPError

async def _request_pdf(url, client, attempt):
    try:
        head = await client.head(url, follow_redirects=True, timeout=20)
        ...
    except (RemoteProtocolError, HTTPError) as err:
        logger.debug("PDF head failed (%s) attempt %d – %s", url, attempt, err)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(base_backoff(attempt))
            return await _request_pdf(url, client, attempt + 1)
        return None        # let caller treat as “no PDF”
"""
inside_request_pdf.py
---------------------
Utility function used by `pdf_async.py` to retrieve a PDF (or decide that no
PDF is available) with retry + exponential‑jitter back‑off.

A *single* network hiccup or non‑PDF response should not abort the whole batch:
we retry a few times; after that the caller interprets `None` as
“no PDF fetched”.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import httpx
from httpx import AsyncClient, HTTPError, RemoteProtocolError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_RETRIES = 4           # total attempts (1 initial + 3 retries)
BASE_DELAY = 0.5          # seconds – first back‑off delay
TIMEOUT_HEAD = 20.0       # seconds for `HEAD`
TIMEOUT_GET = 60.0        # seconds for the final `GET`

logger = logging.getLogger(__name__)


def _backoff_delay(attempt: int) -> float:
    """
    Exponential back‑off with ±300 ms jitter.

    attempt 1 → ~0.5 s  
    attempt 2 → ~1.0 s  
    attempt 3 → ~2.0 s …
    """
    jitter = random.uniform(0, 0.3)
    return BASE_DELAY * (2 ** (attempt - 1)) + jitter


# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #
async def _request_pdf(
    url: str,
    client: AsyncClient,
    attempt: int = 1,
) -> Optional[bytes]:
    """
    Try to download *url* and return its bytes if it is a PDF.

    Returns
    -------
    bytes | None
        PDF payload, or ``None`` if all retries fail / content is not PDF.
    """
    try:
        # 1) Lightweight HEAD probe so we don’t pull huge HTML by mistake
        head = await client.head(url, follow_redirects=True, timeout=TIMEOUT_HEAD)
        ctype = head.headers.get("content-type", "").lower()

        if head.status_code != 200 or "pdf" not in ctype:
            raise ValueError(f"non‑PDF response (status={head.status_code}, type={ctype})")

        # 2) Fetch the actual document
        r = await client.get(url, follow_redirects=True, timeout=TIMEOUT_GET)
        r.raise_for_status()
        return r.content

    except (RemoteProtocolError, HTTPError, ValueError) as err:
        logger.debug("PDF fetch failed (%s) attempt %d – %s", url, attempt, err)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(_backoff_delay(attempt))
            return await _request_pdf(url, client, attempt + 1)

        # Out of retries → caller decides how to handle None
        logger.debug("Giving up on %s after %d attempts", url, attempt)
        return None