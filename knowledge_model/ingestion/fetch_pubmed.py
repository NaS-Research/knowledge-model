"""
PubMed ingestion utilities.

Downloads article metadata and abstracts via the NCBI E‑Utilities API, with
automatic chunking to avoid 414 (URI too long) errors and polite throttling to
respect the three‑requests‑per‑second rule.

Public API
----------
fetch_articles(query: str,
               *,
               max_results: int | None = None,
               summary_chunk_size: int = 200) -> list[dict]
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Iterable, List, Optional

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") or None
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_PAGE_SIZE = 500
if PUBMED_API_KEY:
    REQ_SLEEP_SEC = 0.12
else:
    REQ_SLEEP_SEC = 0.50

logger = logging.getLogger(__name__)


def _chunk(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield *size*-sized lists from *iterable*."""
    buf: list[str] = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def _esearch(query: str, retstart: int) -> list[str]:
    """
    Call NCBI ESearch, retrying on read‑timeouts or transient HTTP errors.
    Returns a list of PMIDs (may be empty on final failure).
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": SEARCH_PAGE_SIZE,
        "retstart": retstart,
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }
    for attempt in range(1, 4):
        try:
            r = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=params, timeout=60)
            r.raise_for_status()
            return r.json().get("esearchresult", {}).get("idlist", [])
        except (requests.Timeout, requests.exceptions.ReadTimeout) as err:
            if attempt == 3:
                logger.error("ESearch timeout after %d retries (retstart=%d) — %s", 3, retstart, err)
                return []
            logger.debug("ESearch retry %d/3 (retstart=%d)…", attempt, retstart)
            time.sleep(2 * attempt)
        except requests.HTTPError as err:
            status = err.response.status_code if err.response else "n/a"
            if status in {500, 502, 503, 504} and attempt < 3:
                logger.debug("ESearch HTTP %s retry %d/3 (retstart=%d)", status, attempt, retstart)
                time.sleep(2 * attempt)
            else:
                logger.error("ESearch HTTP error %s (retstart=%d) — aborting page", status, retstart)
                return []
    return []


def _esummary(pmids: list[str], retries: int = 3, timeout: int = 60) -> dict[str, Any]:
    """
    Call NCBI ESummary for a list of PMIDs.

    Retries up to *retries* times on network‑level problems (timeouts,
    connection errors, 5xx HTTP).  Returns an empty dict on final failure.
    """
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=params, timeout=timeout)
            r.raise_for_status()
            break
        except (requests.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.HTTPError) as err:
            if attempt == retries:
                logger.warning("ESummary failed for %d PMIDs after %d tries — %s", len(pmids), retries, err)
                return {}
            logger.debug("ESummary retry %d/%d for %d PMIDs", attempt, retries, len(pmids))
            time.sleep(2 * attempt)

    time.sleep(REQ_SLEEP_SEC)
    return r.json().get("result", {})


def _efetch_abstract(pmid: str, retries: int = 3, timeout: int = 60) -> str:
    """
    Retrieve the PubMed abstract for *pmid*.
    Retries up to *retries* times on network timeouts (HTTP 408 / read timeout).
    Returns empty string on failure.
    """
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "api_key": PUBMED_API_KEY}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=timeout)
            r.raise_for_status()
            break
        except (requests.Timeout, requests.exceptions.ReadTimeout, requests.HTTPError) as err:
            if attempt == retries:
                logger.warning("EFetch timeout/HTTP error for PMID %s after %d tries — %s", pmid, retries, err)
                return ""
            logger.debug("EFetch retry %d/%d for PMID %s", attempt, retries, pmid)
            time.sleep(2 * attempt)
    time.sleep(REQ_SLEEP_SEC)

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        logger.warning("EFetch XML parse error for PMID %s — empty abstract returned", pmid)
        return ""

    bits = [n.text or "" for n in root.findall(".//AbstractText")]
    return " ".join(b.strip() for b in bits if b.strip())


def _efetch_pmc_fulltext(pmcid: str, retries: int = 3, timeout: int = 60) -> str:
    """
    Retrieve full‑text XML from PubMed Central and return joined paragraphs.

    Retries up to *retries* times on network or transient HTTP errors.
    Falls back to empty string if all attempts fail or XML cannot be parsed.
    """
    pmcid = (
        pmcid.replace("pmc-id:", "")
             .replace("PMC", "")
             .replace(" ", "")
             .replace("+", "")
             .split(";")[0]
             .strip()
    )
    pmcid = f"PMC{pmcid}" if not pmcid.upper().startswith("PMC") else pmcid

    params = {"db": "pmc", "id": pmcid, "retmode": "xml", "api_key": PUBMED_API_KEY}

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=timeout)
            r.raise_for_status()
            break 
        except (requests.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.HTTPError) as err:
            if attempt == retries:
                logger.warning("PMC fetch failed for %s after %d tries — %s", pmcid, retries, err)
                return ""
            logger.debug("PMC retry %d/%d for %s", attempt, retries, pmcid)
            time.sleep(2 * attempt)

    time.sleep(REQ_SLEEP_SEC)

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        logger.warning("PMC XML parse error for %s — returning empty text", pmcid)
        return ""

    paras = [
        "".join(p.itertext()).strip()
        for p in root.findall(".//body//p")
        if "".join(p.itertext()).strip()
    ]
    return "\n\n".join(paras)


def fetch_articles(
    query: str,
    *,
    max_results: Optional[int] = None,
    summary_chunk_size: int = 200,
) -> list[dict[str, Any]]:
    pmids: list[str] = []
    retstart = 0
    while True:
        page = _esearch(query, retstart)
        pmids.extend(page)
        logger.info(
            "ESearch page %d → %d IDs (total %d)",
            retstart // SEARCH_PAGE_SIZE,
            len(page),
            len(pmids),
        )

        if max_results and len(pmids) >= max_results:
            pmids = pmids[:max_results]
            break
        if len(page) < SEARCH_PAGE_SIZE:
            break

        retstart += SEARCH_PAGE_SIZE
        time.sleep(REQ_SLEEP_SEC)

    if not pmids:
        logger.warning("No PMIDs found for query: %s", query)
        return []

    logger.info("Total PMIDs collected: %d", len(pmids))

    articles: list[dict[str, Any]] = []
    for batch in tqdm(
        _chunk(pmids, summary_chunk_size),
        desc="ESummary batches",
        unit="batch",
        colour="cyan",
    ):
        summary = _esummary(batch)
        for uid in summary.get("uids", []):
            entry = summary.get(uid, {})
            id_map = {d["idtype"]: d["value"] for d in entry.get("articleids", [])}

            raw_pmcid = id_map.get("pmcid")
            pmcid = (
                raw_pmcid.replace("pmc-id:", "")
                         .split(";")[0]
                         .strip()
                if raw_pmcid else None
            )
            
            full_text = ""
            if pmcid:
                try:
                    full_text = _efetch_pmc_fulltext(pmcid)
                except (requests.exceptions.RequestException, ET.ParseError) as err:
                    logger.warning(
                        "PMC fetch failed for %s after retries — %s; falling back to abstract",
                        pmcid,
                        err,
                    )

            if full_text:
                text_body = full_text
                section = "FULL"
            else:
                try:
                    text_body = _efetch_abstract(uid)
                    section = "ABSTRACT"
                except requests.HTTPError as err:
                    logger.warning("EFetch failed for PMID %s — %s; skipping abstract", uid, err)
                    text_body = ""
                    section = "NONE"

            articles.append(
                {
                    "pmid": uid,
                    "pmcid": pmcid,
                    "doi": id_map.get("doi"),
                    "title": entry.get("title"),
                    "authors": [a["name"] for a in entry.get("authors", [])],
                    "journal": entry.get("fulljournalname"),
                    "pubdate": entry.get("pubdate"),
                    "section": section,
                    "text": text_body,
                }
            )

        tqdm.write(f"  ↳ accumulated articles: {len(articles)}")
        if max_results and len(articles) >= max_results:
            articles = articles[:max_results]
            break

        time.sleep(REQ_SLEEP_SEC)

    logger.info("Fetched %d complete articles", len(articles))
    return articles

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    demo_query = (
        '("2020/01/01"[PDAT] : "2020/01/31"[PDAT]) '
        "AND hasabstract[text] AND free full text[sb] "
        "AND (clinicaltrial[pt] OR review[pt] OR research-article[pt])"
    )
    arts = fetch_articles(demo_query, max_results=40, summary_chunk_size=10)
    print(
        f"\nSummary ▸ total={len(arts)}, "
        f"full={sum(a['section']=='FULL' for a in arts)}, "
        f"abstracts={sum(a['section']=='ABSTRACT' for a in arts)}"
    )
