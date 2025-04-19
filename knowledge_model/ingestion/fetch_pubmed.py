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

# --------------------------------------------------------------------------- #
# constants & logger
# --------------------------------------------------------------------------- #
load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY") or None
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_PAGE_SIZE = 500        # IDs returned per ESearch page
REQ_SLEEP_SEC = 0.34          # ≤ 3 requests / second (NCBI policy)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": SEARCH_PAGE_SIZE,
        "retstart": retstart,
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }
    r = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _esummary(pmids: list[str]) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }
    r = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("result", {})


def _efetch_abstract(pmid: str) -> str:
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "api_key": PUBMED_API_KEY}
    r = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    time.sleep(REQ_SLEEP_SEC)

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        logger.warning("EFetch XML parse error for PMID %s — empty abstract returned", pmid)
        return ""

    bits = [n.text or "" for n in root.findall(".//AbstractText")]
    return " ".join(b.strip() for b in bits if b.strip())


def _efetch_pmc_fulltext(pmcid: str) -> str:
    """
    Download full‑text XML from PubMed Central and return joined paragraphs.
    Falls back to empty string if parse fails.
    """
    params = {"db": "pmc", "id": pmcid, "retmode": "xml", "api_key": PUBMED_API_KEY}
    r = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=45)
    r.raise_for_status()
    time.sleep(REQ_SLEEP_SEC)

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        logger.warning("PMC XML parse error for %s — returning empty text", pmcid)
        return ""

    # collect paragraph‑level text inside <body>
    paras = [
        "".join(p.itertext()).strip()
        for p in root.findall(".//body//p")
        if "".join(p.itertext()).strip()
    ]
    return "\n\n".join(paras)


# --------------------------------------------------------------------------- #
# public function
# --------------------------------------------------------------------------- #
def fetch_articles(
    query: str,
    *,
    max_results: Optional[int] = None,
    summary_chunk_size: int = 200,
) -> list[dict[str, Any]]:
    # 1 ▸ collect PMIDs ------------------------------------------------------ #
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

    # 2 ▸ fetch metadata + abstract per chunk -------------------------------- #
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

            pmcid = id_map.get("pmcid")
            if pmcid:
                full_text = _efetch_pmc_fulltext(pmcid)
                text_body = full_text if full_text else _efetch_abstract(uid)
                section = "FULL" if full_text else "ABSTRACT"
            else:
                text_body = _efetch_abstract(uid)
                section = "ABSTRACT"

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


# --------------------------------------------------------------------------- #
# standalone test
# --------------------------------------------------------------------------- #
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