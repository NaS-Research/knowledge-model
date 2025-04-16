"""
Fetches article metadata and abstracts from PubMed using E-Utilities.
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

logger = logging.getLogger(__name__)


def fetch_articles(query: str = "cancer immunotherapy", max_results: int = 5) -> list[dict[str, Any]]:
    """
    Fetch metadata and abstracts from PubMed based on a search query.

    Args:
        query: PubMed search string.
        max_results: Maximum number of articles to fetch.

    Returns:
        A list of article dictionaries with metadata and abstract.
    """
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }

    response = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=search_params)
    response.raise_for_status()
    pmids = response.json().get("esearchresult", {}).get("idlist", [])

    logger.info("Found %d PMIDs for query '%s'", len(pmids), query)
    if not pmids:
        return []

    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "api_key": PUBMED_API_KEY,
    }

    summary_response = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=summary_params)
    summary_response.raise_for_status()
    summary_data = summary_response.json()

    articles: list[dict[str, Any]] = []
    for uid in summary_data.get("result", {}).get("uids", []):
        entry = summary_data["result"].get(uid, {})
        pmcid = None
        doi = None
        for aid in entry.get("articleids", []):
            if aid.get("idtype") == "pmcid":
                pmcid = aid.get("value")
            elif aid.get("idtype") == "doi":
                doi = aid.get("value")

        article = {
            "pmid": uid,
            "pmcid": pmcid,
            "doi": doi,
            "title": entry.get("title"),
            "authors": [a.get("name") for a in entry.get("authors", [])],
            "journal": entry.get("fulljournalname"),
            "pubdate": entry.get("pubdate"),
            "abstract": fetch_abstract(uid),
        }
        articles.append(article)

    logger.info("Successfully fetched %d articles", len(articles))
    return articles


def fetch_abstract(pmid: str) -> str:
    """
    Fetch the abstract text for a given PubMed ID.

    Args:
        pmid: PubMed article ID.

    Returns:
        Abstract text or an empty string if not available.
    """
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "api_key": PUBMED_API_KEY,
    }

    response = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params)
    response.raise_for_status()
    time.sleep(0.1)  # Rate limiting

    root = ET.fromstring(response.text)
    nodes = root.findall(".//AbstractText")
    return " ".join((node.text or "").strip() for node in nodes) if nodes else ""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = fetch_articles(query="open access", max_results=3)
    for i, art in enumerate(results, start=1):
        print(f"{i}. PMID: {art['pmid']} | PMCID: {art['pmcid']} | DOI: {art['doi']}")
        print(f"    Title: {art['title']}")
        print(f"    Abstract: {art['abstract'][:120]}...")


if __name__ == "__main__":
    main()
