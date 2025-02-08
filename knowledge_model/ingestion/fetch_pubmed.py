"""
fetch_pubmed.py
Fetches article metadata + abstracts from PubMed, including PMCID if available.
"""

import os
import time
import requests
import logging
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve the PubMed API key from .env
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

logger = logging.getLogger(__name__)
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def fetch_articles(query: str = "cancer immunotherapy", max_results: int = 5) -> list:
    # ESearch: Build the search parameters including the API key if available.
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    if PUBMED_API_KEY:
        search_params["api_key"] = PUBMED_API_KEY

    r_search = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=search_params)
    r_search.raise_for_status()
    s_data = r_search.json()

    pmid_list = s_data.get("esearchresult", {}).get("idlist", [])
    logger.info("Found %d PMIDs for '%s'", len(pmid_list), query)
    if not pmid_list:
        return []

    # ESummary: Retrieve additional metadata for the found PMIDs.
    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "json"
    }
    if PUBMED_API_KEY:
        summary_params["api_key"] = PUBMED_API_KEY

    r_summary = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=summary_params)
    r_summary.raise_for_status()
    sum_data = r_summary.json()

    articles = []
    uids = sum_data.get("result", {}).get("uids", [])
    for uid in uids:
        info = sum_data["result"].get(uid, {})
        pmcid = None
        for aid in info.get("articleids", []):
            if aid.get("idtype") == "pmcid":
                pmcid = aid.get("value")
                break

        art = {
            "pmid": uid,
            "pmcid": pmcid,
            "title": info.get("title"),
            "authors": [au.get("name") for au in info.get("authors", [])],
            "journal": info.get("fulljournalname"),
            "pubdate": info.get("pubdate"),
            "abstract": ""
        }
        # EFetch to get the full abstract
        art["abstract"] = _fetch_abstract(uid)
        articles.append(art)

    logger.info("Successfully fetched %d articles", len(articles))
    return articles

def _fetch_abstract(pmid: str) -> str:
    # Build parameters for efetch including the API key if available.
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    r = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params)
    r.raise_for_status()

    # Throttle requests to approximately 10 per second
    time.sleep(0.1)

    root = ET.fromstring(r.text)
    nodes = root.findall(".//AbstractText")
    if not nodes:
        return ""
    return " ".join((node.text or "").strip() for node in nodes)

def main():
    logging.basicConfig(level=logging.INFO)
    articles = fetch_articles(query="machine learning in cancer", max_results=3)
    for i, art in enumerate(articles, start=1):
        print(f"{i}. PMID: {art['pmid']} | PMCID: {art['pmcid']} | Title: {art['title']}")
        print(f"Abstract (truncated): {art['abstract'][:100]}...")

if __name__ == "__main__":
    main()
