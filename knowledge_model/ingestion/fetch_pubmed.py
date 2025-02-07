"""
fetch_pubmed.py
Queries PubMed via E-utilities and returns article metadata, including PMCID if available.
"""

import requests
import logging

logger = logging.getLogger(__name__)

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def fetch_articles(query: str = "cancer immunotherapy", max_results: int = 5) -> list:
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    r_search = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=search_params)
    r_search.raise_for_status()
    s_data = r_search.json()

    pmid_list = s_data.get("esearchresult", {}).get("idlist", [])
    logger.info("Found %d PMIDs for query '%s'", len(pmid_list), query)
    if not pmid_list:
        return []

    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "json"
    }
    r_summary = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=summary_params)
    r_summary.raise_for_status()
    sum_data = r_summary.json()

    result_list = []
    uids = sum_data.get("result", {}).get("uids", [])
    for uid in uids:
        info = sum_data["result"].get(uid, {})
        # Check for PMCID in 'articleids'
        pmcid = None
        for aid in info.get("articleids", []):
            if aid.get("idtype") == "pmcid":
                pmcid = aid.get("value")
                break

        result_list.append({
            "pmid": uid,
            "pmcid": pmcid,
            "title": info.get("title"),
            "authors": [au.get("name") for au in info.get("authors", [])],
            "journal": info.get("fulljournalname"),
            "pubdate": info.get("pubdate"),
        })
        logger.debug("Fetched article: %s", info.get("title"))

    logger.info("Successfully fetched %d articles", len(result_list))
    return result_list

def main():
    logging.basicConfig(level=logging.INFO)
    articles = fetch_articles(query="machine learning in cancer", max_results=3)
    for i, art in enumerate(articles, start=1):
        print(f"{i}. PMID: {art['pmid']} | PMCID: {art['pmcid']} | Title: {art['title']}")

if __name__ == "__main__":
    main()
