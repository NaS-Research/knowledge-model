"""
fetch_pubmed.py
---------------
Minimal script to query PubMed using the E-utilities API,
returning a list of article metadata (title, authors, etc.).
"""

import requests
import logging

logger = logging.getLogger(__name__)

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_articles(query: str = "cancer immunotherapy", max_results: int = 5) -> list:
    """
    Fetch basic PubMed article metadata given a search query.
    
    :param query: Search term (e.g., "cancer immunotherapy")
    :param max_results: Maximum number of articles to fetch
    :return: A list of dicts containing article metadata (pmid, title, authors, etc.)
    """

    # 1. ESearch: Get a list of PubMed IDs (PMIDs) matching the query
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    search_resp = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=search_params)
    search_resp.raise_for_status()
    search_data = search_resp.json()

    pmid_list = search_data.get("esearchresult", {}).get("idlist", [])
    logger.info("Found %d PMIDs for query '%s'", len(pmid_list), query)
    if not pmid_list:
        return []

    # 2. ESummary: Get summary info (title, authors, etc.) for each PMID
    summary_params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "json"
    }
    summary_resp = requests.get(f"{EUTILS_BASE_URL}/esummary.fcgi", params=summary_params)
    summary_resp.raise_for_status()
    summary_data = summary_resp.json()

    # 3. Parse out relevant metadata
    result_list = []
    uids = summary_data.get("result", {}).get("uids", [])
    for uid in uids:
        article_info = summary_data["result"].get(uid, {})
        article_dict = {
            "pmid": uid,
            "title": article_info.get("title"),
            "authors": [au.get("name") for au in article_info.get("authors", [])],
            "journal": article_info.get("fulljournalname"),
            "pubdate": article_info.get("pubdate"),
        }
        result_list.append(article_dict)
        logger.debug("Fetched article: %s", article_dict["title"])

    logger.info("Successfully fetched %d articles", len(result_list))
    return result_list


def main():
    """
    For quick testing: 
    python -m knowledge_model.ingestion.fetch_pubmed
    """
    logging.basicConfig(level=logging.INFO)
    articles = fetch_articles(query="machine learning in cancer", max_results=3)
    for idx, art in enumerate(articles, start=1):
        print(f"{idx}. PMID: {art['pmid']}, Title: {art['title']}")


if __name__ == "__main__":
    main()
