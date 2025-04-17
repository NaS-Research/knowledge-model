"""
Ingestion pipeline: fetch PubMed metadata, download open‑access PDFs, clean/
chunk text, persist to DB + local files, and upload a consolidated dataset to S3.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import re
import sys
from tqdm import tqdm  # progress bar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

# progress‑log frequency (default 10 if not set in .env)
PROGRESS_EVERY: int = int(os.getenv("PIPELINE_PROGRESS_EVERY", "10"))
# usage comment not needed
tqdm_kwargs = {"mininterval": 1.0, "unit_scale": True}

from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article, ArticleChunk
from knowledge_model.ingestion.download_pdf import download_pmc_pdf
from knowledge_model.ingestion.fetch_pubmed import fetch_articles
from knowledge_model.ingestion.parse_pdfs import parse_pdf
from knowledge_model.ingestion.upload_s3 import upload_dataset_to_s3
from knowledge_model.processing.text_cleaner import clean_text, chunk_text

logger = logging.getLogger(__name__)


TRAIN_FILE = Path("data/science_articles/NaS.jsonl")
CLEAN_ROOT = Path("data/clean")
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)


def _month_query(year: str, month: str) -> str:
    last_day = calendar.monthrange(int(year), int(month))[1]
    start = f'"{year}/{month}/01"[PDAT]'
    end = f'"{year}/{month}/{last_day:02d}"[PDAT]'
    filters = "hasabstract[text] AND free full text[sb]"
    types = "(clinicaltrial[pt] OR review[pt] OR research-article[pt])"
    return f"({start} : {end}) AND {filters} AND {types}"


def _write_chunks(
    pmid: str,
    art_id: int,
    title: str,
    chunks: List[str],
    year: str,
    month: str,
) -> None:
    month_dir = CLEAN_ROOT / year / month
    month_dir.mkdir(parents=True, exist_ok=True)
    base = f"{pmid}_{art_id}"
    if any(p.name.startswith(base) for p in month_dir.glob("*.jsonl")):
        logger.info("Skip duplicate chunk write for %s", base)
        return
    with TRAIN_FILE.open("a", encoding="utf-8") as train_f, \
            (month_dir / f"{base}.jsonl").open("a", encoding="utf-8") as clean_f:
        for text in chunks:
            rec = {"pmid": pmid, "title": title, "text": text}
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            train_f.write(line)
            clean_f.write(line)


def run_pipeline(query: str, *, chunk_size: int = 1_000) -> None:
    logger.info("Fetching articles for query: %s", query)
    articles = fetch_articles(query)
    logger.info("Fetched %d articles", len(articles))

    try:
        year, month = re.search(r'"(\d{4})/(\d{2})/01"\[PDAT]', query).groups()  # type: ignore
    except AttributeError:
        logger.error("Could not extract year/month from query")
        sys.exit(1)

    db = SessionLocal()
    stats = {"pmc": 0, "pdf": 0, "chunks": 0}

    try:
        for i, art in enumerate(
            tqdm(articles, desc="Processing", unit="article", **tqdm_kwargs), start=1
        ):
            pmid = art["pmid"]
            pmcid = (art.get("pmcid") or "").replace("pmc-id:", "").split(";")[0].strip()
            doi = art.get("doi")
            pubdate = clean_text(art.get("pubdate") or "") or None

            existing = db.query(Article).filter_by(pmid=pmid).first()
            title = clean_text(art.get("title") or "Untitled")
            authors = ", ".join(art.get("authors", []))
            journal = clean_text(art.get("journal") or "") or None
            abstract = clean_text(art.get("abstract") or "") or None

            pdf_url: str | None = None
            chunks: list[str] = []
            downloaded = False

            if pmcid and not (existing and existing.pdf_downloaded):
                stats["pmc"] += 1
                try:
                    pdf_path = download_pmc_pdf(pmcid)
                    parsed = parse_pdf(pdf_path)
                    cleaned = clean_text(parsed["text"])
                    chunks = chunk_text(cleaned, chunk_size)
                    pdf_url = upload_dataset_to_s3(pdf_path)
                    os.remove(pdf_path)
                    downloaded = True
                    stats["pdf"] += 1
                    stats["chunks"] += len(chunks)
                    logger.info("Parsed %d chunks for PMCID %s", len(chunks), pmcid)
                except Exception as err:
                    logger.warning("PDF failed for %s (%s): %s", pmid, pmcid, err)

            article = existing or Article(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                pubdate=pubdate,
                abstract=abstract,
                pdf_s3_url=pdf_url,
                doi=doi,
                pdf_downloaded=downloaded,
                content=None,
            )
            db.add(article)
            db.commit()

            # periodic progress log
            if i % PROGRESS_EVERY == 0:
                logger.info("Processed %d / %d articles so far", i, len(articles))

            if downloaded and chunks:
                for chunk_idx, txt in enumerate(chunks):
                    db.add(ArticleChunk(article_id=article.id, chunk_index=chunk_idx, chunk_text=txt))
                db.commit()
                _write_chunks(pmid, article.id, title, chunks, year, month)


        logger.info(
            "Summary – total: %d | with PMCID: %d | PDFs: %d | chunks: %d",
            len(articles), stats["pmc"], stats["pdf"], stats["chunks"],
        )

        if TRAIN_FILE.exists() and TRAIN_FILE.stat().st_size:
            logger.info("Uploading %s to S3…", TRAIN_FILE.name)
            logger.info("Dataset URL: %s", upload_dataset_to_s3(TRAIN_FILE))
        else:
            logger.warning("No dataset written; skipping upload")

    except Exception as err:  # pragma: no cover
        logger.exception("Pipeline error: %s", err)
        db.rollback()
    finally:
        db.close()


def test_open_access() -> None:
    """
    Minimal smoke‑test: ingest a single known open‑access article to verify that
    PDF download, parsing, cleaning, and DB writes are all working.
    """
    logger.info("Running open‑access smoke test…")
    run_pipeline(
        query='"2020/06/01"[PDAT] : "2020/06/01"[PDAT] AND hasabstract[text] AND free full text[sb]',
        chunk_size=1000,
    )
    logger.info("Smoke test complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run PubMed ingestion pipeline.")
    parser.add_argument("year", nargs="?", help="YYYY (default UTC year)")
    parser.add_argument("month", nargs="?", help="MM   (default UTC month)")
    parser.add_argument(
        "--test_oa",
        action="store_true",
        help="Run a one‑article open‑access smoke test and exit",
    )
    args = parser.parse_args()

    if args.test_oa:
        test_open_access()
        return

    now = datetime.now(UTC)
    year = args.year or f"{now.year:04d}"
    month = args.month or f"{now.month:02d}"

    run_pipeline(_month_query(year, month))


if __name__ == "__main__":
    main()
