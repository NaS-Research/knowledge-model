"""
Ingestion pipeline: fetch PubMed metadata, download open‑access PDFs, clean/
chunk text, persist to DB + local files, and upload a consolidated dataset to S3.
"""

from __future__ import annotations

import argparse
import calendar
from knowledge_model.ingestion import json
import logging
import os
import re
import sys
from tqdm import tqdm
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

PROGRESS_EVERY: int = int(os.getenv("PIPELINE_PROGRESS_EVERY", "10"))
tqdm_kwargs = {"mininterval": 1.0, "unit_scale": True}

from knowledge_model.db.db_session import SessionLocal
from knowledge_model.db.sql_models import Article, ArticleChunk
import asyncio
from knowledge_model.ingestion.pdf_async import fetch_pdfs_async
from knowledge_model.ingestion.fetch_pubmed import fetch_articles, _efetch_abstract 
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
    filters = "hasabstract[text]"
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
    """
    Write every *chunk* to (a) the global training file and (b) a
    per‑article file under data/clean/YYYY/MM/.

    We open each target file **once** with a 4 MiB buffer to avoid the
    thousands of open/close syscalls that were previously killing
    throughput.
    """
    month_dir = CLEAN_ROOT / year / month
    month_dir.mkdir(parents=True, exist_ok=True)

    base = f"{pmid}_{art_id}"
    if any(p.name.startswith(base) for p in month_dir.glob("*.jsonl")):
        logger.info("Skip duplicate chunk write for %s", base)
        return

    buffer_size = 4 * 1024 * 1024  # 4 MiB OS‑level buffer
    train_path = TRAIN_FILE
    article_path = month_dir / f"{base}.jsonl"

    with train_path.open("ab", buffering=buffer_size) as train_fh, \
         article_path.open("ab", buffering=buffer_size) as art_fh:
        for text in chunks:
            record = {"pmid": pmid, "title": title, "text": text}
            line: bytes = json.dumps(record) + b"\n"  # orjson -> bytes
            train_fh.write(line)
            art_fh.write(line)


def run_pipeline(query: str, *, chunk_size: int = 1_000) -> None:
    logger.info("Fetching articles for query: %s", query)
    articles = fetch_articles(query)
    # ------------------------------------------------------------------
    # Pre‑fetch PDFs for all articles that have a PMCID.  We run this once
    # so downloads happen in parallel instead of one‑by‑one inside the loop.
    # ------------------------------------------------------------------
    pmcids = {
        (art.get("pmcid") or "")
        .replace("pmc-id:", "")
        .split(";")[0]
        .strip()
        for art in articles
        if art.get("pmcid")
    }
    logger.info("Downloading %d PDFs asynchronously …", len(pmcids))
    pdf_map = asyncio.run(fetch_pdfs_async(list(pmcids)))

    logger.info("Fetched %d articles", len(articles))

    try:
        year, month = re.search(r'"(\d{4})/(\d{2})/01"\[PDAT]', query).groups()
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
            # Grab PMC ID (if present) — skip download when article is not in PMC
            pmcid_raw = art.get("pmcid") or ""
            pmcid = (
                pmcid_raw.replace("pmc-id:", "")
                         .split(";")[0]
                         .strip()
            )
            doi = art.get("doi")
            pubdate = clean_text(art.get("pubdate") or "") or None

            existing = db.query(Article).filter_by(pmid=pmid).first()
            title = clean_text(art.get("title") or "Untitled")
            authors = ", ".join(art.get("authors", []))
            journal = clean_text(art.get("journal") or "") or None
            raw_text = clean_text(art.get("text") or "")
            section_label = art.get("section", "UNKNOWN")

            pdf_url: str | None = None
            chunks: list[str] = []
            downloaded = False
            abstract_text = ""

            # ------------------------------------------------------------------
            # If we already received full‑text XML from EFetch (`section == "FULL"`),
            # skip the PDF step entirely – the XML body is good enough.
            # ------------------------------------------------------------------
            if section_label == "FULL" and raw_text:
                chunks = chunk_text(raw_text, chunk_size)
                stats["chunks"] += len(chunks)
                logger.debug(
                    "Skipped PDF download for %s – full text already present in XML",
                    pmid,
                )
            else:
                pdf_path = None
                if pmcid:
                    pdf_path = pdf_map.get(pmcid)
                    if pdf_path and not (existing and existing.pdf_downloaded):
                        stats["pmc"] += 1
                        try:
                            parsed = parse_pdf(pdf_path)
                            # abstract
                            try:
                                abstract_text = clean_text(_efetch_abstract(pmid))
                            except Exception:
                                abstract_text = ""

                            cleaned = clean_text(parsed["text"])
                            chunks = chunk_text(cleaned, chunk_size)

                            if abstract_text:
                                chunks.insert(0, abstract_text)

                            pdf_url = upload_dataset_to_s3(pdf_path)
                            os.remove(pdf_path)
                            downloaded = True
                            stats["pdf"] += 1
                            stats["chunks"] += len(chunks)
                            logger.info(
                                "Parsed %d chunks for PMCID %s", len(chunks), pmcid
                            )
                        except Exception as err:
                            logger.debug("PDF parse failed for %s: %s", pmcid, err)
                else:
                    if not pmcid:
                        logger.debug("PMID %s has no PMC entry — skipping PDF download", pmid)

            if not downloaded and raw_text:
                chunks = chunk_text(raw_text, chunk_size)
                stats["chunks"] += len(chunks)

            article = existing or Article(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                pubdate=pubdate,
                abstract=(
                    abstract_text
                    if downloaded and abstract_text
                    else (raw_text if section_label == "ABSTRACT" else None)
                ),
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

            # Persist any chunks (PDF full‑text or abstract‑only)
            if chunks:
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