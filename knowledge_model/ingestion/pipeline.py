"""
Ingestion pipeline: fetch PubMed metadata, download open‑access PDFs, clean/
split into retrieval passages, persist to DB + local files, and upload a consolidated dataset to S3.

After building the FAISS index locally, the pipeline now uploads
`faiss.idx` and `passages.jsonl` to an S3 bucket (default
`nas-faiss-index/faiss`) so the FastAPI runtime can download them at
startup.
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
from knowledge_model.ingestion.upload_s3 import upload_directory
from knowledge_model.processing.text_cleaner import clean_text

from knowledge_model.ingestion.build_faiss import build_faiss_index as build_index

logger = logging.getLogger(__name__)



from knowledge_model.config.settings import DATA_ROOT

CORPUS_ROOT = DATA_ROOT / "corpus"
RAW_ROOT    = CORPUS_ROOT / "raw"
CLEAN_ROOT  = CORPUS_ROOT / "clean"
CORPUS_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = CORPUS_ROOT / "science_articles" / "NaS.jsonl"
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

RAW_ROOT.mkdir(parents=True, exist_ok=True)


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
    """Append cleaned chunks to the master JSONL and a per‑article file."""
    month_dir = CLEAN_ROOT / year / month
    month_dir.mkdir(parents=True, exist_ok=True)

    base = f"{pmid}_{art_id}"
    if any(p.name.startswith(base) for p in month_dir.glob("*.jsonl")):
        logger.info("Skip duplicate chunk write for %s", base)
        return

    buffer_size = 4 * 1024 * 1024
    train_path = TRAIN_FILE
    article_path = month_dir / f"{base}.jsonl"

    with train_path.open("ab", buffering=buffer_size) as train_fh, \
         article_path.open("ab", buffering=buffer_size) as art_fh:
        for text in chunks:
            record = {"pmid": pmid, "title": title, "text": text}
            line: bytes = json.dumps(record) + b"\n"
            train_fh.write(line)
            art_fh.write(line)


def run_pipeline(query: str, *, chunk_size: int = 1_000) -> None:
    logger.info("Fetching articles for query: %s", query)
    articles = fetch_articles(query)
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
            passages: list[str] = []
            downloaded = False
            abstract_text = ""

            if section_label == "FULL" and raw_text:
                passages = [raw_text]
                stats["chunks"] += 1
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
                        passages = []
                        if pmcid:
                            passages_dicts = parse_pdf(pdf_path)  # returns list[dict]
                            passages = [d["text"] for d in passages_dicts]
                            if passages:
                                stats["chunks"] += len(passages)
                        pdf_url = upload_dataset_to_s3(pdf_path)
                        os.remove(pdf_path)
                        downloaded = True
                        stats["pdf"] += 1
                        logger.info(
                            "Parsed %d passages for PMCID %s", len(passages), pmcid
                        )
                else:
                    if not pmcid:
                        logger.debug("PMID %s has no PMC entry — skipping PDF download", pmid)

            if not passages and raw_text:
                passages = [raw_text]  # will be split later for FAISS
                stats["chunks"] += 1

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

            if i % PROGRESS_EVERY == 0:
                logger.info("Processed %d / %d articles so far", i, len(articles))

            if passages:
                for chunk_idx, txt in enumerate(passages):
                    db.add(ArticleChunk(article_id=article.id, chunk_index=chunk_idx, chunk_text=txt))
                db.commit()
                _write_chunks(pmid, article.id, title, passages, year, month)


        logger.info(
            "Summary – total: %d | with PMCID: %d | PDFs: %d | chunks: %d",
            len(articles), stats["pmc"], stats["pdf"], stats["chunks"],
        )

        if TRAIN_FILE.exists() and TRAIN_FILE.stat().st_size:
            logger.info("Uploading %s to S3…", TRAIN_FILE.name)
            logger.info("Dataset URL: %s", upload_dataset_to_s3(TRAIN_FILE))

            # Re‑build / refresh the local FAISS index that the API expects.
            faiss_out = DATA_ROOT / "faiss"
            build_index(
                CLEAN_ROOT,          # directory with cleaned JSONL chunks
                faiss_out,           # output directory for FAISS index
                "all-MiniLM-L6-v2",  # sentence‑transformer model
            )
            logger.info("FAISS index rebuilt at %s", faiss_out / 'faiss.idx')

            # Upload faiss.idx and passages.jsonl to S3 so the API can pull them
            FAISS_BUCKET = os.getenv("FAISS_S3_BUCKET", "nas-faiss-index")
            FAISS_PREFIX = os.getenv("FAISS_S3_PREFIX_PATH", "faiss")
            uploaded = upload_directory(
                faiss_out,
                bucket=FAISS_BUCKET,
                prefix=FAISS_PREFIX,
                recurse=False,
            )
            for url in uploaded:
                logger.info("Uploaded %s", url)
            ADAPTER_DIR = Path("adapters/txgemma_lora_instr_v1")
            if ADAPTER_DIR.exists():
                ADAPTER_BUCKET = os.getenv("ADAPTER_S3_BUCKET", "nas-lora")
                ADAPTER_PREFIX = os.getenv("ADAPTER_S3_PREFIX_PATH", "txgemma_lora_instr_v1")
                logger.info("Uploading LoRA adapter to s3://%s/%s …", ADAPTER_BUCKET, ADAPTER_PREFIX)
                uploaded = upload_directory(
                    ADAPTER_DIR,
                    bucket=ADAPTER_BUCKET,
                    prefix=ADAPTER_PREFIX,
                    recurse=True,
                )
                for url in uploaded:
                    logger.info("Uploaded %s", url)
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

    start_ts = datetime.now(UTC)
    logger.info(
        "Pipeline started for %s‑%s at %s",
        year,
        month,
        start_ts.strftime("%Y‑%m‑%d %H:%M:%S %Z"),
    )

    run_pipeline(_month_query(year, month))

    end_ts = datetime.now(UTC)
    elapsed = end_ts - start_ts
    logger.info(
        "Pipeline finished in %s (ended %s)",
        str(elapsed).split(".")[0],
        end_ts.strftime("%Y‑%m‑%d %H:%M:%S %Z"),
    )


if __name__ == "__main__":
    main()