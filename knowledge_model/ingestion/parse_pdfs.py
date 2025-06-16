"""
Batch‑convert PDFs in a source directory to cleaned UTF‑8 text files.

Usage
-----
python -m knowledge_model.ingestion.parse_pdfs \
  --src data/corpus/raw \
  --dst data/corpus/clean_jsonl
"""

import logging
from typing import Any

import fitz
import re
import argparse
import pathlib
from tqdm import tqdm
import json
from knowledge_model.processing.text_cleaner import strip_boiler  # type: ignore
from knowledge_model.processing.passage_splitter import split_passages  # type: ignore

_HEADER_FOOTER_RE = re.compile(r"^(?:Page \d+(?: of \d+)?|©.*|Copyright.*|All rights reserved\.?)$", re.IGNORECASE)
_DIGIT_LINE_RE = re.compile(r"^\s*\d+\s*$")

logger = logging.getLogger(__name__)


# Helper to join hyphenated lines
_HYPHEN_NL_RE = re.compile(r"([A-Za-z0-9])-\n([A-Za-z])")

def _join_hyphenated_lines(text: str) -> str:
    """
    Join words that have been split across a line‑break with a trailing hyphen,
    e.g. ``transfor-\nmation`` → ``transformation``.
    """
    # Repeat replacement until no more patterns are found (handles cascades).
    while True:
        new = _HYPHEN_NL_RE.sub(r"\1\2", text)
        if new == text:
            return new
        text = new

def _strip_page_artifacts(text: str) -> str:
    """Remove obvious page numbers / headers / footers and collapse whitespace."""
    clean_lines: list[str] = []
    for line in text.splitlines():
        if _HEADER_FOOTER_RE.match(line) or _DIGIT_LINE_RE.match(line):
            continue
        clean_lines.append(line)
    cleaned = "\n".join(clean_lines)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = _join_hyphenated_lines(cleaned)
    return cleaned


def parse_pdf(pdf_path: str, passage_size: int = 300, passage_overlap: int = 50) -> list[dict[str, Any]]:
    """
    Extract plain text from a PDF file and split it into overlapping retrieval passages.

    Args:
        pdf_path:          Path to the local PDF file.
        passage_size:      Character length of each passage (default 300).
        passage_overlap:   Overlap between consecutive passages (default 50).

    Returns
    -------
    List[dict]
        One dict per passage with keys:
            - "file_path": absolute PDF path
            - "passage_id": running integer starting at 0
            - "text": cleaned passage text
    """
    try:
        with fitz.open(pdf_path) as doc:
            raw_pages = [page.get_text() for page in doc]
    except Exception as exc:
        logger.warning("Unable to open %s: %s", pdf_path, exc)
        raw_pages = []

    pages = [_strip_page_artifacts(p) for p in raw_pages]
    body  = strip_boiler("\n".join(pages))

    passages = split_passages(body, size=passage_size, overlap=passage_overlap)
    return [
        {"file_path": pdf_path, "passage_id": i, "text": p}
        for i, p in enumerate(passages)
    ]


def process_dir(src: pathlib.Path, dst: pathlib.Path, force: bool = False) -> None:
    pdf_paths = list(src.rglob("*.pdf"))
    logger.info("Parsing %d PDFs from %s → %s", len(pdf_paths), src, dst)
    for path in tqdm(pdf_paths, desc="Parsing PDFs"):
        rel = path.relative_to(src).with_suffix(".jsonl")
        out = dst / rel
        if out.exists() and not force:
            continue
        parsed_passages = parse_pdf(str(path))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in parsed_passages:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs into clean text.")
    parser.add_argument("--src", required=True, help="Directory containing PDFs")
    parser.add_argument("--dst", required=True, help="Output directory for .jsonl files")
    parser.add_argument("--force", action="store_true", help="Re‑parse PDFs even if the .txt already exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_dir(pathlib.Path(args.src), pathlib.Path(args.dst), force=args.force)
    logger.info("Finished parsing PDFs.")


if __name__ == "__main__":
    main()
