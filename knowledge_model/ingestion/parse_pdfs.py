"""
Batch‑convert PDFs in a source directory to cleaned UTF‑8 text files.

Usage
-----
python -m knowledge_model.ingestion.parse_pdfs \
  --src data/corpus/raw \
  --dst data/corpus/clean
"""

import logging
from typing import Any

import fitz
import re
import argparse
import pathlib
from tqdm import tqdm
from knowledge_model.processing.text_cleaner import strip_boiler  # type: ignore

_HEADER_FOOTER_RE = re.compile(r"^(?:Page \d+(?: of \d+)?|©.*|Copyright.*|All rights reserved\.?)$", re.IGNORECASE)
_DIGIT_LINE_RE = re.compile(r"^\s*\d+\s*$")

logger = logging.getLogger(__name__)

def _strip_page_artifacts(text: str) -> str:
    """Remove obvious page numbers / headers / footers and collapse whitespace."""
    clean_lines: list[str] = []
    for line in text.splitlines():
        if _HEADER_FOOTER_RE.match(line) or _DIGIT_LINE_RE.match(line):
            continue
        clean_lines.append(line)
    cleaned = "\n".join(clean_lines)
    return re.sub(r"[ \t]+\n", "\n", cleaned)


def parse_pdf(pdf_path: str) -> dict[str, Any]:
    """
    Extract plain text from a PDF file.

    Args:
        pdf_path: Path to the local PDF file.

    Returns:
        A dictionary with the file path and extracted text content.
    """
    try:
        with fitz.open(pdf_path) as doc:
            raw_pages = [page.get_text() for page in doc]
    except Exception as exc:
        logger.warning("Unable to open %s: %s", pdf_path, exc)
        raw_pages = []

    pages = [_strip_page_artifacts(p) for p in raw_pages]

    body = strip_boiler("\n".join(pages))
    return {
        "file_path": pdf_path,
        "text": body,
    }


def process_dir(src: pathlib.Path, dst: pathlib.Path) -> None:
    pdf_paths = list(src.rglob("*.pdf"))
    for path in tqdm(pdf_paths, desc="Parsing PDFs"):
        parsed = parse_pdf(str(path))
        rel   = path.relative_to(src).with_suffix(".txt")
        out   = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(parsed["text"], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs into clean text.")
    parser.add_argument("--src", required=True, help="Directory containing PDFs")
    parser.add_argument("--dst", required=True, help="Output directory for .txt files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_dir(pathlib.Path(args.src), pathlib.Path(args.dst))
    logger.info("Finished parsing PDFs.")


if __name__ == "__main__":
    main()
