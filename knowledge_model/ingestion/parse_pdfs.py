"""
Parses text from a PDF using PyMuPDF.
"""

import logging
from typing import Any

import fitz
import re

logger = logging.getLogger(__name__)

def _strip_header_footer(text: str) -> str:
    """
    Remove lines that are likely page numbers or running headers/footers.
    """
    clean_lines: list[str] = []
    for line in text.splitlines():
        if _HEADER_FOOTER_RE.match(line) or _DIGIT_LINE_RE.match(line):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


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

    pages = [_strip_header_footer(p) for p in raw_pages]

    return {
        "file_path": pdf_path,
        "text": "\n".join(pages),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    pdf_path = "path/to/pdf_document.pdf"
    parsed = parse_pdf(pdf_path)
    logger.info("Extracted %d characters from %s", len(parsed["text"]), parsed["file_path"])


if __name__ == "__main__":
    main()
