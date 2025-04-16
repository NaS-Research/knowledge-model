"""
Parses text from a PDF using PyMuPDF.
"""

import logging
from typing import Any

import fitz

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str) -> dict[str, Any]:
    """
    Extract plain text from a PDF file.

    Args:
        pdf_path: Path to the local PDF file.

    Returns:
        A dictionary with the file path and extracted text content.
    """
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]

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
