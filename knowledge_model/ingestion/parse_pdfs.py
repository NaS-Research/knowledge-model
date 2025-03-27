"""
parse_pdfs.py
-------------
Extract text from PDFs using PyMuPDF.
"""

import fitz
import logging

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF.
    
    :param pdf_path: Local path to the PDF file
    :return: A dict with the file path and the extracted text
    """
    doc_text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            doc_text.append(text)

    full_text = "\n".join(doc_text)
    return {
        "file_path": pdf_path,
        "text": full_text
    }

def main():
    """
    Basic entry point to parse a single PDF.
    Usage:
        python -m knowledge_model.ingestion.parse_pdfs
    """
    logging.basicConfig(level=logging.INFO)
    pdf_path = "path/to/pdf_document.pdf" # test
    parsed_data = parse_pdf(pdf_path)
    logger.info(
        "Extracted %d characters from %s",
        len(parsed_data["text"]),
        parsed_data["file_path"]
    )

if __name__ == "__main__":
    main()
