"""
Text cleaning utilities for biomedical corpora.

This module strips citations, captions, and boiler‑plate sections,
normalises Unicode, and produces sentence‑aware chunks ready for
downstream embedding or language‑model fine‑tuning.

Example:
    >>> from knowledge_model.processing.text_cleaner import clean_text
    >>> clean_text("Hypertension [1] is high‑blood pressure.")
    'Hypertension is high‑blood pressure.'
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import cache
from pathlib import Path
from typing import Any, List

try:
    from unidecode import unidecode 
except ModuleNotFoundError:
    import unicodedata

    def unidecode(text: str) -> str:
        """Best‑effort ASCII‑fold for environments without `unidecode`."""
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

try:
    import nltk

    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    nltk = None

logger = logging.getLogger(__name__)

REF_TAG_RE = re.compile(r"\[(?:[\w\s,;-]{1,20})\]")
FIG_TAG_RE = re.compile(r"\((?:fig(?:ure)?\s?[A-Za-z0-9]+)\)", re.I)
REF_SECTION_RE = re.compile(r"\n(?:references|bibliography)\b", re.I)
AUTHOR_YEAR_RE = re.compile(r"\([A-Z][A-Za-z]*(?: et al\.)?,?\s?\d{4}[a-z]?\)", re.I)
FIG_CAPTION_RE = re.compile(r"^\s*(Figure|Table)\s+\d+[^.\n]*\n?", re.I | re.M)
UNWANTED_SECTIONS_RE = re.compile(
    r"\n(?:methods?|acknowledg(?:e)?ments?|funding|conflicts? of interest)\b", re.I
)
HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s+(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_CHUNK_SIZE = 1_000


@cache
def remove_references_section(text: str) -> str:
    """Truncate text at the start of 'References' or 'Bibliography' section.

    Args:
        text (str): The input text to process.

    Returns:
        str: The truncated text.
    """
    match = REF_SECTION_RE.search(text)
    return text[: match.start()] if match else text


@cache
def truncate_unwanted_sections(text: str) -> str:
    """Remove content starting from Methods, Acknowledgments, Funding, etc.

    Args:
        text (str): The input text to process.

    Returns:
        str: The truncated text.
    """
    match = UNWANTED_SECTIONS_RE.search(text)
    return text[: match.start()] if match else text


def clean_text(text: str) -> str:
    """Clean scientific text by removing citations, captions, boiler‑plate headings,
    normalising unicode, and collapsing whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = unidecode(unicodedata.normalize("NFKC", text))

    text = REF_TAG_RE.sub(" ", text)
    text = AUTHOR_YEAR_RE.sub(" ", text)

    text = FIG_TAG_RE.sub(" ", text)
    text = FIG_CAPTION_RE.sub(" ", text)

    text = remove_references_section(text)
    text = truncate_unwanted_sections(text)

    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)

    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into sentence‑aware chunks of roughly `chunk_size` words.
    Falls back to word-based if nltk punkt is unavailable.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The desired chunk size.

    Returns:
        List[str]: A list of text chunks.
    """
    if nltk:
        sentences = nltk.sent_tokenize(text)
        chunks, current = [], []
        count = 0
        for sent in sentences:
            words = sent.split()
            if count + len(words) > chunk_size and current:
                chunks.append(" ".join(current))
                current, count = [], 0
            current.extend(words)
            count += len(words)
        if current:
            chunks.append(" ".join(current))
        return chunks
    else:
        # fallback: simple word window
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_jsonl(src_file: str = "data/science_articles/train.jsonl", out_dir: str = "data/clean") -> None:
    """Process a JSONL dataset, clean and chunk text, and save each chunk as JSON.

    Args:
        src_file (str): The source JSONL file.
        out_dir (str): The output directory for cleaned chunks.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Path(src_file).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                continue
            
            pmid = rec.get("pmid", "NA")
            title = rec.get("title", "")
            text = clean_text(rec.get("text", ""))
            chunks = chunk_text(text, 512)

            for i, chunk in enumerate(chunks):
                out_file = out_path / f"{pmid}_{i}.json"
                try:
                    out_data = {
                        "id": f"{pmid}_{i}",
                        "title": title,
                        "chunks": [chunk],
                    }
                    out_file.write_text(json.dumps(out_data, ensure_ascii=False))
                except OSError as e:
                    logger.error(f"Failed to write file {out_file}: {e}")
                    continue


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    process_jsonl()
    logger.info("Cleaned chunks written to data/clean/")
