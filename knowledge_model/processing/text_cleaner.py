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
import string
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

# --- Regexes for text cleaning
REF_TAG_RE = re.compile(r"\[(?:[\w\s,;-]{1,20})\]")
FIG_TAG_RE = re.compile(r"\((?:fig(?:ure)?\s?[A-Za-z0-9]+)\)", re.I)
REF_SECTION_RE = re.compile(r"\n(?:references|bibliography)\b", re.I)
AUTHOR_YEAR_RE = re.compile(r"\([A-Z][A-Za-z]*(?: et al\.)?,?\s?\d{4}[a-z]?\)", re.I)
FIG_CAPTION_RE = re.compile(r"^\s*(Figure|Table)\s+\d+[^.\n]*\n?", re.I | re.M)
UNWANTED_SECTIONS_RE = re.compile(
    r"\n\s*(?:methods?|acknowledg(?:e)?ments?|funding|conflicts? of interest)\b",
    re.I,
)
# Additional regexes/constants
HTML_TAG_RE          = re.compile(r"<[^>]+>")
CONTROL_CHARS_RE     = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
DUP_PUNCT_RE         = re.compile(r"([!?.,;:]){2,}")
SMART_QUOTES = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'
})
# Remove hyphen + *line break* that was inserted by PDF wrapping, but keep real hyphenated terms.
HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")
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



def _standardise_unicode(text: str) -> str:
    """Normalise Unicode, fold to ASCII, and replace smart quotes."""
    text = unicodedata.normalize("NFKC", text).translate(SMART_QUOTES)
    return unidecode(text)


def clean_text(text: str) -> str:
    """Clean scientific text by removing citations, captions, boiler‑plate headings,
    normalising unicode, and collapsing whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = _standardise_unicode(text)
    # strip HTML/XML artifacts and control characters
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    # collapse duplicate punctuation marks (e.g., “!!!” -> “!”)
    text = DUP_PUNCT_RE.sub(r"\1", text)

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
        logger.warning(
            "NLTK Punkt sentence tokenizer not available; "
            "falling back to fixed‑word window chunking."
        )
        # fallback: simple word window
        words = text.split()
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_jsonl(src_file: str = "data/science_articles/train.jsonl", out_dir: str = "data/clean") -> None:
    """Process a JSONL dataset, clean and chunk text, and save each chunk as JSON.

    Args:
        src_file (str): The source JSONL file.
        out_dir (str): The output directory for cleaned chunks.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Path(src_file).open("r", encoding="utf-8") as f:
        logger.info("Parsing %s → %s", src_file, out_dir)
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


# ------------------------------------------------------------------
# Back‑compat alias for legacy code
# ------------------------------------------------------------------
def strip_boiler(text: str) -> str:  # pragma: no cover
    """Deprecated alias kept for older ingestion scripts."""
    return clean_text(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean and chunk a JSONL corpus for downstream RAG."
    )
    parser.add_argument(
        "--src_file",
        "-i",
        default="data/science_articles/train.jsonl",
        help="Source JSONL file to clean (default: %(default)s)",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        default="data/clean",
        help="Output directory for cleaned chunks (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_jsonl(src_file=args.src_file, out_dir=args.out_dir)
    logger.info("Cleaned chunks written to %s", args.out_dir)


__all__ = ["clean_text", "chunk_text", "process_jsonl", "strip_boiler"]