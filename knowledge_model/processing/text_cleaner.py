"""
Text cleaner: strips citations/captions, normalises whitespace, merges hyphen breaks,
sentence‑aware chunking, and drops unwanted sections (Methods, Acknowledgments, etc.).
"""

import json
import re
import unicodedata
from unidecode import unidecode
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    nltk = None
from pathlib import Path
from typing import Any

REF_TAG = re.compile(r"\[(?:[\w\s,;-]{1,20})\]")
FIG_TAG = re.compile(r"\((?:fig(?:ure)?\s?[A-Za-z0-9]+)\)", re.I)
REF_SECTION = re.compile(r"\n(?:references|bibliography)\b", re.I)
AUTHOR_YEAR_TAG = re.compile(r"\([A-Z][A-Za-z]*(?: et al\.)?,?\s?\d{4}[a-z]?\)", re.I)
FIG_CAPTION = re.compile(r"^\s*(Figure|Table)\s+\d+[^.\n]*\n?", re.I | re.M)
UNWANTED_SECTIONS = re.compile(r"\n(?:methods?|acknowledg(?:e)?ments?|funding|conflicts? of interest)\b", re.I)


def clean_text(text: str) -> str:
    """
    Clean scientific text by removing citations, captions, boiler‑plate headings,
    normalising unicode, and collapsing whitespace.
    """
    # Unicode normalisation + ASCII fallback
    text = unidecode(unicodedata.normalize("NFKC", text))

    # Remove inline numeric and author‑year citations
    text = REF_TAG.sub(" ", text)
    text = AUTHOR_YEAR_TAG.sub(" ", text)

    # Remove figure / table tags & captions
    text = FIG_TAG.sub(" ", text)
    text = FIG_CAPTION.sub(" ", text)

    # Truncate at references or other unwanted sections
    text = remove_references_section(text)
    text = truncate_unwanted_sections(text)

    # Merge hyphenated line breaks: "hyper-\ntension" -> "hypertension"
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def remove_references_section(text: str) -> str:
    """
    Truncate text at the start of 'References' or 'Bibliography' section.
    """
    match = REF_SECTION.search(text)
    return text[: match.start()] if match else text


def truncate_unwanted_sections(text: str) -> str:
    """
    Remove content starting from Methods, Acknowledgments, Funding, etc.
    """
    match = UNWANTED_SECTIONS.search(text)
    return text[: match.start()] if match else text


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """
    Split text into sentence‑aware chunks of roughly `chunk_size` words.
    Falls back to word-based if nltk punkt is unavailable.
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
    """
    Process a JSONL dataset, clean and chunk text, and save each chunk as JSON.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Path(src_file).open("r", encoding="utf-8") as f:
        for line in f:
            rec: dict[str, Any] = json.loads(line)
            pmid = rec.get("pmid", "NA")
            title = rec.get("title", "")
            text = clean_text(rec.get("text", ""))
            chunks = chunk_text(text, 512)

            for i, chunk in enumerate(chunks):
                out_file = out_path / f"{pmid}_{i}.json"
                out_data = {
                    "id": f"{pmid}_{i}",
                    "title": title,
                    "chunks": [chunk],
                }
                out_file.write_text(json.dumps(out_data, ensure_ascii=False))


if __name__ == "__main__":
    process_jsonl()
    print("Cleaned chunks written to data/clean/")
