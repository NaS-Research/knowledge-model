"""
Text cleaner: removes references, normalizes spacing, and splits into word chunks.
"""

import json
import re
from pathlib import Path
from typing import Any

REF_TAG = re.compile(r"\[(?:[\w\s,;-]{1,20})\]")
FIG_TAG = re.compile(r"\((?:fig(?:ure)?\s?[A-Za-z0-9]+)\)", re.I)
REF_SECTION = re.compile(r"\n(?:references|bibliography)\b", re.I)


def clean_text(text: str) -> str:
    """
    Clean scientific text by removing references and excess whitespace.
    """
    text = REF_TAG.sub("", text)
    text = FIG_TAG.sub("", text)
    text = remove_references_section(text)
    return re.sub(r"\s+", " ", text).strip()


def remove_references_section(text: str) -> str:
    """
    Truncate text at the start of 'References' or 'Bibliography' section.
    """
    match = REF_SECTION.search(text)
    return text[: match.start()] if match else text


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """
    Split text into word-based chunks of specified size.
    """
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


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
