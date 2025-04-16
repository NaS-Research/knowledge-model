"""
text_cleaner.py
Removes bracketed references, figure references, collapses whitespace,
optionally removes a trailing 'References' section, and splits text into chunks.
"""

import re
# Pre‑compiled patterns for speed and to avoid over‑matching
REF_TAG      = re.compile(r"\[(?:[\w\s,;-]{1,20})\]")               # e.g. [12] or [Smith 2020]
FIG_TAG      = re.compile(r"\((?:fig(?:ure)?\s?[A-Za-z0-9]+)\)", re.I)  # e.g. (Fig 2A) or (Figure S1)
REF_SECTION  = re.compile(r"\n(?:references|bibliography)\b", re.I)    # section header

def clean_text(text: str) -> str:
    """
    Removes bracketed references, figure references, a trailing 'References' section,
    and collapses extra whitespace.
    """
    text = REF_TAG.sub("", text)
    text = FIG_TAG.sub("", text)
    text = remove_references_section(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_references_section(text: str) -> str:
    """
    Removes everything from 'References' or 'Bibliography' to the end (case‑insensitive).
    """
    m = REF_SECTION.search(text)
    return text[: m.start()] if m else text

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """
    Splits text into chunks of roughly 'chunk_size' words each.
    """
    words = text.split()
    chunks, current, count = [], [], 0

    for w in words:
        current.append(w)
        count += 1
        if count >= chunk_size:
            chunks.append(" ".join(current))
            current.clear()
            count = 0

    if current:
        chunks.append(" ".join(current))

    return chunks

def process_jsonl(src_file="data/science_articles/train.jsonl", out_dir="data/clean"):
    """
    Reads a line‑delimited JSONL file produced by the ingestion pipeline,
    cleans & chunks each record, and writes one JSON file per chunk into `out_dir`.
    Output filename pattern: {pmid}_{chunk_index}.json
    """
    import json, pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pmid  = rec.get("pmid", "NA")
            title = rec.get("title", "")
            cleaned = clean_text(rec.get("text", ""))
            chunks  = chunk_text(cleaned, 512)     # 512‑word chunks

            for i, ch in enumerate(chunks):
                out_obj = {
                    "id": f"{pmid}_{i}",
                    "title": title,
                    "chunks": [ch]
                }
                out_path = pathlib.Path(out_dir) / f"{pmid}_{i}.json"
                out_path.write_text(json.dumps(out_obj))

if __name__ == "__main__":
    # default paths match ingestion pipeline
    process_jsonl()
    print("Cleaned chunks written to data/clean/")