"""
text_cleaner.py
Removes basic clutter and splits text into chunks.
"""

import re

def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(Fig.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    words = text.split()
    chunks = []
    current = []
    count = 0

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
