"""
text_cleaner.py
Removes bracketed references, figure references, collapses whitespace,
optionally removes a trailing 'References' section, and splits text into chunks.
"""

import re

def clean_text(text: str) -> str:
    """
    Removes bracketed references, figure references, a trailing 'References' section,
    and collapses extra whitespace.
    """
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(Fig.*?\)", "", text)
    text = remove_references_section(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_references_section(text: str) -> str:
    """
    Removes everything from 'References' to the end of the text (case-insensitive).
    """
    match = re.search(r"\nreferences\b", text, flags=re.IGNORECASE)
    if match:
        text = text[: match.start()]
    return text

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
