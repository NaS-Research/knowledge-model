"""
Utility for turning long cleaned text into overlapping retrieval-friendly passages.

Example
-------
>>> from knowledge_model.processing.passage_splitter import split_passages
>>> split_passages("abc…", size=300, overlap=50)   # returns list[str]
"""

from __future__ import annotations
from typing import List

DEFAULT_SIZE     = 300     # visible characters (≈ tokens /1.2 for en-US)
DEFAULT_OVERLAP  = 50

def split_passages(text: str,
                   size: int = DEFAULT_SIZE,
                   overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Return a list of character-based passages with fixed overlap."""
    if size <= overlap:
        raise ValueError("`size` must be larger than `overlap`")
    passages: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        passages.append(text[start:end].strip())
        start += size - overlap
    return passages