"""
text_cleaner.py
Removes basic clutter like bracketed references and excessive whitespace.
"""

import re

def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)  # remove bracketed references like [1], [2,3]
    text = re.sub(r"\(Fig.*?\)", "", text)  # remove figure references like (Fig. 2)
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra whitespace
    return text
