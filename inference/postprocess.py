
"""
inference.postprocess
---------------------
 
Utility functions that turn **raw model output** into a polished answer ready
for display / downstream usage.
 
Typical fixes:
    * remove special/marker tokens (e.g. <pad>, </s>, ...)
    * trim half‑finished sentences when generation stopped at the token limit
    * squash duplicates & heading artefacts that cause “rambling”
    * hard‑cap answer length to keep the UI neat
    * (optional) pull out inline citations like ‘PMID: 123456’
"""
 
from __future__ import annotations
 
import re
import textwrap
from dataclasses import dataclass
from typing import List, Set
 
# --------------------------------------------------------------------------- #
# configuration constants – tweak as needed
# --------------------------------------------------------------------------- #
 
# Tokens we never want to surface to the user
_SPECIAL_TOKENS: Set[str] = {
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<|endoftext|>",
    "<|assistant|>",
    "<|user|>",
}
 
# Regex to find simple inline citations.  Extend if you adopt a new style.
_CIT_RE = re.compile(r"(PMID|PMCID|DOI):\s*\S+", flags=re.IGNORECASE)
 
 
# --------------------------------------------------------------------------- #
# return container
# --------------------------------------------------------------------------- #
@dataclass
class PostProcessResult:
    """Clean answer plus any citations we spotted."""
    text: str
    citations: List[str]
 
 
# --------------------------------------------------------------------------- #
# helper functions – kept private
# --------------------------------------------------------------------------- #
def _strip_special_tokens(raw: str) -> str:
    """Remove reserved tokens inserted by tokenizer / prompting template."""
    for tok in _SPECIAL_TOKENS:
        raw = raw.replace(tok, "")
    return raw.strip()
 
 
def _trim_to_last_period(txt: str) -> str:
    """If the model stopped mid‑sentence, cut back to the last full stop."""
    last_full_stop = txt.rfind(".")
    if last_full_stop > 0:
        return txt[: last_full_stop + 1]
    return txt
 
 
def _dedup_sentences(txt: str) -> str:
    """
    Drop *exact* duplicate sentences that sometimes appear when the model
    loops.  Keeps original order and spacing.
    """
    seen: Set[str] = set()
    cleaned: List[str] = []
    for sent in re.split(r"(?<=\.)\s+", txt):
        if sent and sent not in seen:
            cleaned.append(sent)
            seen.add(sent)
    return " ".join(cleaned)
 
 
def _cap_length(txt: str, max_words: int | None) -> str:
    """Hard‑cap output length (words) so replies never explode."""
    if max_words is None:
        return txt
    words = txt.split()
    if len(words) <= max_words:
        return txt
    return " ".join(words[:max_words]) + "…"
 
 
def _extract_citations(txt: str) -> List[str]:
    """Return a *unique* list of inline citations we recognise."""
    return list(dict.fromkeys(match.group(0) for match in _CIT_RE.finditer(txt)))
 
 
# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def postprocess(raw_output: str, *, max_words: int | None = 200) -> PostProcessResult:
    """
    Clean `raw_output` from the model and return a structured result.
 
    Parameters
    ----------
    raw_output : str
        Text string exactly as produced by the model.
    max_words : int | None
        If provided, answer will be truncated to this many words (soft cap).
 
    Returns
    -------
    PostProcessResult
        text       – polished answer
        citations  – list like ['PMID: 12345', 'DOI:10.1000/xyz']
    """
    txt = _strip_special_tokens(raw_output)
    txt = _trim_to_last_period(txt)
    txt = _dedup_sentences(txt)
    txt = _cap_length(txt, max_words=max_words)
 
    return PostProcessResult(text=txt, citations=_extract_citations(txt))