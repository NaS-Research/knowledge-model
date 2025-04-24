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
 

 
_SPECIAL_TOKENS: Set[str] = {
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<|endoftext|>",
    "<|assistant|>",
    "<|user|>",
    "###",
}
 
_CIT_RE = re.compile(r"(PMID|PMCID|DOI):\s*\S+", flags=re.IGNORECASE)

_SECTION_HEAD_RE = re.compile(
    r"\b("
    r"Acknowledg(e)?ments?|Funding|Disclosure|Conflict(s)? of Interest|"
    r"Author(ship)? Statement|Disclaimer|References|Source"
    r")\s*:.*",
    flags=re.IGNORECASE | re.DOTALL,
)
 

@dataclass
class PostProcessResult:
    """Clean answer plus any citations we spotted."""
    text: str
    citations: List[str]
 
 
def _strip_special_tokens(raw: str) -> str:
    """Remove reserved tokens inserted by tokenizer / prompting template."""
    for tok in _SPECIAL_TOKENS:
        raw = raw.replace(tok, "")
    return raw.strip()
 
 
def _trim_to_last_period(txt: str) -> str:
    """If generation stopped mid‑sentence, cut back to the last sentence end."""
    idx = max(txt.rfind("."), txt.rfind("?"), txt.rfind("!"))
    return txt[: idx + 1] if idx > 0 else txt
 
 
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

def _remove_boilerplate(txt: str) -> str:
    """
    Cut any trailing boiler‑plate sections (Acknowledgments, Funding, etc.)
    that sometimes leak from PubMed‑style training docs.
    """
    m = _SECTION_HEAD_RE.search(txt)
    return txt[: m.start()] if m else txt
 
 
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
    txt = _remove_boilerplate(txt)
    txt = _cap_length(txt, max_words=max_words)
 
    return PostProcessResult(text=txt, citations=_extract_citations(txt))

def clean(raw_output: str) -> str:
    """
    Lightweight wrapper kept for older callers (e.g. cli_chat) that only
    need the polished text string.

        >>> from inference.postprocess import clean
        >>> print(clean("raw <pad> output..."))
    """
    return postprocess(raw_output, max_words=None).text