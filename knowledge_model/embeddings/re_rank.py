from typing import List, Dict, Any
import logging
from functools import lru_cache
import torch
import numpy as np
from sentence_transformers import CrossEncoder


_CE_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple silicon / Metal backend
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _get_ce() -> CrossEncoder:
    """Lazily load the cross‑encoder the first time we need it."""
    logging.info("Loading cross‑encoder %s on %s …", _CE_ID, _pick_device())
    return CrossEncoder(_CE_ID, device=_pick_device())


def rerank(
    query: str,
    hits: List[Dict[str, Any]],
    *, top_k: int
) -> List[Dict[str, Any]]:
    """
    Re-rank FAISS hits with a cross-encoder and return the *top_k* passages.

    Args
    ----
    query : The original user query (plain text).
    hits  : FAISS-filtered passages (each must have a ``"text"`` key).
    top_k : How many re-ranked results to keep.
    """
    if len(hits) <= top_k:
        return hits

    pairs = [(query, h["text"]) for h in hits]
    ce = _get_ce()
    scores = ce.predict(pairs, convert_to_numpy=True)
    order  = np.argsort(-scores)[:top_k]            # highest → lowest
    reranked = [hits[i] | {"re_score": float(scores[i])} for i in order]
    logging.debug("Re-ranked %d → %d passages", len(hits), len(reranked))
    return reranked