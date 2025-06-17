"""
FastAPI entry point for local Retrieval-Augmented Generation (TxGemma + LoRA).

POST /ask
  Body: {"text": "...", "k": 3}
  Returns: {"answer": "...", "sources": [...]}
"""

from __future__ import annotations

import logging
import re
import numpy as np
import torch
import random
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from inference.prompt_utils import build_prompt, SYSTEM_MSG
from inference.model_loader import load_finetuned_model

from knowledge_model.embeddings.vector_store import LocalFaiss
from knowledge_model.embeddings.re_rank import rerank


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# --- retrieval helper -------------------------------------------------------
def _retrieve(query_text: str, query_vec: np.ndarray, k: int, mult: int) -> list[dict]:
    """
    Retrieve k × mult passages, then re‑rank with the cross‑encoder.

    Parameters
    ----------
    query_vec : np.ndarray
        Normalised embedding of the user query (shape =(1, dim)).
    k : int
        Desired final top‑k to return downstream.
    mult : int
        Multiplier for an initial wider recall before re‑ranking.

    Returns
    -------
    list[dict]
        Re‑ranked passage dictionaries, sorted by relevance score (highest→lowest).
    """
    raw = store.search(query_vec, k=k * mult)
    return rerank(query_text, raw, top_k=k * mult)

# ------------------------------------------------------------------
# Human‑sounding fallback disclaimers (italic, single sentence)
_DISCLAIMERS = [
    "_I don't currently have information available on this specific topic._",
    "_I'm unable to answer your exact question at this time._",
    "_My current resources don't include details on this topic._",
    "_I haven't been trained to answer this particular question yet._",
    "_Information on this specific issue isn't available to me right now._",
    "_I don't yet have sufficient data to accurately address this question._",
    "_Right now, I lack enough information to respond to this query._",
]
# ------------------------------------------------------------------

def _generate_fallback(question: str) -> str:
    """
    Generate a short, uncited answer when retrieval yields no high‑score passages.

    • Picks a random, human‑friendly disclaimer to avoid repetitive wording.
    • Enforces “bullets only” formatting (no section headers, no citations).
    """
    disclaimer = random.choice(_DISCLAIMERS) + "\n"

    prompt = (
        "### System:\n"
        "You are Nicole, a concise biomedical assistant. "
        "Return ONLY bullet lines — no headers, no citations. "
        "No dialogue tags such as 'You:' or 'Assistant:'.\n"
        "Each bullet must end with a period.\n"
        "### User:\n"
        f"{question}\n"
        "### Assistant:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stops  = StoppingCriteriaList([])  # allow full completion
    out    = model.generate(
        **inputs,
        max_new_tokens=320,
        stopping_criteria=stops,
        eos_token_id=tokenizer.eos_token_id,
    )[0]

    answer = tokenizer.decode(out, skip_special_tokens=True).split("### Assistant:")[-1].strip()

    return disclaimer + _postprocess_bullets(answer)

# Collapse model output into a de‑duplicated bullet list.
def _postprocess_bullets(text: str, max_items: int = 10) -> str:
    """
    Convert raw model output into a clean, de‑duplicated bullet list.

    * Strips leading bullet/number markers and surrounding whitespace.
    * Normalises each candidate line to a lowercase, alpha‑only key so that
      near‑duplicates are removed (e.g. differing only by punctuation).
    * Returns at most `max_items` bullets in original order.
    * If no usable content remains, returns the sentinel “Insufficient evidence.”.
    """
    lines: list[str] = []
    seen: set[str] = set()

    for raw in text.splitlines():
        # Remove common bullet markers (•, -, *, digits, whitespace).
        clean = re.sub(r"^[\s•*\-0-9.]+", "", raw).strip()
        # Drop disclaimer / header lines
        if clean.lower().startswith(("**no matching evidence", "###")):
            continue
        # Skip prompt‑leak lines like “### Response”
        if clean.lower().startswith("### response"):
            continue

        # Skip stray dialogue artefacts
        if re.match(r"(?i)^(you|assistant):", clean):
            continue

        if not clean:
            continue

        norm = re.sub(r"\([^)]*\)", "", clean)  # drop examples like (budesonide)
        key  = re.sub(r"[^a-z]", "", norm.lower())[:60]
        if key in seen:
            continue
        seen.add(key)
        lines.append(clean)

        if len(lines) == max_items:
            break

    if not lines:
        return "Insufficient evidence."

    return "\n".join(f"• {l}" for l in lines) if lines else "Insufficient evidence."

EMBEDDER_ID = "BAAI/bge-large-en-v1.5"
BASE_MODEL = "google/txgemma-2b-predict"
ADAPTER_PATH = "adapters/txgemma_lora_instr_v1"
FAISS_PATH = "data/faiss"

embedder = SentenceTransformer(
    EMBEDDER_ID,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

try:
    # Use the classmethod loader to properly initialise from disk.
    store = LocalFaiss.load(FAISS_PATH)
except FileNotFoundError as exc:
    logger.error("FAISS index not found at %s", FAISS_PATH)
    raise

dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load fine‑tuned TxGemma‑LoRA adapter via central loader
tokenizer, model = load_finetuned_model(BASE_MODEL, ADAPTER_PATH)

logger.info("Embedder %s | top‑k=%d | score≥0.80 | model=txgemma‑LoRA", EMBEDDER_ID, 12)
logger.info("Model and FAISS store loaded; ready to serve.")

app = FastAPI(title="TxGemma-RAG")


class AskRequest(BaseModel):
    text: str
    k: int = 12


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


def pack_context(passages: list[dict], max_tokens: int = 800) -> list[dict]:
    selected, used = [], 0
    for p in passages:
        n_tok = len(tokenizer.tokenize(p["text"]))
        if used + n_tok > max_tokens:
            break
        selected.append(p)
        used += n_tok
    return selected


def rag_answer(query: str, k: int = 3) -> dict:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    # 1️⃣ Normal recall – retrieve 2 × k then keep hits ≥ 0.80
    raw_hits = _retrieve(query, q_vec, k, mult=2)
    results  = [p for p in raw_hits if p.get("score", 0) >= 0.80][:k]

    # 2️⃣ Wider‑net recall – retrieve 4 × k then keep hits ≥ 0.65
    if not results:
        raw_hits = _retrieve(query, q_vec, k, mult=4)
        results  = [p for p in raw_hits if p.get("score", 0) >= 0.65][:k]

    # 3️⃣ If still empty ➜ generative fallback
    if not results:
        return {"answer": _generate_fallback(query), "sources": []}

    logger.debug(
        "hits≥0.80: %d | hits≥0.65: %d | returned: %d",
        sum(p.get("score", 0) >= 0.80 for p in raw_hits),
        sum(p.get("score", 0) >= 0.65 for p in raw_hits),
        len(results),
    )
    context = "\n\n".join(p["text"] for p in pack_context(results))

    # If retrieval returns no usable context, short‑circuit with a safe response
    if not context.strip():
        return {"answer": "Insufficient evidence.", "sources": []}
    user_block = f"Context:\n{context}\n\nQuestion:\n{query}"
    prompt = build_prompt(user_block, system_msg=SYSTEM_MSG)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(model.device)
    output = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)[0]
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    if "### Assistant:" in decoded:
        decoded = decoded.split("### Assistant:")[-1].strip()

    answer = _postprocess_bullets(decoded)

    if answer == "Insufficient evidence.":
        # fallback small‑context generation
        answer = _generate_fallback(query)
        results = []

    for p in results:
        if isinstance(p.get("score"), np.floating):
            p["score"] = float(p["score"])

    return {"answer": answer, "sources": results}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**rag_answer(req.text, req.k))