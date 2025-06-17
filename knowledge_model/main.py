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
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from peft import PeftModel
from inference.prompt_utils import build_prompt, SYSTEM_MSG
from inference.model_loader import load_finetuned_model

from knowledge_model.embeddings.vector_store import LocalFaiss
from knowledge_model.embeddings.re_rank import rerank


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
        if not clean:
            continue

        # Normalise for de‑duplication.
        key = re.sub(r"[^a-z]", "", clean.lower())[:60]
        if key in seen:
            continue
        seen.add(key)
        lines.append(clean)

        if len(lines) == max_items:
            break

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
    # retrieve twice as many passages, then keep the top‑k that clear the score threshold
    raw_hits = store.search(np.array(q_vec), k=k * 2)
    raw_hits = rerank(query, raw_hits, top_k=k * 2)  # re‑rank the 2k recalls
    results = [p for p in raw_hits if p.get("score", 0) >= 0.75][:k]
    if not results:
        # fall back to the highest‑score hits if nothing meets the threshold 0.75
        results = raw_hits[:k]
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

    # If post‑processing removed every bullet, drop the sources – nothing is being cited
    if answer == "Insufficient evidence.":
        results = []

    for p in results:
        if isinstance(p.get("score"), np.floating):
            p["score"] = float(p["score"])

    return {"answer": answer, "sources": results}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**rag_answer(req.text, req.k))