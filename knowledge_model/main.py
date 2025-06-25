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
import os
import pathlib
import boto3
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

def _download_if_missing(s3_uri: str, local_path: str) -> None:
    """
    Copy s3://bucket/key to *local_path* unless the file already exists.

    Render containers have ephemeral disk, so the file is cached for the
    lifetime of the instance (avoids repeat downloads on every request).
    """
    if os.path.exists(local_path):
        return

    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    logger.info("Downloading %s → %s …", s3_uri, local_path)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Pull using boto3 (credentials supplied via env vars)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)

embedder: SentenceTransformer | None = None
store: "LocalFaiss" | None = None
tokenizer = None
model = None

def _lazy_init() -> None:
    """
    Load the embedder, FAISS store, and fine‑tuned language model the first time
    we receive a user request.  This ensures Uvicorn binds to the port within a
    second, so Render’s health‑check succeeds.  Subsequent calls are no‑ops.
    """
    global embedder, store, tokenizer, model

    if embedder is not None:
        return 

    EMBEDDER_ID = "BAAI/bge-large-en-v1.5"
    BASE_MODEL   = "google/txgemma-2b-predict"
    ADAPTER_PATH = "adapters/txgemma_lora_instr_v1"
    FAISS_PATH   = "data/faiss"
    S3_PREFIX    = os.getenv("FAISS_S3_PREFIX")

    # Ensure index + metadata exist (download once per container)
    if S3_PREFIX:
        _download_if_missing(f"{S3_PREFIX}/faiss.idx",       f"{FAISS_PATH}.idx")
        _download_if_missing(f"{S3_PREFIX}/faiss.idx.meta", f"{FAISS_PATH}.idx.meta")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    dtype  = torch.float16 if device != "cpu" else torch.float32

    logger.info("Loading embedder %s on %s …", EMBEDDER_ID, device)
    embedder = SentenceTransformer(EMBEDDER_ID, device=device)

    try:
        store = LocalFaiss.load(FAISS_PATH)
    except FileNotFoundError as e:
        logger.warning("%s; retrieval disabled, generation‑only mode", e)
        store = None

    logger.info("Loading base model %s with LoRA from %s …", BASE_MODEL, ADAPTER_PATH)
    tokenizer, model_ = load_finetuned_model(BASE_MODEL, ADAPTER_PATH, torch_dtype=dtype)
    model_.to(device)
    model = model_

    logger.info("Model, embedder, and FAISS store ready.")

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
    if store is None:
        return []  # retrieval disabled

    raw = store.search(query_vec, k=k * mult)
    return rerank(query_text, raw, top_k=k * mult)

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


logger.info("Embedder %s | top‑k=%d | score≥0.80 | model=txgemma‑LoRA", "BAAI/bge-large-en-v1.5", 12)
logger.info("Model and FAISS store loaded; ready to serve.")


app = FastAPI(title="TxGemma-RAG")

from fastapi.middleware.cors import CORSMiddleware

# CORS settings: allow local dev and your production front‑end.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nas-frontend.vercel.app",  # TODO: replace with actual prod URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    """
    Lightweight liveness probe for Render.
    """
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def root() -> dict[str, str]:
    """
    Friendly landing route so browser hits to '/' don't 404.
    """
    return {
        "message": "Nicole RAG API – send POST /ask with {'text': '...', 'k': 3}",
        "docs": "/docs",
        "health": "/health",
    }


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
    _lazy_init()
    if store is None:
        return {"answer": _generate_fallback(query), "sources": []}
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