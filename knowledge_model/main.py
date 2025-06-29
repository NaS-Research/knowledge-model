"""
FastAPI entry point for local Retrieval-Augmented Generation (TxGemma + LoRA).

POST /ask
  Body: {"text": "...", "k": 3}
  Returns: {"answer": "...", "sources": [...]}
"""

from __future__ import annotations

import logging
import re
# ------------------------------------------------------------------
#  Simple intent detector for greetings / thanks / farewells
# ------------------------------------------------------------------
_GREET_RE = re.compile(
    r"\b(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))\b",
    re.I,
)
_THANK_RE = re.compile(r"\b(thanks?|thank you|appreciate(?:\s+it)?)\b", re.I)
_BYE_RE   = re.compile(r"\b(bye|goodbye|see you|later|take care)\b", re.I)

def _simple_intent_reply(text: str) -> str | None:
    """Return a canned reply for trivial small‑talk; else None."""
    if _GREET_RE.search(text):
        return "Hello — I’m Nicole. How can I assist you today?"
    if _THANK_RE.search(text):
        return "You’re very welcome!"
    if _BYE_RE.search(text):
        return "Good‑bye for now; come back any time."
    return None

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

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)

embedder: SentenceTransformer | None = None
store: "LocalFaiss" | None = None
USE_RERANK = True
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

    EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"
    BASE_MODEL   = "google/txgemma-2b-predict"
    ADAPTER_PATH = "adapters/txgemma_lora_instr_v1"
    FAISS_PATH   = "data/faiss"
    S3_PREFIX    = os.getenv("FAISS_S3_PREFIX")
    ADAPTER_S3   = os.getenv("ADAPTER_S3_PREFIX")

    # Ensure index + metadata exist (download once per container)
    if S3_PREFIX:
        _download_if_missing(f"{S3_PREFIX}/faiss.idx",       f"{FAISS_PATH}.idx")
        _download_if_missing(f"{S3_PREFIX}/faiss.idx.meta", f"{FAISS_PATH}.idx.meta")

    # Ensure LoRA adapter files exist
    if ADAPTER_S3:
        _download_if_missing(f"{ADAPTER_S3}/adapter_config.json",
                             f"{ADAPTER_PATH}/adapter_config.json")
        _download_if_missing(f"{ADAPTER_S3}/adapter_model.safetensors",
                             f"{ADAPTER_PATH}/adapter_model.safetensors")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    global USE_RERANK
    USE_RERANK = device != "cpu"
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
        return []
    if not USE_RERANK:
        return store.search(query_vec, k=k)
    raw = store.search(query_vec, k=k * mult)
    return rerank(query_text, raw, top_k=k * mult)

# Human‑sounding fallback disclaimers (italic, single sentence)
# These are followed by a natural transition into fallback bullet points.
_DISCLAIMERS = [
    "_I don't currently have information available on this specific topic, but here are some general points that may help:_",
    "_I'm unable to answer your exact question at this time; however, here are a few related considerations:_",
    "_My current resources don't include details on this topic. Here are some general insights instead:_",
    "_I haven't been trained to answer this particular question yet. You may find these general points useful:_",
    "_Information on this specific issue isn't available to me right now. Here are some related notes:_",
    "_I don't yet have sufficient data to accurately address this question. In the meantime, consider the following:_",
    "_Right now, I lack enough information to respond to this query. Here are a few general points to consider:_",
]
_LAST_DISCLAIMER: str | None = None

def _generate_fallback(question: str) -> str:
    """
    Generate a short, uncited answer when retrieval yields no high‑score passages.

    • Picks a random, human‑friendly disclaimer to avoid repetitive wording.
    • Enforces “bullets only” formatting (no section headers, no citations).
    """
    global _LAST_DISCLAIMER
    # pick a disclaimer different from the one used last time
    choice = random.choice(_DISCLAIMERS)
    if _LAST_DISCLAIMER and len(_DISCLAIMERS) > 1:
        while choice == _LAST_DISCLAIMER:
            choice = random.choice(_DISCLAIMERS)
    _LAST_DISCLAIMER = choice
    disclaimer = choice + "\n\n"

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

def _zero_context_answer(question: str) -> str:
    """
    Generate an answer from model parameters with *no* retrieval context.
    Used when FAISS returns nothing.
    """
    prompt = (
        "### System:\n"
        "You are Nicole, a friendly biomedical assistant. "
        "Answer clearly and concisely. If you truly do not know, say "
        "'I’m not sure about that.'\n"
        "### User:\n"
        f"{question}\n"
        "### Assistant:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=320,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    return tokenizer.decode(out, skip_special_tokens=True).split("### Assistant:")[-1].strip()

def _looks_unhelpful(text: str) -> bool:
    """Heuristic to detect empty or evasive answers."""
    lo = text.lower()
    return (
        len(text.split()) < 3
        or "i’m not sure" in lo
        or "i am not sure" in lo
        or "insufficient evidence" in lo
    )

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


app = FastAPI(title="TxGemma-RAG")

def _preload() -> None:
    """Warm‑start embedder, FAISS and the LM in a background thread."""
    import threading
    threading.Thread(target=_lazy_init, name="prewarm", daemon=True).start()

app.add_event_handler("startup", _preload)


from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://knowledge-model.onrender.com",
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
    # Quick small‑talk shortcut
    quick = _simple_intent_reply(query)
    if quick:
        return {"answer": quick, "sources": []}
    if store is None:
        return {"answer": _generate_fallback(query), "sources": []}
    q_vec = embedder.encode([query], normalize_embeddings=True)
    # 1️⃣ Normal recall – retrieve 3 × k then keep hits ≥ 0.75
    raw_hits = _retrieve(query, q_vec, k, mult=3)
    results  = [p for p in raw_hits if p.get("score", 0) >= 0.75][:k]

    # 2️⃣ Wider‑net recall – retrieve 5 × k then keep hits ≥ 0.55
    if not results:
        raw_hits = _retrieve(query, q_vec, k, mult=5)
        results  = [p for p in raw_hits if p.get("score", 0) >= 0.55][:k]

    # 3️⃣ If still empty ➜ try parameter‑only generation, then fallback
    if not results:
        answer = _zero_context_answer(query)
        if _looks_unhelpful(answer):
            answer = _generate_fallback(query)
        return {"answer": answer, "sources": []}

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