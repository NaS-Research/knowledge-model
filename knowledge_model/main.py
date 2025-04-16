"""
FastAPI entry‑point for local Retrieval‑Augmented Generation (TinyLlama + LoRA).

Endpoints
---------
POST /ask
    JSON body: {"text": "<user question>", "k": <top‑k, default 3>}
    Returns: {"answer": "...", "sources":[{chunk, score, ...}, ...]}

Requirements:
    pip install fastapi uvicorn sentence-transformers peft
    (embeddings.vector_store must have been built beforehand)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch, os
from sentence_transformers import SentenceTransformer
from knowledge_model.embeddings.vector_store import LocalFaiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Retrieval assets ----------
EMBEDDER_ID = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(
    EMBEDDER_ID,
    device="mps" if torch.backends.mps.is_available() else "cpu"
)
store = LocalFaiss.load()  # loads data/faiss.idx + metadata

# ---------- LLM with healthcare LoRA ----------
BASE    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "adapters/tinyllama-health"

tok = AutoTokenizer.from_pretrained(BASE)
# --- load base model safely on CPU then move to MPS ---
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,        # cast bf16 → fp16
    device_map="cpu",                 # load weights on CPU
    low_cpu_mem_usage=True,
)
if torch.backends.mps.is_available():
    base_model = base_model.to("mps")
model = PeftModel.from_pretrained(base_model, ADAPTER).eval()

# ---------- FastAPI wiring ----------
app = FastAPI(title="TinyLlama‑RAG")

class AskRequest(BaseModel):
    text: str
    k: int = 3

# --- helper -----------------------------------------------------------
def pack_context(passages, max_ctx_tokens=1800):
    """
    Select as many passages as fit under `max_ctx_tokens` when tokenized.
    Returns the trimmed list.
    """
    selected, used = [], 0
    for p in passages:
        n_tok = len(tok.tokenize(p["text"]))
        if used + n_tok > max_ctx_tokens:
            break
        selected.append(p)
        used += n_tok
    return selected

def rag_answer(query: str, k: int = 3) -> dict:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    passages = store.search(np.array(q_vec), k=k)
    passages = pack_context(passages)  # trim to fit context window

    context = "\n\n".join(p["text"] for p in passages)
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n\n"
        f"### Answer:\n"
    )
    # Truncate so total prompt ≤ 2 000 tokens (TinyLlama limit is 2 048)
    ids = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2000,
    ).to(model.device)
    out = model.generate(**ids, max_new_tokens=256, eos_token_id=tok.eos_token_id)[0]

    raw = tok.decode(out, skip_special_tokens=True)
    # Keep text **after** the *last* "### Answer" tag, then the first blank line
    if "### Answer" in raw:
        raw = raw.split("### Answer")[-1]
    clean = raw.lstrip(": ").strip()
    # take everything until first double‑newline OR max 5 sentences
    paragraph = clean.split("\n\n")[0]
    answer = " ".join(paragraph.split(". ")[:5]).strip()
    # Ensure JSON‑serializable scores
    for p in passages:
        if isinstance(p.get("score"), (np.floating,)):
            p["score"] = float(p["score"])
    return {"answer": answer, "sources": passages}

@app.post("/ask")
def ask(req: AskRequest):
    return rag_answer(req.text, req.k)