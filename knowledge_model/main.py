"""
FastAPI entry point for local Retrieval-Augmented Generation (TinyLlama + LoRA).

POST /ask
  Body: {"text": "...", "k": 3}
  Returns: {"answer": "...", "sources": [...]}
"""

from __future__ import annotations

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from knowledge_model.embeddings.vector_store import LocalFaiss

EMBEDDER_ID = "all-MiniLM-L6-v2"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "adapters/tinyllama-health"

embedder = SentenceTransformer(
    EMBEDDER_ID,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)
store = LocalFaiss.load()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
if torch.backends.mps.is_available():
    base_model = base_model.to("mps")

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).eval()

app = FastAPI(title="TinyLlama-RAG")


class AskRequest(BaseModel):
    text: str
    k: int = 3


def pack_context(passages: list[dict], max_tokens: int = 1800) -> list[dict]:
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
    results = store.search(np.array(q_vec), k=k)
    context = "\n\n".join(p["text"] for p in pack_context(results))

    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{query}\n\n"
        f"### Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(model.device)
    output = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)[0]
    decoded = tokenizer.decode(output, skip_special_tokens=True)

    if "### Answer" in decoded:
        decoded = decoded.split("### Answer")[-1].lstrip(": ").strip()

    paragraph = decoded.split("\n\n")[0]
    answer = " ".join(paragraph.split(". ")[:5]).strip()

    for p in results:
        if isinstance(p.get("score"), np.floating):
            p["score"] = float(p["score"])

    return {"answer": answer, "sources": results}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return rag_answer(req.text, req.k)
# testt