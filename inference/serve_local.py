
"""
FastAPI micro‑service that exposes the fine‑tuned TxGemma adapter locally.

Run:
    uvicorn inference.serve_local:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
POST /chat
    JSON body: {"prompt": "...", "max_tokens": 512}
    Streams the generated answer token‑by‑token using Server‑Sent Events (SSE).

GET /health
    Simple liveness probe that returns {"status": "ok"}
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator, Dict

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from peft import PeftModel, PeftConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from inference.prompt_utils import build_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_ID: str = "google/txgemma-2b-chat"
ADAPTER_DIR: Path = Path("adapters/txgemma-health")
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="TxGemma‑Health – local inference")


@lru_cache(maxsize=1)
def _load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token
    return tok


@lru_cache(maxsize=1)
def _load_model():
    """
    Returns a PeftModel ready for generation on the selected device.
    Uses LRU cache so we pay start‑up cost only once.
    """
    logger.info("Loading base model %s …", MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {ADAPTER_DIR.resolve()}")

    logger.info("Merging LoRA adapter from %s", ADAPTER_DIR)
    peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, config=peft_cfg, device_map="cpu")
    model = model.to(DEVICE)
    return model


def _stream_generation(prompt: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
    """
    Yields generated tokens one‑by‑one so the client can stream them.
    Implements a simple SSE format: "data: <token>\n\n"
    """
    tokenizer = _load_tokenizer()
    model = _load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        top_p=0.9,
        temperature=0.7,
    )

    from threading import Thread

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield f"data: {token}\n\n"

    thread.join()
    yield "data: [DONE]\n\n"


@app.get("/health", response_class=JSONResponse)
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: Request):
    payload = await req.json()
    prompt_raw: str = payload.get("prompt")
    max_tokens: int = int(payload.get("max_tokens", 512))

    if not prompt_raw:
        raise HTTPException(status_code=400, detail="prompt field is required")

    prompt = build_prompt(prompt_raw)

    logger.info("Received prompt of %d chars, max_tokens=%d", len(prompt_raw), max_tokens)

    return StreamingResponse(
        _stream_generation(prompt, max_tokens=max_tokens),
        media_type="text/event-stream",
    )