"""
inference.server
================

A **tiny FastAPI service** that exposes the fine‑tuned TinyLlama as a REST API.

Endpoints
---------
GET  /health          – simple readiness probe
POST /generate        – run generation and return a cleaned answer

Run locally
-----------
$ uvicorn inference.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.config import Settings
from inference.model import load_model
from inference.postprocess import clean_output 

logger = logging.getLogger(__name__)
settings = Settings()

logger.info("Loading model + adapter …")
tokenizer, lm = load_model(
    adapter_path=settings.adapter_path,
    device=settings.device,
)
logger.info("Model ready ✓")

app = FastAPI(
    title="NaS TinyLlama Inference",
    version="0.1.0",
    docs_url="/",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User question / instruction")
    max_tokens: int = Field(
        512, ge=1, le=2048, description="Max new tokens to generate"
    )
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        0.95, ge=0.0, le=1.0, description=" nucleus sampling p"
    )


class GenerateResponse(BaseModel):
    answer: str
    raw: str
    usage: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    """Kubernetes‑friendly readiness probe."""
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate an answer and return a cleaned version + metadata."""
    logger.info("Generating for prompt (len=%d)…", len(req.prompt))
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt").to(settings.device)
        with torch.no_grad():
            output_ids = lm.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=True,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = clean_output(full_text, req.prompt)

        usage = {
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": len(output_ids[0]) - len(inputs["input_ids"][0]),
        }
        return GenerateResponse(answer=answer, raw=full_text, usage=usage)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference.server:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )