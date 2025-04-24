
"""
Utilities for building prompts and streaming generations during local
inference.  Designed to stay small‑footprint but extensible.

Public API
----------
load_model()          – load base model (+ optional LoRA) once.
build_prompt()        – assemble the final prompt string.
stream_generate()     – generator that yields response chunks.
chat_once()           – blocking helper that returns the full answer.

Run this file directly for a quick smoke‑test:

    python inference/prompt_utils.py "What is sporadic cerebral small‑vessel disease?"
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
import logging
logger = logging.getLogger(__name__)

__all__ = [
    "SYSTEM_MSG",
    "STOP_STRINGS",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "build_prompt",
    "load_model",
    "stream_generate",
    "chat_once",
]

SYSTEM_MSG: str = (
    "You are **Nicole**, a state‑of‑the‑art biomedical large‑language model—"
    "the definitive expert across genetics, molecular biology, pharmacology, "
    "clinical medicine, translational research, and life sciences.  Your audience ranges "
    "from Nobel‑ and Lasker‑prize–winning investigators and attending "
    "physicians to PharmD/PhD trainees and undergraduates.  Respond in a "
    "professional, precise, yet approachable tone:\n"
    "• Provide clear, well‑structured explanations that reflect deep domain "
    "  knowledge.\n"
    "• Paraphrase in your own words—**do not** copy text verbatim or include "
    "  reference lists or citations.\n"
    "• Use concise paragraphs, and bullet or numbered lists when they improve "
    "  readability.\n"
    "• There is no rigid length limit; focus on clarity and completeness.\n"
    "• Begin directly with the answer—no salutations, no headers such as "
    "  \"Answer:\".\n"
    "• Maintain a respectful, collaborative tone suitable for peer "
    "  discussion among scientists and clinicians."
)

STOP_STRINGS: List[str] = ["\n\nUser:", "\n\n###"]

MAX_INPUT_TOKENS: int = 1024        # prompt (system + history + user)
MAX_OUTPUT_TOKENS: int = 256


def _truncate(tokens: List[int], max_len: int) -> List[int]:
    """Keep only the last `max_len` tokens."""
    return tokens if len(tokens) <= max_len else tokens[-max_len:]


def build_prompt(
    user_msg: str,
    conversation: str | None = None,
    system_msg: str = SYSTEM_MSG,
) -> str:
    """
    Assemble the text that will be sent into the LLM.

    Parameters
    ----------
    user_msg : str
        The latest user question / instruction.
    conversation : str | None
        Running transcript of previous turns (already formatted).
    system_msg : str
        System‑level behaviour instruction.

    Returns
    -------
    full_prompt : str
    """
    parts: List[str] = [f"### System:\n{system_msg.strip()}"]
    if conversation:
        parts.append(conversation.strip())
    parts.append(f"### User:\n{user_msg.strip()}")
    parts.append("### Assistant:\n")
    return "\n\n".join(parts)



def load_model(
    model_id: str,
    adapter_dir: Path | None = None,
) -> Tuple[torch.nn.Module, "transformers.PreTrainedTokenizer"]:
    """
    Load base model + (optional) LoRA adapter; return model & tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return model, tokenizer



def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterable[str]:
    """
    Yield text fragments as the model produces them (non‑blocking).

    Example
    -------
    for chunk in stream_generate(model, tok, prompt):
        sys.stdout.write(chunk); sys.stdout.flush()
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = _truncate(inputs["input_ids"][0].tolist(), MAX_INPUT_TOKENS)
    inputs["input_ids"] = torch.tensor(input_ids).unsqueeze(0).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=6,
        repetition_penalty=1.25,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True).start()

    for text in streamer:
        if any(stop in text for stop in STOP_STRINGS):
            break
        yield text


def chat_once(
    model,
    tokenizer,
    user_msg: str,
    conversation: str | None = None,
    **gen_cfg,
) -> str:
    """Blocking call that returns the complete answer string."""
    prompt = build_prompt(user_msg, conversation)
    chunks: List[str] = []
    for chunk in stream_generate(
        model, tokenizer, prompt, **({"max_new_tokens": gen_cfg.get("max_new_tokens")} | gen_cfg)
    ):
        chunks.append(chunk)
    return "".join(chunks).strip()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Quick local generation test.")
    parser.add_argument("question", help="User question / prompt")
    parser.add_argument(
        "--model_id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model repo or local path",
    )
    parser.add_argument(
        "--adapter_dir",
        default="adapters/tinyllama-health",
        help="Directory containing LoRA weights (optional)",
    )
    parser.add_argument("--max_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()

    logger.info("Loading base model %s with adapter %s …", args.model_id, args.adapter_dir)
    model, tok = load_model(args.model_id, Path(args.adapter_dir))

    start = time.time()
    answer = chat_once(model, tok, args.question, max_new_tokens=args.max_tokens)
    elapsed = time.time() - start

    logger.info("=== Answer ==================================================")
    sys.stdout.write(answer + "\n")
    logger.info("Tokens streamed in %.1f s", elapsed)