

"""
Single‑point model loader + convenience helpers for local inference.

Loads the TinyLlama base model, applies the LoRA adapter, and exposes:

    generate(prompt, **overrides)  -> str
    stream(prompt, **overrides)    -> Iterator[str]

Other modules (CLI, FastAPI, etc.) should *only* import from here so
that the model is created once per process.

Author: NaS‑Research
"""

from __future__ import annotations

import time
from typing import Iterator, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)
from peft import PeftModel

# All knob‑tweaking lives in this central config object
from inference.config import Config

# ------------------------------------------------------------------ #
# Lazy singletons – created on first use then cached for the process #
# ------------------------------------------------------------------ #
_tokenizer: AutoTokenizer | None = None
_model: torch.nn.Module | None = None


# -------- internal helpers --------------------------------------- #


def _get_tokenizer() -> AutoTokenizer:
    """
    Load HF tokenizer once and keep it cached.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            Config.BASE_MODEL_ID, use_fast=True
        )
        # Ensure padding / EOS are defined
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def _get_model() -> torch.nn.Module:
    """
    Lazy‑load base TinyLlama, slap on the LoRA adapter, move to the desired
    device & dtype.  Heavy ‑‑ do it once!
    """
    global _model
    if _model is None:
        t0 = time.time()

        base = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto" if Config.DEVICE == "auto" else None,
            low_cpu_mem_usage=True,
        )

        # Inject the adapter
        model = PeftModel.from_pretrained(base, Config.ADAPTER_DIR)

        # Move / cast if an explicit device was requested
        if Config.DEVICE not in ("auto", "cpu"):
            model = model.to(Config.DEVICE)

        model.eval()
        _model = model

        print(
            f"Model + adapter loaded in {time.time() - t0:.1f}s "
            f"on **{Config.DEVICE.upper()}**"
        )
    return _model


def _prepare_prompt(user_prompt: str) -> str:
    """
    Prepend system prompt, strip whitespace, ensure trailing newline.
    """
    user_prompt = user_prompt.strip()
    composite = f"{Config.SYSTEM_PROMPT}\n\n{user_prompt}".rstrip() + _get_tokenizer().eos_token
    return composite


def _default_gen_kwargs() -> Dict[str, Any]:
    """
    Merge Config defaults into GenerationConfig‑style kwargs.
    """
    return dict(
        max_new_tokens=Config.MAX_NEW_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K,
        repetition_penalty=Config.REPETITION_PENALTY,
    )


# -------- public API --------------------------------------------- #


def generate(prompt: str, **overrides) -> str:
    """
    Blocking helper that returns the full decoded answer.

    Example:
        >>> from inference.model import generate
        >>> print(generate("Explain GPCR signalling in 3 bullets."))
    """
    tokenizer = _get_tokenizer()
    model = _get_model()

    full_prompt = _prepare_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(**_default_gen_kwargs() | overrides)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove the prompt portion so the caller only sees the answer
    answer = decoded[len(full_prompt) :].lstrip()
    return answer


def stream(prompt: str, **overrides) -> Iterator[str]:
    """
    Streaming version that yields string chunks as soon as they are produced.

    Yields:
        str chunks (NOT necessarily whole tokens)
    """
    tokenizer = _get_tokenizer()
    model = _get_model()

    full_prompt = _prepare_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_cfg = GenerationConfig(**_default_gen_kwargs() | overrides)

    # Run generation in a background thread so `streamer` produces tokens
    thread = torch._C._spawn.generate_with_streamer(
        model,  # internal util introduced in HF 4.39
        streamer,
        inputs,
        gen_cfg,
    )

    for text in streamer:
        yield text

    thread.join()


# -------- smoke test --------------------------------------------- #


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick model sanity test")
    parser.add_argument("prompt", nargs="*", help="Prompt to send")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output instead of waiting for full reply",
    )
    args = parser.parse_args()

    question = " ".join(args.prompt) or "Tell me about CRISPR in 50 words."

    if args.stream:
        for chunk in stream(question):
            print(chunk, end="", flush=True)
        print()
    else:
        print(generate(question))