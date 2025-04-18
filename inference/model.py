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
import threading
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
        # TinyLlama‑Chat uses the same EOS token for </s>
        if _tokenizer.eos_token is None:
            _tokenizer.eos_token = "</s>"
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
    Build a TinyLlama‑Chat style conversation prompt:
 
        <|system|> ... </s><|user|> ... </s><|assistant|>
 
    The model was fine‑tuned on this template, so using the
    correct role tokens eliminates the runaway gibberish.
    """
    return (
        f"<|system|>{Config.SYSTEM_PROMPT}</s>"
        f"<|user|>{user_prompt.strip()}</s>"
        f"<|assistant|>"
    )


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
        no_repeat_ngram_size=6,
        do_sample=Config.DO_SAMPLE,
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
    Streaming version that yields text chunks as soon as they are produced.

    Example:
        >>> for chunk in stream("Summarise CRISPR"):
        ...     sys.stdout.write(chunk)
    """
    tokenizer = _get_tokenizer()
    model = _get_model()

    full_prompt = _prepare_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_cfg = GenerationConfig(**_default_gen_kwargs() | overrides)

    # Launch generation in a background *Python* thread so that the
    # `streamer` iterator can yield pieces of the response synchronously.
    def _run_generation():
        with torch.inference_mode():
            model.generate(
                **inputs,
                streamer=streamer,
                generation_config=gen_cfg,
            )

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()

    # As soon as new text is available in the streamer we yield it to the caller
    for chunk in streamer:
        yield chunk

    thread.join()


# -------- pipeline helper ---------------------------------------- #

from transformers import TextGenerationPipeline


def load_pipeline(
    adapter_dir: str = Config.ADAPTER_DIR,
    *,
    max_new_tokens: int = Config.MAX_NEW_TOKENS,
    temperature: float = Config.TEMPERATURE,
    top_p: float = Config.TOP_P,
) -> TextGenerationPipeline:
    """
    Convenience wrapper used by `inference.cli_chat` (and future front‑ends).

    It loads the model + tokenizer via our cached helpers, applies generation
    defaults, and returns a ready‑to‑use HF `TextGenerationPipeline`.
    """
    # Ensure the requested adapter path propagates to the global Config before
    # the underlying helpers are called.
    Config.ADAPTER_DIR = adapter_dir

    model = _get_model()           # loads + caches with adapter
    tokenizer = _get_tokenizer()

    # Sync generation defaults
    gen_cfg = model.generation_config
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.temperature = temperature
    gen_cfg.top_p = top_p
    gen_cfg.top_k = Config.TOP_K
    gen_cfg.do_sample = Config.DO_SAMPLE
    gen_cfg.repetition_penalty = Config.REPETITION_PENALTY
    gen_cfg.no_repeat_ngram_size = 6

    return TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device.index if model.device.type == "cuda" else -1,
        batch_size=1,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        do_sample=gen_cfg.do_sample,
        repetition_penalty=gen_cfg.repetition_penalty,
        no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
        return_full_text=True,
    )


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