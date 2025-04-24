"""
High-level, cached accessors for TinyLlama + LoRA adapter used during local inference.

Example:
    >>> from inference.model import generate, stream
    >>> print(generate("Summarise CRISPR in 30 words."))

Notes:
    This module caches heavyweight objects once per process.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Dict, Iterator, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextGenerationPipeline,
    TextIteratorStreamer,
)

from inference.config import Config

LOGGER = logging.getLogger(__name__)
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[torch.nn.Module] = None
NO_REPEAT_NGRAM_SIZE: int = 6


def _get_tokenizer() -> AutoTokenizer:
    """
    Loads the Hugging Face tokenizer once and caches it for efficiency.
    
    Returns:
        AutoTokenizer: The loaded tokenizer.

    Raises:
        Exception: If loading the tokenizer fails.
    """
    global _tokenizer
    try:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(
                Config.BASE_MODEL_ID, use_fast=True
            )
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            if _tokenizer.eos_token is None:
                _tokenizer.eos_token = "</s>"
    except Exception as exc:
        LOGGER.exception("Failed to load tokenizer: %s", exc)
        raise
    return _tokenizer


def _get_model() -> torch.nn.Module:
    """
    Lazily loads the base TinyLlama model, applies the LoRA adapter, and 
    moves it to the specified device.
    
    Returns:
        torch.nn.Module: The loaded model.

    Raises:
        Exception: If loading the model fails.
    """
    global _model
    try:
        if _model is None:
            t0 = time.time()

            base = AutoModelForCausalLM.from_pretrained(
                Config.BASE_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto" if Config.DEVICE == "auto" else None,
                low_cpu_mem_usage=True,
            )

            model = PeftModel.from_pretrained(base, Config.ADAPTER_DIR)

            if Config.DEVICE not in ("auto", "cpu"):
                model = model.to(Config.DEVICE)

            model.eval()
            _model = model

            LOGGER.info(
                "Model + adapter loaded in %.1fs on **%s**",
                time.time() - t0,
                Config.DEVICE.upper(),
            )
    except Exception as exc:
        LOGGER.exception("Failed to load model: %s", exc)
        raise
    return _model


def _prepare_prompt(user_prompt: str) -> str:
    """
    Constructs a conversation prompt in the TinyLlama‑Chat format.

    Args:
        user_prompt (str): The user's input prompt.

    Returns:
        str: The formatted prompt for the model.
    """
    return (
        f"<|system|>{Config.SYSTEM_PROMPT}</s>"
        f"<|user|>{user_prompt.strip()}</s>"
        f"<|assistant|>"
    )


@functools.cache
def _default_gen_kwargs() -> Dict[str, Any]:
    """
    Merges configuration defaults into a dictionary suitable for generation 
    parameters.

    Returns:
        Dict[str, Any]: The default generation parameters.
    """
    return {
        "max_new_tokens": Config.MAX_NEW_TOKENS,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
        "top_k": Config.TOP_K,
        "repetition_penalty": Config.REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "do_sample": Config.DO_SAMPLE,
    }


def generate(prompt: str, **overrides) -> str:
    """
    Generates a complete response based on the provided prompt.

    Args:
        prompt (str): The prompt to generate a response for.
        **overrides: Additional generation parameters.

    Returns:
        str: The generated response.
    """
    tokenizer = _get_tokenizer()
    model = _get_model()

    full_prompt = _prepare_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig(**_default_gen_kwargs() | overrides)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer = decoded[len(full_prompt) :].lstrip()
    return answer


def stream(prompt: str, **overrides) -> Iterator[str]:
    """
    Streams generated text chunks in real-time as they are produced.

    Args:
        prompt (str): The prompt to generate a response for.
        **overrides: Additional generation parameters.

    Yields:
        Iterator[str]: Chunks of generated text.
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

    def _run_generation():
        with torch.inference_mode():
            model.generate(
                **inputs,
                streamer=streamer,
                generation_config=gen_cfg,
            )

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()

    for chunk in streamer:
        yield chunk

    thread.join()


def load_pipeline(
    adapter_dir: str = Config.ADAPTER_DIR,
    *,
    max_new_tokens: int = Config.MAX_NEW_TOKENS,
    temperature: float = Config.TEMPERATURE,
    top_p: float = Config.TOP_P,
) -> TextGenerationPipeline:
    """
    Loads the model and tokenizer into a TextGenerationPipeline.

    Args:
        adapter_dir (str): Directory of the adapter.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.

    Returns:
        TextGenerationPipeline: The configured text generation pipeline.
    """
    Config.ADAPTER_DIR = adapter_dir

    model = _get_model()
    tokenizer = _get_tokenizer()

    gen_cfg = model.generation_config
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.temperature = temperature
    gen_cfg.top_p = top_p
    gen_cfg.top_k = Config.TOP_K
    gen_cfg.do_sample = Config.DO_SAMPLE
    gen_cfg.repetition_penalty = Config.REPETITION_PENALTY
    gen_cfg.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE

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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
            LOGGER.info(chunk, end="", flush=True)
        print()
    else:
        answer = generate(question)
        LOGGER.info(answer)