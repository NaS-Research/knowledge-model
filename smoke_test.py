"""
Quick sanity‑check that a TinyLlama base model plus a domain LoRA adapter
can be loaded and produce tokens.

Example
-------
$ python smoke_test.py --tokens 128
"""

from __future__ import annotations

import argparse
import functools
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "adapters/tinyllama-health"
DEFAULT_MAX_TOKENS = 400
DEFAULT_REP_PEN = 1.35 
DEFAULT_NO_REPEAT = 6
DEFAULT_TEMP = 0.5 
DEFAULT_MAX_WORDS = 220 
INSTRUCTION_PREFIX = (
    "You are a medical Q&A assistant. "
    "Answer in plain language (≤150 words), avoid numbered headings, "
    "and cite 2–3 PubMed IDs in square brackets.\n\n"
)
PROMPTS: list[str] = [
    INSTRUCTION_PREFIX
    + "Give a concise overview of sporadic cerebral small‑vessel disease: its typical underlying vascular pathology, common MRI findings, and the main avenues being explored for prevention or treatment."
]

logger = logging.getLogger(__name__)


@functools.cache
def get_device() -> str:
    """Select the best available inference device.

    Returns:
        str: The device type ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(adapter_dir: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base TinyLlama and merge with the LoRA adapter.

    Args:
        adapter_dir (str): Path to the LoRA adapter directory.
        device (str): Device type for model loading.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
    except (OSError, ValueError) as e:
        logger.error("Failed to load model or adapter: %s", e)
        raise
    return model.to(device).eval(), tokenizer


def run_smoke_test(adapter_dir: str, prompts: list[str], tokens: int,
                   rep_pen: float, no_repeat: int, temp: float,
                   max_words: int) -> None:
    """Run the smoke test for the model with given parameters.

    Args:
        adapter_dir (str): Path to the LoRA adapter directory.
        prompts (list[str]): List of prompts to test.
        tokens (int): Max new tokens to generate.
        rep_pen (float): Repetition penalty for generation.
        no_repeat (int): No repeat n-gram size.
        temp (float): Sampling temperature.
        max_words (int): Word cap for answer; 0 means no truncation.
    """
    device = get_device()
    logger.info("Loading base model '%s' with adapter '%s' on %s",
                BASE_MODEL, adapter_dir, device)
    model, tok = load_model(adapter_dir, device)

    logger.info("=== Smoke test outputs ===")
    for prompt in prompts:
        prompt = prompt if prompt.startswith(INSTRUCTION_PREFIX) else INSTRUCTION_PREFIX + prompt
        prompt += "\n\nAssistant:"
        inputs = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        streamer = TextStreamer(tok,
                                skip_prompt=True,
                                skip_special_tokens=True)

        try:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=rep_pen,
                    no_repeat_ngram_size=no_repeat,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    streamer=streamer,
                )
        except RuntimeError as e:
            logger.error("Error during generation: %s", e)
        logger.info("-" * 60)


def main() -> None:
    """Main function to run the smoke test.

    Parses command-line arguments and initiates the smoke test.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run a quick forward‑pass smoke test.")
    parser.add_argument(
        "--adapter_dir",
        default=DEFAULT_ADAPTER_DIR,
        help="Path to the LoRA adapter folder to test (default: adapters/tinyllama-health)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to test instead of default list",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--max_words", type=int, default=DEFAULT_MAX_WORDS,
        help="Word cap for answer; 0 means no truncation (default: 220)",
    )
    parser.add_argument("--rep_pen", type=float, default=DEFAULT_REP_PEN,
                        help="repetition_penalty (default: 1.35)")
    parser.add_argument("--no_repeat", type=int, default=DEFAULT_NO_REPEAT,
                        help="no_repeat_ngram_size (default: 6)")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="sampling temperature (default: 0.5)")
    args = parser.parse_args()
    prompts = [args.prompt] if args.prompt else PROMPTS
    run_smoke_test(args.adapter_dir, prompts, args.tokens,
                   args.rep_pen, args.no_repeat, args.temp, args.max_words)


if __name__ == "__main__":
    main()