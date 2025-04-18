"""
Smoke test for the TinyLlama‑Health LoRA adapter.

Quickly loads the base TinyLlama model plus our health‑domain LoRA
weights, generates a few short answers, and prints them.  This is *not*
an evaluation — just a sanity check that the checkpoint is readable and
produces text.

Usage
-----
$ python smoke_test.py
# or specify a different adapter path
$ python smoke_test.py --adapter_dir path/to/adapter
# or specify a single prompt and number of tokens
$ python smoke_test.py --prompt "Explain..." --tokens 400
"""

from __future__ import annotations

import argparse
import textwrap

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "adapters/tinyllama-health"
DEFAULT_MAX_TOKENS = 400
DEFAULT_REP_PEN = 1.35      # repetition_penalty for generation
DEFAULT_NO_REPEAT = 6       # no_repeat_ngram_size
DEFAULT_TEMP = 0.5        # lower temperature for less rambling
DEFAULT_MAX_WORDS = 220    # 0 disables post‑generation truncation
INSTRUCTION_PREFIX = (
    "You are a medical Q&A assistant. "
    "Answer in plain language (≤150 words), avoid numbered headings, "
    "and cite 2–3 PubMed IDs in square brackets.\n\n"
)
PROMPTS: list[str] = [
    INSTRUCTION_PREFIX
    + "Give a concise overview of sporadic cerebral small‑vessel disease: its typical underlying vascular pathology, common MRI findings, and the main avenues being explored for prevention or treatment."
]


def get_device() -> str:
    """Select the best available inference device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(adapter_dir: str, device: str):
    """Load base TinyLlama and merge with the LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model.to(device).eval(), tokenizer


def run_smoke_test(adapter_dir: str, prompts: list[str], tokens: int,
                   rep_pen: float, no_repeat: int, temp: float,
                   max_words: int) -> None:
    device = get_device()
    print(f"Loading base model + adapter on **{device}** …")
    model, tok = load_model(adapter_dir, device)

    print("\n=== Smoke test outputs ===")
    for prompt in prompts:
        prompt = prompt if prompt.startswith(INSTRUCTION_PREFIX) else INSTRUCTION_PREFIX + prompt
        prompt += "\n\nAssistant:"
        inputs = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False  # avoid an <eos> token at the end of the prompt
        ).to(device)
        # stream tokens to stdout as they are generated
        streamer = TextStreamer(tok,
                                skip_prompt=True,
                                skip_special_tokens=True)

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
                streamer=streamer,      # <-- live streaming
            )
        # neat separation between prompts
        print("\n" + "-" * 60)


def main() -> None:
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