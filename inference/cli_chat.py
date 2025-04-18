
"""
inference.cli_chat
------------------

Tiny, single–file command‑line chat interface for your fine‑tuned
TinyLlama‑Health model (+ LoRA adapters).

❯ python -m inference.cli_chat                       # interactive REPL
❯ python -m inference.cli_chat --prompt "Hello"      # one‑shot generation

This script deliberately keeps *only* glue‑code.  Loading the model,
building prompts and cleaning output live in helper modules so that
Web/FastAPI/Gradio front‑ends can reuse the same logic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from inference import config                       # device + defaults
from inference.model import load_pipeline          # returns TextGenerationPipeline
from inference.prompt_utils import build_prompt    # wraps system + user
from inference.postprocess import clean            # trims loops / sections

# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    """Collect command‑line flags."""
    parser = argparse.ArgumentParser(
        description="Chat with TinyLlama‑Health in your terminal"
    )
    # one‑shot mode
    parser.add_argument("--prompt", help="User prompt; skip REPL when supplied")

    # model / generation params
    parser.add_argument(
        "--adapter",
        default=config.DEFAULT_ADAPTER_DIR,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--max_new", type=int, default=256, help="Maximum new tokens to generate"
    )
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top‑p nucleus value")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print tokens as they arrive (non‑blocking)",
    )

    # system prompt override
    parser.add_argument(
        "--system",
        help="Override default system prompt (see inference/config.py)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Chat helpers
# --------------------------------------------------------------------------- #


def _generate(
    pipe,
    system_prompt: str,
    user_msg: str,
    stream: bool,
    max_new: int,
    temp: float,
    top_p: float,
) -> str:
    """Run one generation cycle (optionally streamed)."""
    prompt = build_prompt(system_prompt, user_msg)

    t0 = time.time()
    if stream:  # token streaming ─ print as they appear
        answer: list[str] = []
        for token in pipe(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            stream=True,
        ):
            fragment = token["generated_text"][-1]
            answer.append(fragment)
            sys.stdout.write(fragment)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        output = "".join(answer)
    else:
        # one‑shot completion
        raw = pipe(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            stream=False,
        )[0]["generated_text"]
        output = raw.split(prompt, 1)[-1]  # drop echoed prompt
        print(clean(output))

    dt = time.time() - t0
    print(f"⏱️  {dt:.2f}s | {len(output.split())} words")

    return output


def _repl(pipe, system_prompt: str, args: argparse.Namespace) -> None:
    """Interactive loop."""
    print("Type 'exit' or ':q' to quit.\n")
    while True:
        try:
            user_msg = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_msg.lower() in {"exit", "quit", ":q"}:
            break

        _generate(
            pipe,
            system_prompt,
            user_msg,
            stream=args.stream,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
        )


def main() -> None:
    args = _parse_args()

    # sanity‑check adapter directory
    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        sys.exit(f"[ERR] adapter dir not found: {adapter_dir}")

    print(f"Loading base model + adapter on **{config.DEVICE}** …")
    pipe = load_pipeline(
        adapter_dir=str(adapter_dir),
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_p=args.top_p,
        stream=args.stream,
    )

    system_prompt = args.system or config.DEFAULT_SYSTEM_PROMPT

    if args.prompt:
        # one‑shot generation
        _generate(
            pipe,
            system_prompt,
            args.prompt,
            stream=args.stream,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
        )
    else:
        # interactive chat
        _repl(pipe, system_prompt, args)


if __name__ == "__main__":
    main()