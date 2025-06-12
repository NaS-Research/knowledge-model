from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from inference import config
from inference.model import load_pipeline, stream as model_stream 
from inference.postprocess import clean


def _build_chat_prompt(
    system_prompt: str,
    history: list[tuple[str, str]],
    user_msg: str,
) -> str:
    """
    Compose a **TxGemma‑Chat** style conversation prompt that includes the
    whole history plus the new user message:

        <start_of_turn>system
        ...
        <end_of_turn>
        <start_of_turn>user
        ...
        <end_of_turn>
        <start_of_turn>model
        ...
        <end_of_turn>
        ...
        <start_of_turn>user
        NEW
        <end_of_turn>
        <start_of_turn>model

    TxGemma‑Chat recognises the role tags, so it will continue the dialogue naturally.
    """
    parts: list[str] = [f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"]
    for role, text in history:
        parts.append(f"<start_of_turn>{role}\n{text}<end_of_turn>\n")
    parts.append(f"<start_of_turn>user\n{user_msg.strip()}<end_of_turn>\n<start_of_turn>model\n")
    return "".join(parts)


def _parse_args() -> argparse.Namespace:
    """Collect command‑line flags."""
    parser = argparse.ArgumentParser(
        description="Chat with TxGemma‑Health in your terminal"
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
        "--max_new",
        type=int,
        default=256,
        help="Ceiling for new tokens (auto‑shrinks if it would exceed context window)",
    )
    parser.add_argument("--temp", type=float, default=0.65, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.92, help="Top‑p nucleus value")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print tokens as they arrive (non‑blocking)",
    )
    parser.add_argument("--top_k", type=int, default=0, help="Top‑k filtering")
    parser.add_argument(
        "--rep_pen", type=float, default=1.05, help="Repetition penalty"
    )
    parser.add_argument(
        "--no_repeat", type=int, default=3, help="No‑repeat n‑gram size"
    )
    
    parser.add_argument(
        "--word_limit",
        type=int,
        default=0,
        help="If >0, truncate the assistant's reply to this many words.",
    )

    parser.add_argument(
        "--system",
        help="Override default system prompt (see inference/config.py)",
    )
    return parser.parse_args()


def _generate(
    pipe,
    system_prompt: str,
    user_msg: str,
    history: list[tuple[str, str]],
    *,
    stream: bool,
    max_new: int,
    temp: float,
    top_p: float,
    args: argparse.Namespace,
    word_limit: int,
) -> str:
    """Run one generation cycle (optionally streamed)."""
    prompt = _build_chat_prompt(system_prompt, history, user_msg)
 
    TOK_CTX = 2048
    prompt_len = pipe.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
    if prompt_len + max_new > TOK_CTX:
        max_new = max(64, TOK_CTX - prompt_len)
        print(f"[info] max_new truncated to {max_new} to fit context window.")

    t0 = time.time()
    if stream:
        # use the low‑level streaming helper to avoid unsupported kwargs
        answer: list[str] = []
        for fragment in model_stream(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=args.rep_pen,
            no_repeat_ngram_size=args.no_repeat,
            top_k=args.top_k,
        ):
            answer.append(fragment)
            sys.stdout.write(fragment)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        output = "".join(answer)
    else:
        raw = pipe(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            top_k=args.top_k,
            repetition_penalty=args.rep_pen,
            no_repeat_ngram_size=args.no_repeat,
        )[0]["generated_text"]
        output = raw.split(prompt, 1)[-1]
        print(clean(output))

    if output and output[-1] not in ".!?":
        tail: str = ""
        for fragment in model_stream(
            _build_chat_prompt(system_prompt, history + [("user", ""), ("assistant", output)], ""),
            max_new_tokens=20,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=args.rep_pen,
            no_repeat_ngram_size=args.no_repeat,
            top_k=args.top_k,
        ):
            tail += fragment
            if stream:
                sys.stdout.write(fragment)
                sys.stdout.flush()
            if fragment and fragment[-1] in ".!?":
                break
        output += tail
        if not stream and tail:
            print(tail, end="", flush=True)

    if word_limit and len(output.split()) > word_limit:
        output_words = output.split()[:word_limit]
        output = " ".join(output_words).rstrip(" ,;") + "."
        if stream:
            sys.stdout.write("\r" + output + " " * 10 + "\n")
            sys.stdout.flush()
        else:
            print("\n[truncated to", word_limit, "words]")

    history.append(("user", user_msg))
    history.append(("assistant", output))

    dt = time.time() - t0
    print(f"⏱️  {dt:.2f}s | {len(output.split())} words")

    return output


def _repl(pipe, system_prompt: str, args: argparse.Namespace) -> None:
    """Interactive loop."""
    history: list[tuple[str, str]] = []
    print("Type 'exit' or ':q' to quit.\n")
    print("Type ':reset' to clear chat history.  (Use --help for generation flags)")
    while True:
        try:
            user_msg = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_msg.lower() in {"exit", "quit", ":q"}:
            break

        if user_msg.lower() in {":reset"}:
            history.clear()
            print("[history cleared]")
            continue

        _generate(
            pipe,
            system_prompt,
            user_msg,
            history,
            stream=args.stream,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            args=args,
            word_limit=args.word_limit,
        )


def main() -> None:
    args = _parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        sys.exit(f"[ERR] adapter dir not found: {adapter_dir}")

    print(f"Loading TxGemma base model + adapter on **{config.DEVICE}** …")
    pipe = load_pipeline(
        adapter_dir=str(adapter_dir),
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_p=args.top_p,
    )

    system_prompt = args.system or config.DEFAULT_SYSTEM_PROMPT

    if args.prompt:
        _generate(
            pipe,
            system_prompt,
            args.prompt,
            [],
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            args=args,
            word_limit=args.word_limit,
        )
    else:
        # interactive chat
        _repl(pipe, system_prompt, args)


if __name__ == "__main__":
    main()
