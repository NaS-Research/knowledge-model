

#!/usr/bin/env python
"""
merge_corpora.py
----------------
Validate, shuffle, and merge the raw article corpus with instruction pairs,
oversampling instructions so they constitute ~20 % of the final dataset.

Usage:
    python data_tools/merge_corpora.py \
        --raw data/combined/combined_corpus.jsonl \
        --instr data/instructions/instruction_pairs.jsonl \
        --out  data/combined/combined_v2.jsonl \
        --oversample 4
"""

from __future__ import annotations
import argparse
import json
import random
import pathlib
import sys
from typing import Iterable

REQUIRED_KEYS = {"instruction", "input", "output"}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_instructions(records: Iterable[dict]) -> None:
    for idx, rec in enumerate(records, 1):
        if set(rec.keys()) != REQUIRED_KEYS:
            sys.exit(f"[ERROR] Line {idx} has incorrect keys: {rec.keys()}")


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge raw corpus with instruction pairs.")
    p.add_argument("--raw", required=True, type=pathlib.Path, help="Raw article corpus JSONL")
    p.add_argument("--instr", required=True, type=pathlib.Path, help="Instruction pairs JSONL")
    p.add_argument("--out", required=True, type=pathlib.Path, help="Output merged JSONL")
    p.add_argument("--oversample", type=int, default=4,
                   help="How many times to replicate instruction pairs (default 4 ≈20 %)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_records = load_jsonl(args.raw)
    instr_records = load_jsonl(args.instr)

    # validate and shuffle instruction pairs
    validate_instructions(instr_records)
    random.shuffle(instr_records)
    print(f"✅  Validated & shuffled {len(instr_records):,} instruction pairs")

    # oversample instructions
    mixed = raw_records + instr_records * args.oversample
    random.shuffle(mixed)
    print(f"🔀  Mixed corpus size: {len(mixed):,} records "
          f"({len(instr_records)*args.oversample:,} instructions, "
          f"{len(raw_records):,} raw)")

    # ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, mixed)
    print(f"📝  Wrote merged corpus → {args.out.resolve()}")


if __name__ == "__main__":
    main()