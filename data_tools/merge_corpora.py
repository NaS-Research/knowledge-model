#!/usr/bin/env python
"""
merge_corpora.py

Combine a raw biomedical‑article corpus with an instruction‑tuning set, writing a
single shuffled JSONL that can be used for fine‑tuning.

Example
-------
$ python data_tools/merge_corpora.py \
    --raw data/combined/combined_corpus.jsonl \
    --instr data/instructions/instruction_pairs.jsonl \
    --out data/combined/combined_v2.jsonl \
    --oversample 4
"""

from __future__ import annotations

# Standard library
import argparse
import json
import logging
import pathlib
import random
import sys
from typing import Iterable, List, Any


REQUIRED_KEYS: set[str] = {"instruction", "input", "output"}
DEFAULT_OVERSAMPLE: int = 4
LOG_FORMAT: str = "%(levelname)s: %(message)s"
LOG_LEVEL: int = logging.INFO
RANDOM_SEED: int | None = None  # set to int for reproducible shuffles


def _configure_logging() -> None:
    """Configure root logger once for CLI usage."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, force=True)


def load_jsonl(path: pathlib.Path) -> List[dict[str, Any]]:
    """Load a JSON‑Lines file into a list.

    Args:
        path: Path to a ``*.jsonl`` file.

    Returns:
        A list with one :class:`dict` per line.

    Raises:
        FileNotFoundError: If *path* does not exist.
        SystemExit: If any line cannot be decoded as JSON.
    """
    if not path.exists():
        logging.error("File not found: %s", path)
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for ln_no, raw in enumerate(fh, 1):
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:  # pragma: no cover
                logging.error("Line %s in %s is not valid JSON: %s", ln_no, path, exc)
                sys.exit(1)
    return records


def validate_instructions(records: Iterable[dict[str, Any]]) -> None:
    """Validate that each record has exactly the required keys.

    Args:
        records: Iterable of instruction‑pair dictionaries.

    Raises:
        SystemExit: If any record is missing or has extra keys.
    """
    for idx, rec in enumerate(records, 1):
        if set(rec) != REQUIRED_KEYS:
            logging.error(
                "Instruction pair at line %d has incorrect keys: %s", idx, sorted(rec.keys())
            )
            sys.exit(1)


def write_jsonl(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> None:
    """Write *records* to *path* in JSON‑Lines format.

    Args:
        path: Destination file.
        records: Iterable of serialisable dictionaries.
    """
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge a raw corpus with instruction pairs, "
            "optionally oversampling the instructions so they comprise ≈20 % of the output."
        )
    )
    parser.add_argument("--raw", required=True, type=pathlib.Path, help="Raw article corpus JSONL")
    parser.add_argument(
        "--instr", required=True, type=pathlib.Path, help="Instruction pairs JSONL"
    )
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output merged JSONL")
    parser.add_argument(
        "--oversample",
        type=int,
        default=DEFAULT_OVERSAMPLE,
        help=f"Replication factor for instructions (default {DEFAULT_OVERSAMPLE})",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    """CLI entry‑point."""
    _configure_logging()
    args = parse_args()

    if RANDOM_SEED is not None:  # deterministic runs if a seed is chosen
        random.seed(RANDOM_SEED)

    raw_records = load_jsonl(args.raw)
    instr_records = load_jsonl(args.instr)

    validate_instructions(instr_records)

    random.shuffle(instr_records)
    logging.info("Validated & shuffled %,d instruction pairs", len(instr_records))

    mixed: list[dict[str, Any]] = raw_records + instr_records * args.oversample
    random.shuffle(mixed)
    logging.info(
        "Mixed corpus size: %,d (%,d instructions, %,d raw)",
        len(mixed),
        len(instr_records) * args.oversample,
        len(raw_records),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, mixed)
    logging.info("Merged corpus written → %s", args.out.resolve())


if __name__ == "__main__":
    main()