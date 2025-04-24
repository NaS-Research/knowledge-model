"""
pipeline_runner.py
------------------
Single‑shot orchestration script.

Workflow
1. Ingest one month of PubMed articles (default **February 2013**).
2. Merge the updated raw corpus with instruction pairs (≈80 / 20).
3. Fine‑tune TinyLlama with a LoRA adapter on the mixed corpus.

Run
----
python pipeline_runner.py          # uses default 2013‑02
python pipeline_runner.py --year 2020 --month 06
"""

from __future__ import annotations
import argparse
import logging
import subprocess
from pathlib import Path

from knowledge_model.ingestion.pipeline import _month_query, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("pipeline_runner")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ingest → merge → fine‑tune cycle")
    p.add_argument("--year", default="2013", help="YYYY (default 2013)")
    p.add_argument("--month", default="02", help="MM (default 02 = Feb)")
    p.add_argument("--raw_corpus", default="data/science_articles/NaS.jsonl")
    p.add_argument("--instr_file", default="data/instructions/instruction_pairs.jsonl")
    p.add_argument("--mixed_out", default="data/combined/combined_v2.jsonl")
    p.add_argument("--adapter_out", default="adapters/nicole-v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Ingest the specified month
    query = _month_query(args.year, args.month)
    log.info("=== Ingest step — PubMed query ===\n%s", query)
    run_pipeline(query, chunk_size=1_000)

    # 2) Merge corpus with instruction pairs
    log.info("=== Merge step ===")
    subprocess.run(
        [
            "python", "data_tools/merge_corpora.py",
            "--raw", args.raw_corpus,
            "--instr", args.instr_file,
            "--out", args.mixed_out,
        ],
        check=True,
    )

    # 3) Fine‑tune LoRA
    log.info("=== Fine‑tune step ===")
    subprocess.run(
        [
            "accelerate", "launch", "training/train_lora.py",
            "--model_name_or_path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--train_file", args.mixed_out,
            "--output_dir", args.adapter_out,
            "--num_train_epochs", "2",
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "5e-5",
            "--lora_r", "32",
            "--lora_alpha", "64",
            "--lora_target_modules", "q_proj", "v_proj", "k_proj", "o_proj",
            "--max_seq_length", "1536",
            "--report_to", "none",
        ],
        check=True,
    )

    log.info("Pipeline complete – new adapter saved to %s", Path(args.adapter_out).resolve())


if __name__ == "__main__":
    main()
