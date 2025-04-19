"""
training/train_lora.py
----------------------
Lightweight CLI for fine‑tuning a causal‑LM with Low‑Rank Adaptation (LoRA).

Example
-------
accelerate launch training/train_lora.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file data/combined/combined_v2.jsonl \
  --output_dir adapters/nicole-v2 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_r 16 --lora_alpha 32 \
  --lora_target_modules q_proj v_proj k_proj o_proj \
  --max_seq_length 1536
"""

from __future__ import annotations
import argparse, logging, os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_lora")


# --------------------------------------------------------------------------- #
# Args                                                                        #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine‑tune for causal‑LMs")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_seq_length", type=int, default=1536)

    # LoRA params
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])

    p.add_argument("--report_to", default="none")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset helper                                                              #
# --------------------------------------------------------------------------- #
def load_dataset_tokenize(path: str | Path, tok, max_len: int):
    ds = load_dataset("json", data_files=str(path), split="train")

    def to_text(rec):
        if "text" in rec:
            return rec["text"]
        if "instruction" in rec and "output" in rec:
            return f"### Instruction:\n{rec['instruction']}\n\n### Response:\n{rec['output']}"
        raise ValueError("Record missing 'text' or instruction keys")

    def tokenize(rec):
        return tok(to_text(rec), truncation=True, max_length=max_len)

    return ds.map(tokenize, remove_columns=ds.column_names)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base model %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info("Adding LoRA adapters …")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tok.pad_token = tok.eos_token

    logger.info("Loading dataset %s", args.train_file)
    train_ds = load_dataset_tokenize(args.train_file, tok, args.max_seq_length)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    logger.info("Starting training …")
    trainer.train()
    logger.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
