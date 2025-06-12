"""
training/train_lora.py
----------------------
Lightweight CLI for fine‑tuning a causal‑LM with Low‑Rank Adaptation (LoRA).

Example
-------
accelerate launch training/train_lora.py \
  --model_name_or_path google/txgemma-2b-predict \
  --train_file data/lora/combined/combined.jsonl \
  --output_dir adapters/txgemma_lora_v1 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --load_in_4bit \
  --lora_r 8 --lora_alpha 16 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max_seq_length 1536
"""

from __future__ import annotations
import argparse, logging, os
import math
from pathlib import Path

import torch
USE_MPS = torch.backends.mps.is_available()
USE_CUDA = torch.cuda.is_available()
DEVICE   = "mps" if USE_MPS else ("cuda" if USE_CUDA else "cpu")

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_lora")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine‑tune for causal‑LMs")
    p.add_argument("--model_name_or_path", default="google/txgemma-2b-predict")
    p.add_argument(
        "--train_file",
        default="data/lora/combined/combined.jsonl",
        help="Path to the training JSONL file (default: %(default)s)",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_seq_length", type=int, default=512)

    # LoRA params
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load base model with 4‑bit NF4 quantization (GPU only)")

    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--report_to", default="none")
    return p.parse_args()

def load_dataset_tokenize(path: str | Path, tok, max_len: int):
    """
    Load a .jsonl corpus that may contain either:
      • raw text chunks   – {"text": "..."}
      • instruction pairs – {"instruction": "...", "output": "..."}
    The function converts every record into a single training string,
    tokenises with padding/truncation, adds causal‑LM labels, and drops
    any record that could not be interpreted.
    """
    ds = load_dataset("json", data_files=str(path), split="train")

    def to_text(rec):
        if rec.get("text"):
            return rec["text"]
        if rec.get("instruction") and rec.get("output"):
            return f"### Instruction:\n{rec['instruction']}\n\n### Response:\n{rec['output']}"
        return None

    def tokenize(rec):
        txt = to_text(rec)
        if not txt:
            return {}
        enc = tok(
            txt,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # map & immediately filter out empty dicts
    tokenised = (
        ds.map(tokenize, remove_columns=ds.column_names)
          .filter(lambda x: x != {})
    )
    return tokenised


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base model %s", args.model_name_or_path)
    if USE_MPS:
        # 1. Load to CPU in fp32 to avoid MPS bf16 crash, then cast ↓
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        base = base.to(torch.float16)       # cast weights
        model = base.to("mps")              # move to MPS
    elif USE_CUDA:
        if args.load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                attn_implementation="eager",
                device_map="auto",
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                attn_implementation="eager",
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True,
            )
    else:  # pure CPU box
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
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
    
    # Ensure model config matches checkpointing choice
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    logger.info("Loading dataset %s", args.train_file)
    train_ds = load_dataset_tokenize(args.train_file, tok, args.max_seq_length)
    try:
        ds_len = len(train_ds)
    except TypeError:
        with open(args.train_file, "r", encoding="utf-8") as fh:
            ds_len = sum(1 for _ in fh)

    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = math.ceil(ds_len / eff_batch) * max(1, int(args.num_train_epochs))
  
    logger.info("Dataset rows=%d • effective batch=%d → max_steps=%d",
                ds_len, eff_batch, total_steps)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=total_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=False,
        bf16=False,
    )
    
    logger.info("Training on %s | fp16=%s | bf16=%s", DEVICE, train_args.fp16, train_args.bf16)

    resume_ckpt = Path(args.output_dir, "trainer_state.json").exists()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=True if resume_ckpt else None)

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()