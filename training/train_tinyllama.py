"""
Fine-tune TinyLlama-1.1B on Mac (MPS) using LoRA adapters.

Corpus: data/science_articles/NaS.jsonl
"""

import os
import time
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from knowledge_model.ingestion.upload_s3 import upload_directory

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "adapters/tinyllama-health"


def load_tokenizer() -> Any:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token
    return tok


def load_model(device: str) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    return base.to(device)


def tokenize_dataset(tokenizer, file_path: str) -> Any:
    raw_ds = load_dataset("json", data_files=file_path)["train"]
    raw_ds = raw_ds.remove_columns([col for col in raw_ds.column_names if col not in {"text"}])

    def tokenize(example: dict) -> dict:
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return raw_ds.map(tokenize, batched=False)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = load_tokenizer()
    base_model = load_model(device)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    dataset_path = "data/science_articles/NaS.jsonl"
    dataset = tokenize_dataset(tokenizer, dataset_path)

    args = TrainingArguments(
        output_dir="training/tiny_out",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=25,
        save_steps=250,
        gradient_checkpointing=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    start = time.time()
    print(f"Training started on {len(dataset)} chunks...")
    trainer.train()
    mins, secs = divmod(int(time.time() - start), 60)
    print(f"Training finished in {mins} min {secs} sec")

    if os.path.exists(ADAPTER_DIR):
        import shutil
        shutil.rmtree(ADAPTER_DIR)

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapters saved to {ADAPTER_DIR}")

    upload_directory(
        ADAPTER_DIR,
        bucket="nas-knowledge-model-dataset",
        prefix=f"adapters/{os.path.basename(ADAPTER_DIR)}"
    )
    print("Adapter uploaded to S3")
    print(f"Training complete on {len(dataset)} article chunks for NaS.")


if __name__ == "__main__":
    main()