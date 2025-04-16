"""
Fine‑tune TinyLlama‑1.1B on your Mac mini (MPS) with LoRA.
Corpus: data/science_articles/train.jsonl
"""

import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token               # silence warning

    # --- load on CPU in fp16, then move to MPS ---
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # cast bfloat16 → fp16
        device_map="cpu",            # load weights on CPU first
        low_cpu_mem_usage=True,
    )
    base = base.to(device)           # now move to "mps"

    # ---- PEFT / LoRA config (very small) ----
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    # ---- Data ----
    raw_ds = load_dataset("json", data_files="data/science_articles/train.jsonl")["train"]

    # Drop non‑text metadata columns before tokenization
    raw_ds = raw_ds.remove_columns([col for col in raw_ds.column_names if col not in {"text"}])

    def tokenize(example):
        out = tok(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = raw_ds.map(tokenize, batched=False)

    # ---- Training args ----
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
        train_dataset=ds,
    )
    trainer.train()

    # ---- Save adapters ----
    adapter_dir = "adapters/tinyllama-health"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Adapters saved to {adapter_dir}")

if __name__ == "__main__":
    main()