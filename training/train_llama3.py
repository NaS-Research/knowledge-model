import torch, os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from dotenv import load_dotenv

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"    # <= fits in 16 GB

def main():
    load_dotenv()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token = tok.eos_token                       # avoid warnings

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "mps"} if torch.backends.mps.is_available() else "auto",
    )
    model.config.use_cache = False                      # needed for MPS back‑prop

    ds = load_dataset("json", data_files="data/science_articles/train.jsonl")["train"]
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir="training/tiny_out",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,     # effective batch = 8
        learning_rate=2e‑4,
        logging_steps=20,
        save_steps=200,
        fp16=True,                         # MPS accepts fp16
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model("training/tinyllama-health")

if __name__ == "__main__":
    main()