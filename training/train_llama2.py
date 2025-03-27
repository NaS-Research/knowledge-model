import os
# WARNING: Disabling the MPS high watermark may lead to system instability if memory is exhausted.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

def main():
    # Load environment variables (e.g. HF_TOKEN)
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    # Load the tokenizer (using the slow tokenizer for legacy behavior)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_auth_token=hf_token,
        trust_remote_code=True,
        use_fast=False
    )

    # Ensure the tokenizer has a pad token; if not, use the eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model with BF16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_auth_token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load your training dataset; ensure your JSONL file exists at the given location
    dataset = load_dataset("json", data_files={"train": "data/science_articles/train.jsonl"})

    # Preprocess the dataset: tokenize the "text" field.
    # Note: max_length is reduced from 512 to 256
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_dataset = dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["pmid", "title", "text"]
    )

    # Create a data collator for language modeling (no masked LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments with halved hyperparameters:
    # - num_train_epochs from 4 to 2
    # - per_device_train_batch_size from 2 to 1
    # - gradient_accumulation_steps from 32 to 16
    # - logging_steps from 50 to 25
    # - save_steps from 500 to 250
    training_args = TrainingArguments(
        output_dir="training/output",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=3e-4,
        logging_steps=25,
        save_steps=250,
        eval_strategy="no",
        fp16=False,   # Using BF16 on Apple devices
        bf16=True,
        report_to="none",
        logging_dir="./logs",
        remove_unused_columns=False  # Do not drop necessary fields
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("training/fine_tuned_llama2")
    tokenizer.save_pretrained("training/fine_tuned_llama2")

if __name__ == "__main__":
    main()