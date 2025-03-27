import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def main():
    # Specify the model repository; you might use the quantized Llama 3 8B model from Hugging Face
    model_name_or_path = "meta-llama/Llama-3.1-8B"  # adjust if needed

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=True)

    # Load your JSONL dataset (assuming it is in data/science_articles/train.jsonl)
    dataset = load_dataset("json", data_files={"train": "data/science_articles/train.jsonl"})
    
    # Define a data collator (for causal language modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="training/output",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=3e-4,
        logging_steps=50,             # Log every 50 steps
        save_steps=500,
        evaluation_strategy="no",
        fp16=True,
        report_to="none",             # Set to "wandb" or "tensorboard" if desired
        logging_dir="./logs",         # Directory for logs (optional)
    )
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained("training/fine_tuned_llama3")
    tokenizer.save_pretrained("training/fine_tuned_llama3")

if __name__ == "__main__":
    main()