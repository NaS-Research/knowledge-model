import os
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Specify the model repository and your Hugging Face token
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_TOKEN")  # Ensure HF_TOKEN is set in your .env file

    # Load tokenizer and model using the token for authentication
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=hf_token)

    # Load the training dataset (make sure the file exists at data/science_articles/train.jsonl)
    dataset = load_dataset("json", data_files={"train": "data/science_articles/train.jsonl"})
    
    # Define a data collator for causal language modeling (MLM is disabled)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments with reduced per-device batch size and increased gradient accumulation to lower memory usage
    training_args = TrainingArguments(
        output_dir="training/output",
        num_train_epochs=4,
        per_device_train_batch_size=2,          # Reduced batch size
        gradient_accumulation_steps=32,         # Increased accumulation to keep effective batch size similar
        learning_rate=3e-4,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        fp16=True,
        report_to="none",
        logging_dir="./logs",
    )
    
    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("training/fine_tuned_llama3")
    tokenizer.save_pretrained("training/fine_tuned_llama3")

if __name__ == "__main__":
    main()