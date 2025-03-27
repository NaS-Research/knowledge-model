# test_llama2.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 1) Load tokenizer (use_auth_token=True to access gated models)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=True
    )

    # 2) Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=True,
        device_map="auto"
        # load_in_8bit=True  # optional for memory savings
    )

    # 3) Simple prompt
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4) Generate text
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)

    # 5) Decode and print
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("NaS says:", result)

if __name__ == "__main__":
    main()
