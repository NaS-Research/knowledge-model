import os
# Force Transformers / Accelerate to skip bitsandbytes to avoid CPU‑only errors
os.environ["ACCELERATE_DISABLE_BITSANDBYTES"] = "1"

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
import time
import threading

OFFLOAD_DIR = Path(".hf_offload")
OFFLOAD_DIR.mkdir(exist_ok=True)


ADAPTER_PATH = Path("adapters/txgemma_lora_instr_v1")
BASE_MODEL   = "google/txgemma-2b-predict"
TEST_QUESTION = (
    "List red-flag symptoms of acute myocardial infarction."
)


def _spinner(msg: str, stop_event: threading.Event) -> None:
    """Simple console spinner shown while heavy work happens."""
    symbols = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    while not stop_event.is_set():
        sys.stderr.write(f"\r{msg} {symbols[idx % len(symbols)]}")
        sys.stderr.flush()
        idx += 1
        time.sleep(0.1)
    # clear line
    sys.stderr.write("\r" + " " * (len(msg) + 2) + "\r")


def load_model():
    """
    Load the TxGemma base model on CPU (bfloat16) and attach the LoRA adapter.
    Displays a small spinner so users can see that work is ongoing.
    """
    stop = threading.Event()
    t = threading.Thread(target=_spinner, args=("Loading base model…", stop))
    t.start()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        torch_dtype="bfloat16",
        device_map={"": "cpu"},
    )

    stop.set()
    t.join()
    print("Base model loaded.")

    print("Attaching LoRA adapter…", file=sys.stderr)
    model = PeftModel.from_pretrained(
        base,
        ADAPTER_PATH,
        device_map={"": "cpu"},
        offload_folder=str(OFFLOAD_DIR),
    )
    print("Adapter attached.", file=sys.stderr)
    return model


def generate_answer(model, prompt: str, max_new_tokens: int = 120) -> str:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(output[0], skip_special_tokens=True)


def main() -> None:
    print("Loading model and adapter…")
    model = load_model()

    prompt = f"### Instruction:\n{TEST_QUESTION}\n\n### Response:"
    answer = generate_answer(model, prompt)

    print("\n--- Generated answer ---\n")
    print(answer)
    print("\n------------------------\n")


# PyTest-style assertion (optional)
def test_adapter_smoke():
    model = load_model()
    prompt = f"### Instruction:\n{TEST_QUESTION}\n\n### Response:"
    answer = generate_answer(model, prompt, max_new_tokens=80)
    assert "chest" in answer.lower() or "pain" in answer.lower()


if __name__ == "__main__":
    main()