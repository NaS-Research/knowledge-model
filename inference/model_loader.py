from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, psutil
import torch
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:          # bitsandbytes not available on this machine
    _HAS_BNB = False

def load_finetuned_model(
    base_id: str = "google/txgemma-2b-chat",
    adapter_dir: str | Path = "adapters/txgemma_lora_instr_v1",
):
    """
    Return (tokenizer, LoRA‑augmented Gemma‑2B model) in a Mac‑friendly,
    memory‑conservative configuration.

    * 4‑bit quantisation when < 24 GB RAM, else 8‑bit.  
    * Eager attention so Metal‑backend runs efficiently.  
    * `device_map="auto"` lets HF shard layers between CPU and M‑series GPU.  
    """
    # Disable CUDA‑style allocator warm‑up that allocates > 50 % RAM on Apple Silicon
    os.environ["TRANSFORMERS_NO_EXPERIMENTAL_CACHING_ALLOCATOR"] = "1"

    quant_cfg = None
    if _HAS_BNB and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        # Decide 4‑ vs 8‑bit based on available RAM
        mem_gb = psutil.virtual_memory().total / 2**30
        use_4bit = mem_gb < 24
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            load_in_8bit=not use_4bit,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    # Ensure a valid pad token (Gemma models sometimes omit one)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        attn_implementation="eager",      # keep module tree compatible with LoRA
        torch_dtype="float16",            # Metal backend prefers fp16
        device_map=None,                  # single‑device load; avoids off‑load index
        quantization_config=quant_cfg,
    ).eval()

    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        device_map="auto",
    ).eval()

    model = model.to("cpu")  # ensure all weights materialised on same device

    return tokenizer, model