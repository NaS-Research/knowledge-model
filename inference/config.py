

"""
inference/config.py
-------------------

Centralised configuration object for everything that happens **after**
training – loading the base model, applying LoRA adapters and controlling
generation parameters.

Edit values here (or via env‑vars) instead of sprinkling “magic numbers”
throughout the code‑base.
"""

from pathlib import Path
from typing import Dict, Any

from pydantic import BaseSettings, Field  # pydantic handles .env + type‑checking


# --------------------------------------------------------------------------- #
# Settings dataclass
# --------------------------------------------------------------------------- #

class InferenceSettings(BaseSettings):
    # --------------------------------------------------------------------- #
    # Model artefacts
    # --------------------------------------------------------------------- #
    base_model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HF model id or local path for the *base* model weights",
    )
    adapter_dir: Path = Field(
        default=Path("adapters/tinyllama-health"),
        description="Folder that contains the LoRA adapter weights & tokenizer",
    )

    # --------------------------------------------------------------------- #
    # Hardware / execution
    # --------------------------------------------------------------------- #
    device: str = Field(
        default="mps",  # overrides: export DEVICE=cuda:0
        description="torch device: 'cuda:0' | 'mps' | 'cpu'",
    )
    dtype: str = Field(
        default="float16",
        description="torch.dtype for model weights – keep fp16 on consumer GPUs",
    )

    # --------------------------------------------------------------------- #
    # Decoding / generation defaults
    # --------------------------------------------------------------------- #
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = False  # set True in serve_fastapi for SSE streaming

    # --------------------------------------------------------------------- #
    # Prompt engineering
    # --------------------------------------------------------------------- #
    system_prompt: str = (
        "You are a helpful biomedical assistant. "
        "Answer clearly in plain language and cite sources when available."
    )

    # --------------------------------------------------------------------- #
    # Miscellaneous
    # --------------------------------------------------------------------- #
    log_level: str = "INFO"
    seed: int = 42

    class Config:
        env_prefix = ""  # allow bare env names like DEVICE, MAX_NEW_TOKENS
        case_sensitive = False


# Only instantiate once so every import shares the same object
settings = InferenceSettings()


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def generation_kwargs(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convenience helper used by CLI / FastAPI.
    Returns a dict you can splat directly into `model.generate(**kwargs)`.
    """
    kwargs: Dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "repetition_penalty": settings.repetition_penalty,
    }
    if extra:
        kwargs.update(extra)
    return kwargs