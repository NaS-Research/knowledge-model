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

from pydantic_settings import BaseSettings  # Pydantic v2 moved BaseSettings to this package
from pydantic import Field, validator       # Field & validator still live in core pydantic
import torch


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
    dtype: torch.dtype = Field(
        default=torch.float16,
        description="torch.dtype for model weights – keep fp16 on consumer GPUs",
    )

    # --------------------------------------------------------------------- #
    # Decoding / generation defaults
    # --------------------------------------------------------------------- #
    max_new_tokens: int = 800
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    do_sample: bool = Field(
        default=True,
        description="Enable nucleus/temperature sampling. "
                    "Set False to force greedy decoding."
    )
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

    # --------------------------------------------------------------------- #
    # Validators – convert str → torch.dtype & sanity‑check device
    # --------------------------------------------------------------------- #
    @validator("dtype", pre=True)
    def _as_torch_dtype(cls, v):
        """
        Allow passing 'float16', 'bfloat16' etc. as env‑vars / CLI flags
        and transparently turn them into real torch.dtypes.
        """
        if isinstance(v, str):
            try:
                return getattr(torch, v)
            except AttributeError as exc:
                raise ValueError(f"Unsupported dtype '{v}'") from exc
        return v

    @validator("device", pre=True)
    def _normalise_device(cls, v):
        """
        Accept shorthand like 'cuda' → cuda:0
        """
        if v == "cuda":
            return "cuda:0"
        return v

    class Config:
        env_prefix = ""  # allow bare env names like DEVICE, MAX_NEW_TOKENS
        case_sensitive = False


# Only instantiate once so every import shares the same object
settings = InferenceSettings()

## --------------------------------------------------------------------------- #
## 🔄  Legacy constant aliases (back‑compat with older code paths)
## --------------------------------------------------------------------------- #
## Older parts of the code‑base (e.g. inference/model.py) still reference
## attributes such as `Config.BASE_MODEL_ID` that previously lived on a static
## Config class.  We patch equivalent constants onto the new Pydantic class
## *after* instantiation so they inherit any env‑var / CLI overrides.
InferenceSettings.BASE_MODEL_ID = settings.base_model          # type: ignore[attr-defined]
InferenceSettings.ADAPTER_DIR  = settings.adapter_dir
InferenceSettings.DEVICE       = settings.device
InferenceSettings.DTYPE        = settings.dtype
InferenceSettings.SYSTEM_PROMPT      = settings.system_prompt
InferenceSettings.MAX_NEW_TOKENS     = settings.max_new_tokens
InferenceSettings.TEMPERATURE        = settings.temperature
InferenceSettings.TOP_P              = settings.top_p
InferenceSettings.TOP_K              = settings.top_k
InferenceSettings.REPETITION_PENALTY = settings.repetition_penalty
InferenceSettings.DO_SAMPLE          = settings.do_sample
InferenceSettings.NO_REPEAT_NGRAM_SIZE = settings.no_repeat_ngram_size
# Defaults expected by CLI helpers
InferenceSettings.DEFAULT_ADAPTER_DIR  = settings.adapter_dir
InferenceSettings.DEFAULT_SYSTEM_PROMPT = settings.system_prompt

# --------------------------------------------------------------------------- #
# 🔁  Auto‑export *every* settings field as an UPPER‑CASE constant
# --------------------------------------------------------------------------- #
for _field, _value in settings.model_dump().items():
    const_name = _field.upper()
    # Skip constants we explicitly mapped above
    if not hasattr(InferenceSettings, const_name):
        setattr(InferenceSettings, const_name, _value)


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
        "do_sample": settings.do_sample,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "repetition_penalty": settings.repetition_penalty,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
    }
    if extra:
        kwargs.update(extra)
    return kwargs

# --------------------------------------------------------------------------- #
# 🛠  Back‑compat alias
# --------------------------------------------------------------------------- #
# Older parts of the code‑base (e.g. inference/model.py) expect a top‑level
# object named `Config`.  Instead of updating every call‑site we expose a thin
# alias that points at the new Pydantic class.
#
#     from inference.config import Config
#
# now returns the same object as `InferenceSettings`.
#
Config = InferenceSettings  # type: ignore

# Module‑level passthroughs so callers can import directly:
DEFAULT_ADAPTER_DIR = settings.adapter_dir
DEFAULT_SYSTEM_PROMPT = settings.system_prompt
DEVICE = settings.device
DTYPE = settings.dtype
NO_REPEAT_NGRAM_SIZE = settings.no_repeat_ngram_size
