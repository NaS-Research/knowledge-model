"""
inference.config
-----------------
Centralised, strongly‑typed settings object that controls **all** post‑training
inference behaviour: model paths, hardware flags, and generation defaults.

Example:
    ```python
    from inference.config import settings, generation_kwargs

    model_id = settings.base_model
    generate_args = generation_kwargs()
    ```
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

from pydantic_settings import BaseSettings
from pydantic import Field, validator 
import torch


class InferenceSettings(BaseSettings):
    base_model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HF model id or local path for the *base* model weights",
    )
    adapter_dir: Path = Field(
        default=Path("adapters/tinyllama-health"),
        description="Folder that contains the LoRA adapter weights & tokenizer",
    )

    device: str = Field(
        default="mps",  # overrides: export DEVICE=cuda:0
        description="torch device: 'cuda:0' | 'mps' | 'cpu'",
    )
    dtype: torch.dtype = Field(
        default=torch.float16,
        description="torch.dtype for model weights – keep fp16 on consumer GPUs",
    )

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
    stream: bool = False

    system_prompt: str = (
        "You are a helpful biomedical assistant. "
        "Answer clearly in plain language and cite sources when available."
    )

    log_level: str = "INFO"
    seed: int = 42

    @validator("dtype", pre=True)
    def _as_torch_dtype(cls, v):
        """Convert string representation of dtype to torch.dtype.

        Args:
            v: The value to convert.

        Returns:
            Corresponding torch.dtype.

        Raises:
            ValueError: If the dtype is unsupported.
        """
        if isinstance(v, str):
            try:
                return getattr(torch, v)
            except AttributeError as exc:
                logger.error("Unsupported dtype '%s'", v)
                raise ValueError(f"Unsupported dtype '{v}'") from exc
        return v

    @validator("device", pre=True)
    def _normalise_device(cls, v):
        """Normalize device shorthand.

        Args:
            v: The device string to normalize.

        Returns:
            Normalized device string.
        """
        if v == "cuda":
            return "cuda:0"
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = InferenceSettings()


InferenceSettings.BASE_MODEL_ID = settings.base_model
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
InferenceSettings.DEFAULT_ADAPTER_DIR  = settings.adapter_dir
InferenceSettings.DEFAULT_SYSTEM_PROMPT = settings.system_prompt


for _field, _value in settings.model_dump().items():
    const_name = _field.upper()
    if not hasattr(InferenceSettings, const_name):
        setattr(InferenceSettings, const_name, _value)


def generation_kwargs(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a dict of `model.generate` kwargs.

    Args:
        extra: Optional overrides that will update the defaults.

    Returns:
        Fully‑populated kwargs dictionary ready to splat into
        `transformers.PreTrainedModel.generate`.

    Notes:
        Pure helper – no I/O or side effects.
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


Config = InferenceSettings

DEFAULT_ADAPTER_DIR = settings.adapter_dir
DEFAULT_SYSTEM_PROMPT = settings.system_prompt
DEVICE = settings.device
DTYPE = settings.dtype
NO_REPEAT_NGRAM_SIZE = settings.no_repeat_ngram_size
