"""
inference package
=================

High‑level helpers for **running the fine‑tuned TxGemma‑2B locally**.

Sub‑modules
-----------
cli_chat      – small CLI wrapper (`python -m inference.cli_chat`)
config        – central configuration (model path, device, gen kwargs, …)
model         – lazy loader + singleton access to the PEFT model / tokenizer
postprocess   – clean‑up utilities for raw generation output
server        – FastAPI web server exposing `/chat` endpoint

Typical usage
-------------
```python
from inference.model import get_pipeline
from inference.postprocess import clean_generation

generator = get_pipeline()
raw = generator("Tell me a joke")[0]["generated_text"]
print(clean_generation(raw))
```
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "config",
    "model",
    "postprocess",
]

# --------------------------------------------------------------------------- #
# Lazy sub‑module loader
# --------------------------------------------------------------------------- #
# Import sub‑modules only when they are first accessed so that running
# `python -m inference.<module>` does not trigger a duplicate import warning.
def __getattr__(name: str) -> ModuleType:  # PEP 562
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if TYPE_CHECKING:  # help static analysers & editors
    import inference.config as config
    import inference.model as model
    import inference.postprocess as postprocess