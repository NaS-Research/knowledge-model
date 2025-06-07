

"""
Shared utilities for the ingestion sub‑package.

We import `orjson` once and re‑export it under the name `json` so that
all ingestion modules can simply write

    from knowledge_model.ingestion import json

and transparently get the faster encoder / decoder.
"""

import orjson as json

# Re‑export the most common helpers so star‑imports keep working
dumps = json.dumps       # bytes → already UTF‑8
loads = json.loads

__all__ = ["json", "dumps", "loads"]