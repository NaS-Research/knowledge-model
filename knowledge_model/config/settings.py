"""
Central configuration values used across the knowledge‑model package.
Anything that depends on the *location* of the repository (e.g. the data
directory) lives here so we only have to change it once.

This module is intentionally tiny and dependency‑free.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = _PROJECT_ROOT / "data"

__all__ = ["DATA_ROOT"]