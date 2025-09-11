"""src/train.py
Model construction and training utilities for HydraSketch-Φ.
The actual mixed-signal continual-learning implementation is proprietary and
therefore removed from this public refactor – executing any of the functions
below will deliberately raise so that users do not run with dummy logic.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "build_model",
]

def build_model(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
    """Stub that prevents silent fall-back to an empty model.

    The real HydraSketch-Φ model spans photonic-PCM hybrid layers and custom
    CUDA kernels which cannot be open-sourced at this moment.
    """
    raise RuntimeError(
        "Model construction is proprietary and has been stripped from the "
        "public release in accordance with the STRICT NO-FALLBACK RULE."
    )
