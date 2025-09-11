"""src/evaluate.py
------------------------------------------------------------------------
All evaluation / statistical analysis utilities that *train.py* or
external users may wish to import.  For the purpose of this refactor the
file only contains minimal scaffolding.
------------------------------------------------------------------------"""
from __future__ import annotations

import numpy as np


def compute_accuracy(y_true, y_pred) -> float:  # noqa: ANN001
    """Return classification accuracy as float in [0, 1]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between ground-truth and predictions")
    return float((y_true == y_pred).mean())
