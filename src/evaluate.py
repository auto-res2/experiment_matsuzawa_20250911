"""
evaluate.py – evaluation helpers and plotting utilities
"""
from __future__ import annotations

from typing import List

import matplotlib

matplotlib.use("Agg")  # headless backend for servers/CI
import matplotlib.pyplot as plt

__all__ = ["annotate_line_plot"]

def annotate_line_plot(ax: plt.Axes, xs: List[float], ys: List[float]):
    """Write every *(x, y)* tuple directly next to its marker on *ax*.

    Keeping this helper separate from *main.py* avoids code duplication across
    multiple experiments and isolates all Matplotlib-specific logic in one
    module.
    """
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
