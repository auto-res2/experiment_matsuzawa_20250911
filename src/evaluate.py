# src/evaluate.py
"""Evaluation and visualisation helpers.
Keeps plotting in its own module so that train.py can stay free from any
matplotlib import cost when running on a headless GPU cluster.
"""
from pathlib import Path
from typing import Sequence

import matplotlib

# Use a non-interactive backend that works without an X-server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 – after backend selection


def line_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Save a labelled PDF line plot with value annotations."""
    plt.figure()  # new figure for thread safety
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.as_posix(), bbox_inches="tight", format="pdf")
    plt.close()
