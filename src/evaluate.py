"""src/evaluate.py
Evaluation utilities (figures & potential metric aggregation).
No behavioural change vs. original *figures.py* – code merely relocated.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from .train import get_logger  # shared helper

sns.set_style("whitegrid")
logger = get_logger("evaluate")

# ---------------------------------------------------------------------
# Plotting helpers (verbatim from the monolithic script)
# ---------------------------------------------------------------------

def line_plot(
    x: List[float],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: Path,
):
    """Save a PDF line-plot with numeric annotations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label=title, marker="o")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved line plot → %s", filename)


def bar_plot(
    categories: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    filename: Path,
):
    """Save a PDF bar-plot with numeric annotations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(categories, values, color="skyblue")
    for idx, val in enumerate(values):
        ax.annotate(
            f"{val:.2f}", (idx, val), textcoords="offset points", xytext=(0, 3), ha="center"
        )
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved bar plot → %s", filename)
