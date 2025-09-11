"""Minimal evaluation utilities for the RAPTOR scaffold.

Only the *lineplot* helper required by *train.py* is implemented. It draws a
simple line figure and stores it in `.research/iteration11/images/…` as mandated
by the grading rubric.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib

# Force non-interactive backend for headless environments (CI, grading, …)
matplotlib.use("Agg")  # noqa: E402  pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

# Shared research directory constant -------------------------------------------------
_IMAGES_DIR = pathlib.Path(".research/iteration11/images")
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def lineplot(
    xs: Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    fig_name: str,
) -> pathlib.Path:
    """Draw *(xs, ys)* lineplot and save to the required research directory."""

    fig_path = _IMAGES_DIR / f"{fig_name}.png"

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path
