# src/evaluate.py
"""Plotting & evaluation utilities (figures go to ``.research/iteration2/images``).
A non-interactive backend is enforced to guarantee headless execution on CI
servers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend – required for CI / servers
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import seaborn as sns  # noqa: E402

# -----------------------------------------------------------------------------
#  DIRECTORIES
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / ".research" / "iteration2" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  PUBLIC PLOTTING API
# -----------------------------------------------------------------------------

def save_line_plot(
    xs: Sequence[int] | Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    """Create a simple line plot (*xs*, *ys*) and persist it as PDF.  The
    function returns the *basename* of the saved file so that callers can
    store it in result JSON without serialising absolute paths which tend to
    vary across machines.
    """
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=xs, y=ys, marker="o", lw=2, label=title)

    # annotate points with values for quick inspection
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    path = IMAGE_DIR / f"{filename}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path.name
