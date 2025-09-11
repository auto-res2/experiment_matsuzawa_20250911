# src/evaluate.py
# -------------------------------------------------------------
# Evaluation helpers: serialization & plotting utilities.
# -------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib

matplotlib.use("pdf")  # enforce non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

RESULTS_DIR = Path(".research/iteration8")
FIGURES_DIR = RESULTS_DIR / "images"

__all__ = [
    "save_results",
    "line_plot",
]


def save_results(name: str, payload: Dict[str, Any]) -> None:
    """Save `payload` to *.json inside .research/iteration8/ and echo it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # Print to STDOUT for CI visibility
    print("\n=====", name.upper(), "RESULTS =====")
    print(json.dumps(payload, indent=2))


def line_plot(xs: List[float], ys: List[float], xlabel: str, ylabel: str, title: str, filename: str) -> None:  # noqa: N802
    """Generate a PDF line-plot with value annotations."""
    sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.2)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", label=ylabel)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Generated figure: {path}")
