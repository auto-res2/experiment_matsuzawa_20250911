# src/evaluate.py
"""Evaluation & visualisation utilities."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # head-less rendering on servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_accuracy_curve"]


def plot_accuracy_curve(acc_list: list[float], out_path: str) -> None:  # noqa: D401,E501
    x = np.arange(len(acc_list))
    y = np.array(acc_list) * 100.0
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="Top-1 Acc", linewidth=2)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i % 5 == 0:
            plt.annotate(f"{yi:.1f}", (xi, yi))
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(out_path, bbox_inches="tight", format="pdf")
    finally:
        plt.close()
