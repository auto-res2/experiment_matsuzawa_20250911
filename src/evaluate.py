# src/evaluate.py
"""Evaluation utilities (metrics, energy accounting, plotting)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
#  Safety helpers – duplicated to avoid circular imports
# ---------------------------------------------------------------------------


def abort(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.stderr.flush()
    os.kill(os.getpid(), 1)


# ---------------------------------------------------------------------------
#  Energy accounting helpers
# ---------------------------------------------------------------------------

def current_power_draw_watts() -> float:
    """Query instantaneous power draw of GPU 0 in watts via nvidia-smi.

    If the command is unavailable (e.g. on a CPU-only machine or in CI), the
    function returns 0 so that downstream calculations still succeed.
    """

    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out.split("\n")[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        # Fallback – no GPU or nvidia-smi missing.
        return 0.0


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_accuracy(rounds: List[int], accs: List[float], fig_path: Path):
    """Line plot with value annotation (mirrors single-file script)."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accs, label="MAESTRO-CTGNN", marker="o")
    for x, y in zip(rounds, accs):
        plt.text(x, y, f"{y:.2f}", fontsize=6, va="bottom")
    plt.xlabel("Federated round")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy – Experiment 1")
    plt.legend()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path.name


# ---------------------------------------------------------------------------
#  Lightweight JSON persistence helper
# ---------------------------------------------------------------------------

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)
