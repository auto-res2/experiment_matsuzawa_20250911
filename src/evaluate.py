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


def _run_cmd(cmd: List[str]) -> str:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as exc:
        abort(f"Command {cmd} failed: {exc.stderr}\n{exc}")
        return ""  # unreachable – keeps mypy happy


def current_power_draw_watts() -> float:
    """Query instantaneous power draw of GPU 0 in watts via nvidia-smi."""
    out = _run_cmd(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    return float(out.split("\n")[0])


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_accuracy(rounds, accs, fig_path: Path):
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
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path.name


# ---------------------------------------------------------------------------
#  Lightweight JSON persistence helper
# ---------------------------------------------------------------------------

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)
