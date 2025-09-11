from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import dgl

from .preprocess import RESULTS_DIR, FIG_DIR


def evaluate(
    model: torch.nn.Module,
    g: dgl.DGLGraph,
    idx: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute ROC-AUC for binary node classification."""
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata["feat"])[idx]
        pred = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        y_true = labels[idx].cpu().numpy()
        return float(roc_auc_score(y_true, pred))


def plot_seed_auc(dataset: str, auc_seeds: List[float]) -> str:
    """Create bar-plot of ROC-AUC per seed and return figure filename."""
    fig_name = f"accuracy_{dataset}.pdf"
    xs = list(range(len(auc_seeds)))
    plt.figure(figsize=(6, 4))
    plt.bar(xs, auc_seeds, color="skyblue")
    for x, v in zip(xs, auc_seeds):
        plt.text(x, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("ROC-AUC")
    plt.xlabel("Seed index")
    plt.title(f"Experiment-1 ROC-AUC – {dataset}")
    plt.savefig(FIG_DIR / fig_name, bbox_inches="tight")
    plt.close()
    return fig_name


def save_metrics_json(prefix: str, metrics: Dict[str, Any]) -> Path:
    """Save metrics dict as JSON in the prescribed research directory and echo to stdout."""
    out_path = RESULTS_DIR / f"{prefix}_{metrics['dataset']}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # Print the JSON immediately for the framework validator
    print(f"\n[JSON dump → {out_path}]")
    print(json.dumps(metrics, indent=2))
    return out_path
