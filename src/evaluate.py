"""
src/evaluate.py
Evaluation + simple plotting for CaFe-EDGE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import List

import matplotlib.pyplot as plt
import torch
import yaml
from torch_geometric.data import Batch as PyGBatch

from .train import CaFeEDGE
from .preprocess import StreamEdgeDataset

# ---------------------------------------------------------------------------
#                       CONFIG & DIRECTORY HANDLING
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

RESEARCH_DIR = ROOT / ".research" / "iteration1"
IMAGES_DIR = RESEARCH_DIR / "images"
for _d in [RESEARCH_DIR, IMAGES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "evaluate_cafe_edge",
]

# ---------------------------------------------------------------------------
#                             PLOTTING
# ---------------------------------------------------------------------------

def _annotate(ax):
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_accuracy(results: dict) -> Path:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Accuracy"], [results["accuracy_mu"]], color="#4c72b0")
    _annotate(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("CaFe-EDGE – Accuracy (μ over seeds)")
    fig.tight_layout()
    path = IMAGES_DIR / "accuracy_cafe_edge.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
#                             EVALUATION
# ---------------------------------------------------------------------------

def evaluate_cafe_edge(ckpt_paths: List[Path]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_test = StreamEdgeDataset("test")
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: PyGBatch.from_data_list(x),
    )

    metrics_per_seed = []
    for ckpt in ckpt_paths:
        model = CaFeEDGE(in_dim=ds_test[0].ndata["x"].shape[1]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        correct = total = 0
        latency_ms, chosen_depths = [], []

        for batch in dl_test:
            batch = batch.to(device)
            feat_for_prodef = torch.rand(1, 4).to(device)
            t0 = time.time()
            with torch.no_grad():
                logits, depth, _ = model.predict_with_prodef(batch, feat_for_prodef)
            t1 = time.time()
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += int((preds == batch.y).sum())
            total += batch.y.numel()
            latency_ms.append((t1 - t0) * 1e3)
            chosen_depths.append(depth)

        metrics_per_seed.append(
            {
                "accuracy": correct / total,
                "avg_latency_ms": mean(latency_ms),
                "latency_std_ms": pstdev(latency_ms) if len(latency_ms) > 1 else 0.0,
                "avg_chosen_depth": mean(chosen_depths),
            }
        )

    accs = [m["accuracy"] for m in metrics_per_seed]
    lats = [m["avg_latency_ms"] for m in metrics_per_seed]
    depths = [m["avg_chosen_depth"] for m in metrics_per_seed]

    results = {
        "accuracy_mu": mean(accs),
        "accuracy_sigma": pstdev(accs) if len(accs) > 1 else 0.0,
        "latency_mu_ms": mean(lats),
        "latency_sigma_ms": pstdev(lats) if len(lats) > 1 else 0.0,
        "depth_mu": mean(depths),
        "depth_sigma": pstdev(depths) if len(depths) > 1 else 0.0,
    }

    json_path = RESEARCH_DIR / "experiment1_cafe_edge.json"
    json_path.write_text(json.dumps(results, indent=2))
    print("\n================= EXPERIMENT 1 – CaFe-EDGE =================\n")
    print(json.dumps(results, indent=2))

    fig_path = _plot_accuracy(results)
    print(f"Saved figure → {fig_path}")
    return json_path, fig_path, results
