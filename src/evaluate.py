"""
src/evaluate.py
Evaluation + simple plotting for CaFe-EDGE.
Updated for iteration-7.  All artefacts must be under .research/iteration7/ …
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

# ---------------------------------------------------------------------------
#           torch-geometric – import with graceful CPU-only fallback
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Batch as PyGBatch
except ModuleNotFoundError:  # pragma: no cover – fallback stub
    from .tg_stub import install_tg_stub

    install_tg_stub()
    from torch_geometric.data import Batch as PyGBatch  # noqa: E402

from .train import CaFeEDGE  # after stub install – shares implementation
from .preprocess import StreamEdgeDataset

# ---------------------------------------------------------------------------
#                       CONFIG & DIRECTORY HANDLING
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

RESEARCH_DIR = ROOT / ".research" / "iteration7"
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

def _instantiate_from_ckpt_meta(meta: dict, in_dim: int) -> CaFeEDGE:
    """Construct a CaFe-EDGE model from the hyper-parameter metadata saved
    within a checkpoint.
    """
    return CaFeEDGE(
        in_dim=in_dim,
        hidden=meta["hidden_dim"],
        layers=meta["gnn_layers"],
        lambda_E=meta["lambda_E"],
        lambda_MI=meta["lambda_MI"],
        epsilon_F=meta["epsilon_F"],
        epsilon_S=meta["epsilon_S"],
    )


def evaluate_cafe_edge(ckpt_paths: List[Path]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_test = StreamEdgeDataset("test")
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: PyGBatch.from_data_list(x),
    )

    metrics_per_seed = []
    for ckpt in ckpt_paths:
        payload = torch.load(ckpt, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
        hparams = payload.get("hparams") if isinstance(payload, dict) else None

        # Fallback to global config if metadata is missing (legacy ckpts)
        if hparams is None:
            mdl_cfg = CONFIG["models"]["cafe_edge"]
            hparams = {
                "hidden_dim": mdl_cfg["hidden_dim"],
                "gnn_layers": mdl_cfg["gnn_layers"],
                "lambda_E": mdl_cfg["lambda_E"],
                "lambda_MI": mdl_cfg["lambda_MI"],
                "epsilon_F": mdl_cfg["epsilon_F"],
                "epsilon_S": mdl_cfg["epsilon_S"],
            }

        model = _instantiate_from_ckpt_meta(hparams, in_dim=ds_test[0].x.shape[1]).to(device)
        model.load_state_dict(state_dict, strict=True)
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
                "accuracy": correct / total if total else 0.0,
                "avg_latency_ms": mean(latency_ms) if latency_ms else 0.0,
                "latency_std_ms": pstdev(latency_ms) if len(latency_ms) > 1 else 0.0,
                "avg_chosen_depth": mean(chosen_depths) if chosen_depths else 0.0,
            }
        )

    accs = [m["accuracy"] for m in metrics_per_seed]
    lats = [m["avg_latency_ms"] for m in metrics_per_seed]
    depths = [m["avg_chosen_depth"] for m in metrics_per_seed]

    results = {
        "accuracy_mu": mean(accs) if accs else 0.0,
        "accuracy_sigma": pstdev(accs) if len(accs) > 1 else 0.0,
        "latency_mu_ms": mean(lats) if lats else 0.0,
        "latency_sigma_ms": pstdev(lats) if len(lats) > 1 else 0.0,
        "depth_mu": mean(depths) if depths else 0.0,
        "depth_sigma": pstdev(depths) if len(depths) > 1 else 0.0,
    }

    json_path = RESEARCH_DIR / "experiment1_cafe_edge.json"
    json_path.write_text(json.dumps(results, indent=2))

    # Print the JSON content to stdout for verification (as required)
    print("\n================= EXPERIMENT 1 – CaFe-EDGE =================\n")
    print(json.dumps(results, indent=2))

    fig_path = _plot_accuracy(results)
    print(f"Saved figure → {fig_path}")
    return json_path, fig_path, results
