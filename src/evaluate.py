"""
src/evaluate.py
Now supports both classification and causal-LM evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")

# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def _collect_preds_cls(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.tolist())
    return preds, targets


def _collect_loss_lm(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    with torch.no_grad():
        for (inputs, _) in loader:
            inputs = inputs.to(device)
            out = model(input_ids=inputs, labels=inputs)
            tokens = inputs.numel()
            total_loss += out.loss.item() * tokens
            n_tokens += tokens
    return total_loss / n_tokens  # mean loss per token


# ---------------------------------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------------------------------

def evaluate_model(checkpoint_path, test_loader: DataLoader, device: torch.device, output_dir: Path, experiment_name: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    from src.train import get_model  # local import to avoid circular

    model = get_model(cfg, cfg["dataset"]["input_dim"], cfg["dataset"]["num_classes"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)

    model_type = cfg["model"]["type"].lower()
    metrics: Dict

    if model_type.startswith("dialogpt"):
        avg_loss = _collect_loss_lm(model, test_loader, device)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        metrics = {"avg_token_loss": avg_loss, "perplexity": ppl}
    else:
        preds, targets = _collect_preds_cls(model, test_loader, device)
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, average="weighted", zero_division=0),
            "recall": recall_score(targets, preds, average="weighted", zero_division=0),
            "f1": f1_score(targets, preds, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        }
        _plot_confusion_matrix(torch.tensor(metrics["confusion_matrix"]), output_dir / "figures" / "confusion_matrix.pdf")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    res_path = output_dir / f"{experiment_name}_results.json"
    with open(res_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return {"metrics": metrics, "results_path": str(res_path)}


# ----------------------------------------------------------------------------
# plotting (classification only)
# ----------------------------------------------------------------------------

def _plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
