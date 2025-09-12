"""src/evaluate.py
Model evaluation, metric computation & visualisation utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

# Use a non-interactive backend – required in head-less evaluation set-ups
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=C0413
import numpy as np  # noqa: E402  pylint: disable=C0413
import torch  # noqa: E402  pylint: disable=C0413
from sklearn.metrics import confusion_matrix  # noqa: E402  pylint: disable=C0413

# -----------------------------------------------------------------------------
#                        Metrics & visualisation helpers
# -----------------------------------------------------------------------------

def evaluate_on_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float, np.ndarray]:
    """Return (*accuracy*, *confusion-matrix*) on the held-out test set."""

    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            acc = (preds.argmax(1) == yb).float().mean().item()
            cm = confusion_matrix(yb.numpy(), preds.argmax(1).numpy(), labels=[0, 1, 2])
    return acc, cm


def plot_curves(
    train_losses: List[float],
    val_accs: List[float],
    cm: np.ndarray,
    exp_name: str,
    image_dir: Path,
) -> List[str]:
    """Generate & persist all figure files – return their relative names."""

    image_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    # ---------------- training-loss curve ----------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", markersize=2, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training Loss Curve")
    plt.legend()
    loss_file = f"training_loss_{exp_name}.pdf"
    plt.savefig(image_dir / loss_file, bbox_inches="tight")
    plt.close()

    # ---------------- validation-accuracy curve ----------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_accs, marker="o", markersize=2, color="green", label="Validation acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    acc_file = f"val_accuracy_{exp_name}.pdf"
    plt.savefig(image_dir / acc_file, bbox_inches="tight")
    plt.close()

    # ---------------- confusion-matrix heat-map ----------------
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    cm_file = f"confusion_matrix_{exp_name}.pdf"
    plt.savefig(image_dir / cm_file, bbox_inches="tight")
    plt.close()

    return [loss_file, acc_file, cm_file]


# -----------------------------------------------------------------------------
#                      Persist numeric results as JSON
# -----------------------------------------------------------------------------

def persist_results(
    results: dict,
    research_dir: Path,
) -> Path:
    """Save *results* dictionary into *research_dir* returning the file path."""

    research_dir.mkdir(parents=True, exist_ok=True)
    path = research_dir / f"results_{results['experiment_name']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path
