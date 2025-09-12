"""src/evaluate.py
Evaluation utilities and visualisations.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import seaborn as sns
import torch
import matplotlib

# Headless backend – *must* be set before pyplot import.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

__all__ = [
    "evaluate",
    "plot_training_curves",
    "plot_confusion_matrix",
]


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """Return accuracy together with the per-sample predictions and labels."""

    model.eval()
    correct = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

    acc = correct / len(loader.dataset)
    return acc, all_preds, all_labels


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    fig_dir: Path,
) -> Tuple[str, str]:
    """Generate loss and accuracy curves and return file names (not full paths)."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(losses) + 1))

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", label="Training Loss")
    for x, y in zip(epochs, losses):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    fname_loss = fig_dir / "training_loss.pdf"
    plt.savefig(fname_loss, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accuracies, marker="o", color="green", label="Test Accuracy")
    for x, y in zip(epochs, accuracies):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy over Epochs")
    plt.ylim(0, 1)
    plt.legend()
    fname_acc = fig_dir / "accuracy.pdf"
    plt.savefig(fname_acc, bbox_inches="tight")
    plt.close()

    return fname_loss.name, fname_acc.name


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    fig_dir: Path,
) -> str:
    """Plot a 10×10 confusion matrix and return the file name."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    fname_cm = fig_dir / "confusion_matrix.pdf"
    plt.savefig(fname_cm, bbox_inches="tight")
    plt.close()
    return fname_cm.name
