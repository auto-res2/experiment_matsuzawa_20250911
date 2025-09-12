from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

__all__ = [
    "evaluate",
    "save_line_plot",
    "save_confusion_matrix",
]


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion):
    """Evaluate ``model`` over ``loader`` – returns (loss, acc, preds, targets)."""

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_targets),
    )


def save_line_plot(values: List[float], ylabel: str, filename: Path, title: str):
    plt.figure()
    epochs = list(range(1, len(values) + 1))
    sns.lineplot(x=epochs, y=values, marker="o", label=ylabel)
    for x, y in zip(epochs, values):
        plt.text(x, y, f"{y:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
    finally:
        plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, filename: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
    finally:
        plt.close()
