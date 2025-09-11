# src/evaluate.py
"""Evaluation helpers: metrics & evaluation loop."""
from __future__ import annotations

from typing import List, Dict

import torch
from torch.utils.data import DataLoader

# =========================
#  Metrics
# =========================

def top1(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def min_class_accuracy(outputs: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
    preds = torch.cat([o.argmax(1) for o in outputs])
    t = torch.cat(targets)
    accuracies = []
    for cls in torch.unique(t):
        cls_mask = t == cls
        accuracies.append((preds[cls_mask] == cls).float().mean().item())
    return min(accuracies) if accuracies else 0.0


# =========================
#  Evaluation Loop (stub)
# =========================

def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device | str) -> Dict[str, float]:
    """Runs a forward pass to compute evaluation metrics. Currently placeholder."""
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(clips)
            outputs.append(logits.cpu())
            targets.append(labels.cpu())

    return dict(top1=top1(torch.cat(outputs), torch.cat(targets)),
                min_class=min_class_accuracy(outputs, targets))
