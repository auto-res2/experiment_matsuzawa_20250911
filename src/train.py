from __future__ import annotations

# src/train.py
# -------------------------------------------------------------
# Model building and training utilities extracted from the single
# HydraSketch-Φ script.  Only minimal refactoring was applied.
# -------------------------------------------------------------
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoImageProcessor, AutoModelForImageClassification

from .preprocess import (
    ExperimentConfig,
    EdgeBenchPlaceholder,
    abort,
)

__all__ = [
    "build_vision_backbone",
    "run_experiment_1",
]

# ------------------------------------------------------------
# MODEL CONSTRUCTION
# ------------------------------------------------------------

def build_vision_backbone(cfg: ExperimentConfig) -> nn.Module:
    """Load the ResNet-18 backbone (classifier stripped)."""
    processor = AutoImageProcessor.from_pretrained(cfg.vision_backbone)
    model = AutoModelForImageClassification.from_pretrained(cfg.vision_backbone)
    model.classifier = nn.Identity()

    # Wrap the processor so that it can live inside an nn.Sequential
    class _ProcessorWrapper(nn.Module):
        def __init__(self, proc):
            super().__init__()
            self._proc = proc

        def forward(self, x):  # noqa: D401 – imperative style okay
            return self._proc(images=x, return_tensors="pt").pixel_values.squeeze(0)

    wrapped_processor = _ProcessorWrapper(processor)
    return nn.Sequential(wrapped_processor, model)


# ------------------------------------------------------------
# EXPERIMENT-1  (skeleton – will abort without real dataset)
# ------------------------------------------------------------

def run_experiment_1(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Full-stack continual-learning benchmark on EdgeBench-48.

    NOTE: This function will abort once the placeholder dataset is
    accessed, honouring the strict NO-FALLBACK rule from the original
    specification.
    """

    # Dataset (placeholder)
    train_dataset = EdgeBenchPlaceholder()

    # Dataloader (never reached unless real parser exists)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    # Model + optimiser ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vision_backbone(cfg).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training loop (placeholder – will never execute) -----------------
    best_val_acc = 0.0
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        for _batch in train_loader:  # pragma: no cover – never reached
            abort("Training attempted without real dataset – aborting as per NO-FALLBACK rule.")

    # Unreached – provided for completeness.
    return {"best_val_acc": best_val_acc, "history": history}
