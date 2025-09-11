# src/train.py
"""Model definitions and training utilities for PHOENIX-Mem experiments."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================
#  Model
# =========================

class PhoenixMem(nn.Module):
    """Highly simplified PHOENIX-Mem backbone (vision-RNN)."""

    def __init__(self, use_causal: bool = True, ecc_r: int = 14, sram_protect: bool = True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 224×224 → 112×112
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 128),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(64, 100)  # assume 100 classes
        # Metadata for later serialisation
        self.metadata = dict(use_causal=use_causal, ecc_r=ecc_r, sram_protect=sram_protect)

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,C,H,W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.encoder(x)
        feats = feats.view(b, t, -1)
        out, _ = self.gru(feats)
        logits = self.classifier(out[:, -1])
        return logits


# =========================
#  Training Helper
# =========================

def train_model(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict,
    device: torch.device | str,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Basic supervised training loop – returns trained model & logs."""

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    epoch_logs: dict[str, float] = {}

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, len(loader))
        epoch_logs[f"epoch_{epoch}_loss"] = avg_loss
        print(f"Epoch {epoch}: loss={avg_loss:.3f}  time={epoch_time:.1f}s")

    return model, epoch_logs
