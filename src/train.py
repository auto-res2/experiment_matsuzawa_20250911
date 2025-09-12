"""src/train.py
Model definitions and training utilities.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

__all__ = [
    "SimpleCNN",
    "train_one_epoch",
]


class SimpleCNN(nn.Module):
    """A very small convolutional network for Fashion-MNIST (1×28×28 → 10)."""

    def __init__(self, num_classes: int = 10) -> None:  # noqa: D401
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Standard supervised training loop for a single epoch."""

    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)
