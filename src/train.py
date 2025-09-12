"""src/train.py
Model definition and training utilities.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------------------------------------------------
#                               Model definition
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple multi-layer perceptron used for the Iris experiment."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D401
        return self.model(x)


# -----------------------------------------------------------------------------
#                           Training / validation loop
# -----------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    epochs: int,
    seed: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Train *model* returning the loss / accuracy curves."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    for epoch in range(1, epochs + 1):
        # ------------------- training phase -------------------
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ------------------ validation phase ------------------
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                vloss = criterion(preds, yb).item()
                vacc = (preds.argmax(1) == yb).float().mean().item()
        val_losses.append(vloss)
        val_accs.append(vacc)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={epoch_loss:.4f}  val_loss={vloss:.4f}  val_acc={vacc:.4f}"
            )

    return train_losses, val_losses, val_accs
