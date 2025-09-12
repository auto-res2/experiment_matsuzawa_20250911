import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader

__all__ = [
    "SimpleCNN",
    "train_epoch",
    "Metrics",
]


class SimpleCNN(nn.Module):
    """A very small CNN for MNIST-sized images (1×28×28)."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # noqa: D401 – standard forward signature
        return self.net(x)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Runs one training epoch and returns (loss, accuracy)."""

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@dataclass
class Metrics:
    train_loss: list
    train_acc: list
    val_loss: list
    val_acc: list
    test_loss: float
    test_acc: float
