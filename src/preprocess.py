"""src/preprocess.py
Dataset downloading, splitting and DataLoader setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = ["get_dataloaders"]


def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and test loaders according to the configuration."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    root = cfg["dataset"]["root"]
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Robust download – guard against transient network issues
    try:
        train_ds = datasets.FashionMNIST(root=str(root_path), train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root=str(root_path), train=False, download=True, transform=transform)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Dataset download failed – check your internet connection.") from exc

    if cfg["dataset"].get("train_subset"):
        train_ds = Subset(train_ds, list(range(cfg["dataset"]["train_subset"])))
    if cfg["dataset"].get("test_subset"):
        test_ds = Subset(test_ds, list(range(cfg["dataset"]["test_subset"])))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
