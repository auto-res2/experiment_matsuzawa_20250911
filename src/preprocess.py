"""Data-set and misc helper utilities."""
from pathlib import Path
from typing import Dict, Tuple

import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = [
    "set_seed",
    "load_yaml",
    "get_data_loaders",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _split_indices(total_size: int, val_size: int):
    indices = list(range(total_size))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def get_data_loaders(cfg: Dict):
    """Returns train/val/test data loaders according to *cfg*."""

    transform = transforms.Compose([transforms.ToTensor()])
    root = cfg["dataset"]["root"]
    download_flag = cfg["dataset"].get("download", True)

    try:
        train_full = datasets.MNIST(
            root=root,
            train=True,
            download=download_flag,
            transform=transform,
        )
        test_set = datasets.MNIST(
            root=root,
            train=False,
            download=download_flag,
            transform=transform,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Failed to download or load MNIST – check network connectivity.") from e

    # ---------------- split -----------------
    val_subset_size = cfg["dataset"].get("val_subset")
    if val_subset_size is None:
        val_subset_size = int(0.1 * len(train_full))

    train_idx, val_idx = _split_indices(len(train_full), val_subset_size)

    if cfg["dataset"].get("train_subset"):
        train_idx = train_idx[: cfg["dataset"]["train_subset"]]
    if cfg["dataset"].get("val_subset"):
        val_idx = val_idx[: cfg["dataset"]["val_subset"]]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full, val_idx)

    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
