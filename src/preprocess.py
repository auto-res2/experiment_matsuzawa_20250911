"""src/preprocess.py
Dataset download, loading & DataLoader preparation utilities.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
import torch

# -----------------------------------------------------------------------------
#                               Dataset helpers
# -----------------------------------------------------------------------------

def download_dataset(url: str, dest: Path) -> None:
    """Download *url* to *dest* (if not already present)."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return  # already downloaded

    print(f"Downloading dataset from {url} …")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                shutil.copyfileobj(r.raw, fh)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download dataset → {exc}") from exc
    print("Dataset successfully downloaded.")


def load_iris_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return *(features, labels)* numpy arrays from the Iris CSV file."""

    features: List[List[float]] = []
    labels: List[int] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue  # skip malformed lines
            features.append([float(x) for x in parts[:4]])
            label_str = parts[4]
            if label_str == "Iris-setosa":
                labels.append(0)
            elif label_str == "Iris-versicolor":
                labels.append(1)
            elif label_str == "Iris-virginica":
                labels.append(2)
            else:
                raise ValueError(f"Unknown label '{label_str}' in dataset.")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


class IrisDataset(torch.utils.data.Dataset):
    """Thin torch.utils.data.Dataset wrapper for (x, y) numpy arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# -----------------------------------------------------------------------------
#                      Train/val/test split & DataLoaders
# -----------------------------------------------------------------------------

def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(x))
    rng.shuffle(idxs)

    test_size = int(len(x) * test_split)
    val_size = int(len(x) * val_split)

    test_idx = idxs[:test_size]
    val_idx = idxs[test_size : test_size + val_size]
    train_idx = idxs[test_size + val_size :]

    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


def create_dataloaders(cfg):
    """Prepare DataLoaders as configured – returns *(train, val, test)* loaders."""

    # ---------- data acquisition ----------
    url = cfg["dataset"]["url"]
    local_path = Path(cfg["dataset"]["local_path"])
    download_dataset(url, local_path)
    x, y = load_iris_dataset(local_path)

    # ---------- splitting ----------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _split_dataset(
        x,
        y,
        cfg["dataset"]["val_split"],
        cfg["dataset"]["test_split"],
        cfg["training"]["seed"],
    )

    train_ds = IrisDataset(x_train, y_train)
    val_ds = IrisDataset(x_val, y_val)
    test_ds = IrisDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds))

    return train_loader, val_loader, test_loader
