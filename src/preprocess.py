"""Data-preparation stubs so that the training script can run end-to-end even
in environments where the real datasets are unavailable.  The functions create
small synthetic folders with dummy tensors to satisfy the expected directory
layout.
"""
from __future__ import annotations

import pathlib
from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config_loader import DatasetCfg

__all__ = ["prepare_dataset", "build_dataloader"]


def _create_dummy_dataset(root: pathlib.Path, n_samples: int = 64) -> None:
    """Populate *root* with dummy tensors so that *build_dataloader* can read
    them later.  This keeps the scaffold self-contained and lightweight.
    """
    if root.exists():
        return  # Already prepared in a previous call.
    for split in ("train", "val"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # We simply save tensors as `.pt` files;  the build_dataloader helper
        # will load them back.
        rng = torch.randn(n_samples, 3, 32, 32)  # Tiny dummy images.
        torch.save(rng, split_dir / "data.pt")


# -----------------------------------------------------------------------------
# Public interface -------------------------------------------------------------
# -----------------------------------------------------------------------------

def prepare_dataset(cfg: DatasetCfg) -> pathlib.Path:  # noqa: D401
    """Ensure a dummy dataset exists and return its root *Path*."""

    root = pathlib.Path(".tmp_data") / cfg.id
    _create_dummy_dataset(root)
    return root


def build_dataloader(root: pathlib.Path, batch_size: int, *, split: Literal["train", "val"]
                    ) -> DataLoader:  # noqa: D401
    """Return a *torch.utils.data.DataLoader* for the synthetic data."""

    data_path = root / f"{split}/data.pt"
    tensors = torch.load(data_path)
    dataset = TensorDataset(tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
