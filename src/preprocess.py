"""
preprocess.py – data-handling helpers (downloading, wrapping, transforms)
"""
from __future__ import annotations

import pathlib
from typing import Any

import torch
from torchvision import transforms

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "The 'datasets' package is required.  Install via  `pip install datasets`"
    ) from e

__all__ = ["strict_download_dataset", "VisionWrapper"]


# ---------------------------------------------------------------------------
# ⚙️  Download helper
# ---------------------------------------------------------------------------

def strict_download_dataset(hf_id: str, **kwargs) -> Any:
    """Download a HuggingFace dataset and *fail hard* on any error."""
    try:
        ds = load_dataset(hf_id, **kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"❌  Dataset '{hf_id}' could not be downloaded – aborting run. {exc}"
        ) from exc
    return ds


# ---------------------------------------------------------------------------
# 🖼️  Vision wrapper for HF image datasets
# ---------------------------------------------------------------------------

def _default_transform() -> transforms.Compose:
    """Returns the exact sequence of transforms used in the original script."""
    return transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Use per-channel mean/std tuples (3 values) to satisfy torchvision
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class VisionWrapper(torch.utils.data.Dataset):
    """Lightweight `torch.utils.data.Dataset` around HuggingFace vision splits."""

    _IMG_KEYS = ("image", "IMG", "img")  # observed variants across datasets

    def __init__(self, hf_ds, *, transform: transforms.Compose | None = None):
        self.ds = hf_ds
        self.transform = transform if transform is not None else _default_transform()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = None
        for k in self._IMG_KEYS:
            if k in item:
                img = item[k]
                break
        if img is None:
            raise KeyError("Dataset sample lacks an image key – cannot proceed.")
        return self.transform(img)
