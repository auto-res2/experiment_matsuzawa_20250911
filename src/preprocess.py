# src/preprocess.py
"""Data download & preprocessing utilities – now uses CIFAR-10 for a self-contained, 
licence-free testbed in accordance with the fail-fast policy (no silent fallbacks).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Tuple

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

__all__ = [
    "build_loader",
]

# ============================================================
#  Secure Downloader (retained for future large-scale datasets)
# ============================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, sha256_hex: str | None = None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_hex and _sha256(target) == sha256_hex:
            return  # OK
        print(f"Checksum mismatch or unknown – re-downloading {target.name}")
        target.unlink()
    print(f"Downloading {url} → {target}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    if sha256_hex and _sha256(target) != sha256_hex:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {target}")


# ============================================================
#  Toy Dataset – CIFAR-10 wrapped as a clip stream (T = 1)
# ============================================================

class _CIFAR10Clips(Dataset):
    """Wraps torchvision.CIFAR10 so that each sample mimics a video clip.

    Output shape: (T=1, C=3, H=224, W=224) to match PhoenixMem expectation.
    """

    def __init__(self, root: Path, train: bool, transform):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    # ------------- Dataset API -------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # Add temporal dimension – shape becomes (1,C,H,W)
        clip = img.unsqueeze(0)
        return clip, label


# ============================================================
#  Loader helper (public)
# ============================================================

def build_loader(root: Path, cfg: dict) -> Tuple[DataLoader, int]:
    """Constructs a DataLoader for the experiments.

    The function intentionally uses CIFAR-10 to guarantee that the pipeline has
    concrete numerical data without requiring restricted datasets.
    """

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ds = _CIFAR10Clips(root=root / "cifar10", train=True, transform=tfm)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    n_classes = 10
    return loader, n_classes
