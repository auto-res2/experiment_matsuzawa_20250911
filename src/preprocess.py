# src/preprocess.py
"""Data download & preprocessing utilities."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path
from typing import Tuple

import requests
import torch
from torch.utils.data import Dataset, DataLoader

# =========================
#  Secure Downloader
# =========================

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


# =========================
#  Dataset (stub) – Ego4D-m
# =========================
EGO4DM_URL = "https://ego4d-data.org/static/download/ego4d_mini_release.tar"
EGO4DM_SHA = None  # Unknown


class Ego4DM(Dataset):
    """Minimal stub. Will raise if data inaccessible as per No-Fallback rule."""

    def __init__(self, split: str, root: Path):
        super().__init__()
        self.split = split
        self.root = root
        self._ensure_data()

        # NOTE: Real extraction & index building omitted – dataset requires license.
        raise RuntimeError("Ego4D-m loader not implemented. Provide dataset locally to proceed.")

    # --------------------------------------------------
    def _ensure_data(self):
        tar_path = self.root / "ego4d_mini_release.tar"
        if not tar_path.exists():
            try:
                download(EGO4DM_URL, tar_path, EGO4DM_SHA)
            except Exception as e:
                raise RuntimeError(
                    "Ego4D-m dataset could not be downloaded automatically. "
                    "Place the files under data/ manually.") from e

    # -------- Dataset API (never reached since ^ raises) --------
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Dataset not available")


# =========================
#  Loader helper (toy)
# =========================

def build_loader(root: Path, cfg: dict) -> Tuple[DataLoader, int]:
    """Returns DataLoader and class-count stub (100)."""
    dataset = Ego4DM("train", root)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    return loader, 100
