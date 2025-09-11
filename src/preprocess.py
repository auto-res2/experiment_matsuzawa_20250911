# src/preprocess.py
"""Data loading & preprocessing pipeline."""
from __future__ import annotations

import pathlib
import random
from typing import Dict, List

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import Subset

_DATA_ROOT = pathlib.Path("data")
_DATA_ROOT.mkdir(exist_ok=True)


def _ensure_dataset(ds):
    if ds is None:
        raise RuntimeError("Dataset unavailable – aborting experiment as per policy")


def load_tiny_imagenet_splits(cfg: Dict, seed: int | None = None) -> List[Subset]:
    """Create 20 × 10 class sequential splits for Tiny-ImageNet."""

    seed = cfg["global"]["seed_list"][0] if seed is None else seed
    ds_name = cfg["datasets"]["tiny_imagenet"]["hf_repo"]

    try:
        ds = load_dataset(ds_name, cache_dir=_DATA_ROOT)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Dataset download failed – experiment halted") from e

    _ensure_dataset(ds)

    tr = T.Compose(
        [
            T.Resize(cfg["datasets"]["tiny_imagenet"]["img_size"]),
            T.RandomCrop(56),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def _transform(ex):
        ex["image"] = tr(ex["image"])
        return ex

    ds = ds.with_transform(_transform)

    # Build 20 sequential tasks (10 classes each)
    random.seed(seed)
    all_labels = list(range(200))
    random.shuffle(all_labels)
    tasks: list[Subset] = []

    for t in range(cfg["datasets"]["tiny_imagenet"]["num_tasks"]):
        cls = all_labels[t * 10 : (t + 1) * 10]
        idx = [i for i, l in enumerate(ds["train"]["label"]) if l in cls]
        tasks.append(Subset(ds["train"], idx))

    return tasks
