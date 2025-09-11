# src/preprocess.py
"""Dataset download and tokenisation utilities."""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset

from .train import DatasetCfg

_DATA_ROOT = pathlib.Path("data")
_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def prepare_dataset(cfg: DatasetCfg) -> pathlib.Path:
    """Download *cfg.hf_id* via 🤗 Datasets (if not already cached) and serialize
    the splits as torch tensors.  Returns the root directory that contains the
    cached dataset."""
    ds_root = _DATA_ROOT / cfg.hf_id.replace("/", "__")
    if ds_root.exists():
        return ds_root  # dataset already prepared

    print(f"Downloading dataset {cfg.hf_id} …", flush=True)
    try:
        ds_kwargs = {}
        if cfg.config:
            ds_kwargs["name"] = cfg.config
        dataset = load_dataset(cfg.hf_id, **ds_kwargs)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to download {cfg.hf_id}. The exact exception was:\n{exc}\n"
        ) from exc

    ds_root.mkdir(parents=True, exist_ok=True)
    # Serialise splits for fast future loading --------------------------------
    try:
        torch.save(dataset[cfg.split_train], ds_root / "train.pt")
        torch.save(dataset[cfg.split_test], ds_root / "test.pt")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not cache dataset to disk: {exc}") from exc

    return ds_root
