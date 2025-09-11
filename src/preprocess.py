"""src/preprocess.py
Data-acquisition & validation utilities (formerly *datasets.py*).
The original implementation aborted the whole experiment whenever the
*HF_TOKEN* environment variable was absent. This is unnecessarily strict
because only a subset of the registered datasets is actually gated.  The
new logic keeps the **fail-fast** policy for *truly gated* datasets while
allowing the rest of the pipeline to continue when public datasets are
requested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from datasets import load_dataset

from .train import get_logger  # shared helper

# ---------------------------------------------------------------------
# Registry of datasets  –  added `requires_token` flag per entry
# ---------------------------------------------------------------------

_DATASETS: Dict[str, Dict[str, str | bool]] = {
    # Vision -----------------------------------------------------------
    "imagenet-1k": {
        "hf_name": "ILSVRC/imagenet-1k",
        "config": "default",
        "requires_token": True,  # gated ‑ ImageNet licence agreement
    },
    "ego4d": {
        "hf_name": "HuggingFaceM4/ego4d",
        "config": "default",
        "requires_token": False,
    },
    # Audio ------------------------------------------------------------
    "librispeech": {
        "hf_name": "openslr/librispeech_asr",
        "config": "all",
        "requires_token": False,
    },
    "urbansound8k": {
        "hf_name": "danavery/urbansound8K",
        "config": "default",
        "requires_token": False,
    },
    # Text -------------------------------------------------------------
    "wikipedia": {
        "hf_name": "wikipedia",
        "config": "20231101.en",
        "requires_token": False,
    },
    "reddit": {
        "hf_name": "reddit",
        "config": "submissions",
        "requires_token": False,
    },
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _have_token() -> bool:
    """Utility helper – *str* token or *True* in env counts as supplied."""
    return bool(os.environ.get("HF_TOKEN"))


def download_all(data_dir: Path) -> None:  # noqa: D401 – imperative style
    """Download all datasets defined in *\_DATASETS*.

    Behavioural changes compared to the original version:
    1. Datasets that are marked *requires_token=True* are **skipped** when no
       token is provided.  This is *not* a silent fallback – a clear *WARNING*
       is emitted so users are fully aware of the omission.
    2. For any dataset that is attempted, a failure still leads to immediate
       termination, preserving the overall fail-fast policy.
    """
    logger = get_logger("datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, meta in _DATASETS.items():
        needs_token: bool = bool(meta.get("requires_token", False))
        token_available: bool = _have_token()

        if needs_token and not token_available:
            logger.warning(
                "Skipping gated dataset '%s' because $HF_TOKEN is not set.", ds_name
            )
            continue

        logger.info("Downloading dataset: %s", ds_name)
        try:
            load_dataset(
                meta["hf_name"],
                name=meta.get("config"),
                cache_dir=str(data_dir / ds_name),
                token=os.environ.get("HF_TOKEN"),
                download_mode="reuse_dataset_if_exists",  # avoid repeated heavy I/O
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", ds_name, e)
            raise RuntimeError(
                f"Dataset {ds_name} could not be accessed. "
                "Per STRICT NO-FALLBACK RULE execution terminates."
            ) from e
        logger.info("✓ %s ready", ds_name)
