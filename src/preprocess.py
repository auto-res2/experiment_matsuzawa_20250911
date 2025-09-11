"""src/preprocess.py
Data-acquisition & validation utilities (formerly *datasets.py*).
The HF-token check and STRICT NO-FALLBACK policy are kept verbatim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from datasets import load_dataset

from .train import get_logger  # shared helper

# ---------------------------------------------------------------------
# Registry of datasets
# ---------------------------------------------------------------------

_DATASETS: Dict[str, Dict[str, str]] = {
    # Vision -------------------------------------------------------------
    "imagenet-1k": {
        "hf_name": "ILSVRC/imagenet-1k",
        "config": "default",
    },
    "ego4d": {
        "hf_name": "HuggingFaceM4/ego4d",
        "config": "default",
    },
    # Audio --------------------------------------------------------------
    "librispeech": {
        "hf_name": "openslr/librispeech_asr",
        "config": "all",
    },
    "urbansound8k": {
        "hf_name": "danavery/urbansound8K",
        "config": "default",
    },
    # Text ---------------------------------------------------------------
    "wikipedia": {
        "hf_name": "wikipedia",
        "config": "20231101.en",
    },
    "reddit": {
        "hf_name": "reddit",
        "config": "submissions",
    },
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _check_hf_token() -> None:
    """Validate that the user exported a Hugging-Face token for gated sets."""
    if os.environ.get("HF_TOKEN") is None:
        raise RuntimeError(
            "Environment variable HF_TOKEN is missing. "
            "Please export a valid Hugging-Face access token to download gated datasets."
        )


def download_all(data_dir: Path) -> None:
    """Download all datasets defined in *_DATASETS*.
    The function terminates with *RuntimeError* if any dataset is unavailable –
    enforcing the STRICT NO-FALLBACK RULE.
    """
    logger = get_logger("datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    # ImageNet is gated → ensure token is present before starting expensive I/O
    _check_hf_token()

    for ds_name, meta in _DATASETS.items():
        logger.info("Downloading dataset: %s", ds_name)
        try:
            load_dataset(
                meta["hf_name"],
                name=meta.get("config"),
                cache_dir=str(data_dir / ds_name),
                token=os.environ["HF_TOKEN"],
                download_mode="force_redownload",  # full acquisition per policy
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", ds_name, e)
            raise RuntimeError(
                f"Dataset {ds_name} could not be accessed. "
                "Per STRICT NO-FALLBACK RULE execution terminates."
            ) from e
        logger.info("✓ %s ready", ds_name)
