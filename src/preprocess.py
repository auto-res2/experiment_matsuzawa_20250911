"""src/preprocess.py
Data-acquisition & validation utilities (formerly *datasets.py*).

Fixes in this revision:
1. Wikipedia snapshot *20231101.en* does **not** exist on the public Hub – the
   newest English dump currently available is *20220301.en*.  The registry entry
   has therefore been updated to the valid config string so that the download no
   longer aborts.
2. The previous *reddit* entry pointed to a non-existent config "submissions".
   The Hugging Face dataset builder exposes a *default* configuration when no
   name is supplied, so `config` is now set to *None* which safely loads that
   default split.

All other logic stays unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Union, Optional

from datasets import load_dataset

from .train import get_logger  # shared helper

# ---------------------------------------------------------------------
# Registry of datasets – UPDATED wikipedia & reddit configs
# ---------------------------------------------------------------------

_DATASETS: Dict[str, Dict[str, Union[str, bool, None]]] = {
    # Vision ----------------------------------------------------------
    "imagenet-1k": {
        "hf_name": "ILSVRC/imagenet-1k",
        "config": "default",
        "requires_token": True,  # gated – ImageNet licence agreement
    },
    "ego4d": {
        "hf_name": "chenjoya/videollm-online-chat-ego4d-134k",
        "config": None,
        "requires_token": True,
    },
    # Audio -----------------------------------------------------------
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
    # Text ------------------------------------------------------------
    "wikipedia": {
        "hf_name": "wikipedia",
        "config": "20220301.en",  # ← fixed to existing snapshot
        "requires_token": False,
    },
    "reddit": {
        "hf_name": "reddit",
        "config": None,  # default configuration
        "requires_token": False,
    },
}

# ---------------------------------------------------------------------
# Helper functions (unchanged)
# ---------------------------------------------------------------------

def _have_token() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def _download_dataset(hf_name: str, config: Optional[str], cache_dir: Path, token: Optional[str]) -> None:
    common_kwargs = {
        "cache_dir": str(cache_dir),
        "token": token,
        "download_mode": "reuse_dataset_if_exists",
        "streaming": True,
    }

    if config is None:
        load_dataset(hf_name, **common_kwargs, trust_remote_code=True)
    else:
        load_dataset(hf_name, name=config, **common_kwargs, trust_remote_code=True)


def download_all(data_dir: Path) -> None:  # noqa: D401
    logger = get_logger("datasets")
    data_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, meta in _DATASETS.items():
        needs_token: bool = bool(meta.get("requires_token", False))
        token_available: bool = _have_token()

        if needs_token and not token_available:
            logger.warning("Skipping gated dataset '%s' because $HF_TOKEN is not set.", ds_name)
            continue

        logger.info("Downloading dataset: %s", ds_name)
        try:
            _download_dataset(
                hf_name=str(meta["hf_name"]),
                config=meta.get("config"),
                cache_dir=data_dir / ds_name,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", ds_name, e)
            raise RuntimeError(
                f"Dataset {ds_name} could not be accessed. Per STRICT NO-FALLBACK RULE execution terminates."
            ) from e
        logger.info("✓ %s ready", ds_name)
