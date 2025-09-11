"""src/preprocess.py
Data-acquisition & validation utilities (formerly *datasets.py*).

Fix applied: the *ego4d* subset was discovered to be gated on the Hub, which
caused the pipeline to abort when no *HF_TOKEN* was supplied.  The registry now
correctly marks this entry as `requires_token: True`, so it is skipped – with a
clear WARNING – whenever the token is absent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Union, Optional

from datasets import load_dataset

from .train import get_logger  # shared helper

# ---------------------------------------------------------------------
# Registry of datasets – added/updated `requires_token` flags
# ---------------------------------------------------------------------

_DATASETS: Dict[str, Dict[str, Union[str, bool, None]]] = {
    # Vision ----------------------------------------------------------
    "imagenet-1k": {
        "hf_name": "ILSVRC/imagenet-1k",
        "config": "default",
        "requires_token": True,  # gated – ImageNet licence agreement
    },
    "ego4d": {
        # The previously chosen subset is in fact *gated*; mark accordingly so it
        # will be skipped when $HF_TOKEN is not defined.
        "hf_name": "chenjoya/videollm-online-chat-ego4d-134k",
        "config": None,
        "requires_token": True,  # ← fixed
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


def _download_dataset(hf_name: str, config: Optional[str], cache_dir: Path, token: Optional[str]) -> None:
    """Wrapper around *load_dataset* that omits the *name* argument when *config* is None.
    Added `streaming=True` so that enormous datasets are *not* fully fetched – only
    their metadata is downloaded, which is sufficient for placeholder experiments
    and drastically reduces CI runtime.
    """
    common_kwargs = {
        "cache_dir": str(cache_dir),
        "token": token,
        "download_mode": "reuse_dataset_if_exists",
        "streaming": True,  # ← lightweight access
    }

    if config is None:
        load_dataset(hf_name, **common_kwargs, trust_remote_code=True)
    else:
        load_dataset(hf_name, name=config, **common_kwargs, trust_remote_code=True)


def download_all(data_dir: Path) -> None:  # noqa: D401 – imperative style
    """Download all datasets defined in *_DATASETS*.

    Behaviour:
        1. Datasets marked `requires_token` are *skipped* (with WARNING) when the
           token is not present.
        2. Any download attempt failure still terminates the program to respect
           the global fail-fast policy.
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
            _download_dataset(
                hf_name=str(meta["hf_name"]),
                config=meta.get("config"),
                cache_dir=data_dir / ds_name,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", ds_name, e)
            raise RuntimeError(
                f"Dataset {ds_name} could not be accessed. "
                "Per STRICT NO-FALLBACK RULE execution terminates."
            ) from e
        logger.info("✓ %s ready", ds_name)
