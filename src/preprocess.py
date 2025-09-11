"""src/preprocess.py
====================
Provides *very small* synthetic data streams and hold-out buffers so that
src.main can execute end-to-end in <2 s on the CI CPU runner.  The real
paper uses OpenImages-v7, ESC-50 and WikiText-103 – here we emulate them
with random numbers to keep the repository light-weight and licence-free.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch

__all__ = ["build_streams", "build_holdout_buffers"]


# -----------------------------------------------------------------------------
# 1. Continuous random streams per modality
# -----------------------------------------------------------------------------

_STREAM_LEN = 10  # reduced – src.main uses itertools.cycle anyway


def _random_image() -> Dict:
    return {"pixels": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)}


def _random_audio() -> Dict:
    # 64-bin log-mel spectrogram (fake)
    return {"mels": np.random.randn(1, 64, 401).astype(np.float32)}


def _random_text() -> Dict:
    n_tokens = random.randint(5, 15)
    return {"input_ids": torch.randint(0, 32_000, (n_tokens,)).tolist()}


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def build_streams(cfg) -> Dict[str, List]:
    """Return three *finite* lists; src.main wraps them in cycle()."""

    vision = [_random_image() for _ in range(_STREAM_LEN)]
    audio = [_random_audio() for _ in range(_STREAM_LEN)]
    text = [_random_text() for _ in range(_STREAM_LEN)]
    return {"vision": vision, "audio": audio, "text": text}


def build_holdout_buffers(cfg) -> Dict[str, List]:
    """10 static samples per modality – used for the cheap accuracy proxy."""

    return {
        "vision": [_random_image() for _ in range(10)],
        "audio": [_random_audio() for _ in range(10)],
        "text": [_random_text() for _ in range(10)],
    }
