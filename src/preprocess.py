"""Very small stand-in for the real MedMNIST+VQ-tokeniser pipeline.

The genuine RAPTOR code retrains a VQGAN encoder and tokenises the whole
image set.  We cannot ship that inside the six-file playground, so this
stub returns *random* tensors while honouring the same public interface.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


class ImageTokenDataset(Dataset):
    """Return random 64×64 RGB tensors – enough for unit tests."""

    def __init__(self, hf_name: str, split: str, vq_encoder):  # noqa: D401 – match caller signature
        super().__init__()
        random.seed(hash(hf_name + split) & 0xFFFF)
        self.length = 64  # deterministic small dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> torch.Tensor:  # noqa: D401
        g = torch.Generator().manual_seed(idx)
        return torch.randn(3, 64, 64, generator=g)