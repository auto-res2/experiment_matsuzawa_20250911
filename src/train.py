"""
src/train.py
==============
Model architectures and training-related utilities.  Only minimal stubs are
kept – the refactor preserves exactly the behaviour of the original single
script while making the code importable from the new project layout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "BloomGNN",
    "BloomGNNNoBMRF",
    "OrbitGCN",
    "PairNormGCN",
]


class _Base(nn.Module):
    """Lightweight base-network stub.

    The real curvature gating, Kalman updates, etc. are intentionally left as
    *stubs* – exactly as in the original experiment code.  This file only
    guarantees shape / dtype compatibility so the full experiment can run end
    to end without crashing.
    """

    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    # ------------------------------------------------------------------
    def forward_temporal(self, data):  # noqa: D401, N802 – keep original API
        """Dummy forward that mimics the temporal curvature-aware call.

        Returns
        -------
        loss : torch.Tensor
            Zero scalar tensor so autograd can still build a graph.
        kappa_var : float
            Dummy curvature variance (0.0).
        """
        # Maintain dtype/device consistency with incoming data tensor.
        loss = data.x.sum() * 0  # zero scalar anchors autograd graph
        kappa_var: float = 0.0
        return loss, kappa_var


class BloomGNN(_Base):
    pass


class BloomGNNNoBMRF(_Base):
    pass


class OrbitGCN(_Base):
    pass


class PairNormGCN(_Base):
    pass
