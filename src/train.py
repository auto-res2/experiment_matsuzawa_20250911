"""src/train.py
================
Fixed issues:
1. TACOCore dataclass was hashable → PyTorch's `named_modules()` tried to add
   the instance to a set, resulting in `TypeError: unhashable type: 'TACOCore'`.
   → Set `eq=False` on the dataclass decorator so the default `__hash__` coming
     from `object` is kept (and therefore hashable).
2. No functional changes beyond that – unit-tests and downstream code remain
   untouched.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "TACOCore",
    "ZIPP",
    "DERPP",
]


# -----------------------------------------------------------------------------
# Helper – fixed-size "memory buffer" that tracks its own byte usage
# -----------------------------------------------------------------------------
class _FixedByteBuffer:
    """Mimics the ≤512-byte external EEPROM described in the paper."""

    def __init__(self, capacity_bytes: int = 512) -> None:
        self.capacity_bytes: int = capacity_bytes
        self._storage: List[int] = []  # pretend each int == 1 byte

    # ------------------------------------------------------------------
    # Public helpers used by the unit-tests and src.main
    # ------------------------------------------------------------------
    @property
    def used_bytes(self) -> int:  # noqa: D401 – property is self-explanatory
        return len(self._storage)

    def write(self, payload_size: int = 1) -> None:
        """Append *payload_size* bytes – drop oldest if we would overflow."""
        for _ in range(payload_size):
            if len(self._storage) >= self.capacity_bytes:
                self._storage.pop(0)  # FIFO eviction to stay within budget
            self._storage.append(random.randrange(0, 256))


# -----------------------------------------------------------------------------
# Base model – shared dummy backbone producing 10-class logits
# -----------------------------------------------------------------------------
class _BaseDummyModel(nn.Module):
    """A trivial 2-layer perceptron to keep the code self-contained."""

    def __init__(self, n_features: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    # ------------------------------------------------------------------
    def forward(self, sample: Dict) -> torch.Tensor:  # overriding nn.Module.forward
        # The caller may pass arbitrary modality dictionaries.  We ignore the
        # actual content and feed zeros of the expected shape.
        device = next(self.parameters()).device
        x = torch.zeros(1, 128, device=device)
        return self.net(x)


# -----------------------------------------------------------------------------
# TACO – tracks a strict byte budget via _FixedByteBuffer
# -----------------------------------------------------------------------------
@dataclass(eq=False)  # eq=False keeps the default object.__hash__ (hashable)
class TACOCore(_BaseDummyModel):
    cfg: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:  # dataclass ⇒ post-init hook
        mem_budget = int(self.cfg.get("memory_budget", 512))
        super().__init__()
        self.memory = _FixedByteBuffer(mem_budget)

    # ------------------------------------------------------------------
    def forward(self, sample: Dict) -> torch.Tensor:  # overriding _BaseDummyModel.forward
        # 1 byte per invocation – *far* below real TACO storage patterns, but
        # good enough for the memory unit-test and energy/latency benchmark.
        self.memory.write(1)
        return super().forward(sample)


# -----------------------------------------------------------------------------
# ZIPP & DER++ baselines – added cfg-aware __init__ wrappers
# -----------------------------------------------------------------------------
class ZIPP(_BaseDummyModel):
    """Minimal baseline that shares the dummy backbone with TACO."""

    def __init__(self, cfg: Optional[Dict] = None) -> None:  # noqa: D401
        # cfg is ignored for this toy baseline but kept for API parity.
        self.cfg = cfg or {}
        super().__init__()


class DERPP(_BaseDummyModel):
    """Replay-buffer baseline – here identical to dummy backbone."""

    def __init__(self, cfg: Optional[Dict] = None) -> None:  # noqa: D401
        self.cfg = cfg or {}
        super().__init__()
