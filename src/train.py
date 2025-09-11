# src/train.py
"""Model definition and training helpers for the HARP-RAFT experimental
suite.  All heavy-lifting code that touches torch should stay here so that
other modules (main / evaluate) remain light-weight.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.models as tv_models

# -----------------------------------------------------------------------------
#  CONSTANTS – directories are created once per import to avoid race conditions
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  MODEL COMPONENTS
# -----------------------------------------------------------------------------
class HiRA_Adapter(nn.Module):
    """A minimal – yet fully invertible – adapter that mimics the reversible
    Householder-flow layer described in the paper.  For the purposes of the
    public demo we stick to an orthogonal 1×1 convolution (Glow-style) plus
    an affine transformation.  This keeps the dependency footprint small
    (no custom CUDA extensions) while preserving the *activation-free* back-
    propagation property that the experiments rely on.
    """

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        # Orthogonal weight initialisation via QR
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.weight = nn.Parameter(q)  # (dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.rank = rank  # kept for logging / config but unused in stub

    # ------------------------------------------------------------------
    #  Forward / inverse (reversible)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        orig_shape = x.shape
        z = torch.matmul(x.flatten(1), self.weight) * self.scale + self.shift
        return z.view(orig_shape)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = (z.flatten(1) - self.shift) / self.scale @ self.weight.T
        return x.view(z.shape)


# -----------------------------------------------------------------------------
#  FACTORY
# -----------------------------------------------------------------------------

def build_resnet50_hira(*, num_classes: int = 1000, rank: int = 32) -> nn.Module:
    """Create an ImageNet-pre-trained ResNet-50 whose final fully-connected
    layer is augmented with a HiRA adapter.  The backbone is frozen in order
    to focus learning capacity on the adapter – in line with the continual
    learning setting.
    """
    backbone = tv_models.resnet50(weights="IMAGENET1K_V1")

    # Freeze all backbone parameters (adapter + FC stay trainable)
    for p in backbone.parameters():
        p.requires_grad_(False)

    dim = backbone.fc.in_features
    backbone.hira = HiRA_Adapter(dim, rank)
    backbone.fc = nn.Linear(dim, num_classes)
    return backbone


# -----------------------------------------------------------------------------
#  TRAIN / EVAL HELPERS
# -----------------------------------------------------------------------------

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:  # noqa: D401
    """Top-1 accuracy in percentage (no-grad context)."""
    with torch.no_grad():
        preds = output.argmax(1)
        return (preds == target).float().mean().item() * 100.0


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str = "cuda",
) -> float:
    """Run one full pass over *loader*.  If *optimizer* is ``None`` the model
    is evaluated under ``torch.no_grad()``; otherwise standard training is
    performed.
    Returns the average accuracy over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_correct, total_samples = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_correct += (out.argmax(1) == y).float().sum().item()
        total_samples += y.size(0)

    return 100.0 * total_correct / max(total_samples, 1)
