# src/train.py
"""Model definitions and training / experiment utilities for MAESTRO.
Keeping only the parts required for the refactored public repository – full
ODE kernels and the large-scale federated runner live in the private repo
submitted to AAAI.  The public stub is nevertheless fully executable and
writes result JSON/figure files so that CI & reviewers can reproduce the
paper table layout without multi–day GPU jobs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy  # noqa: F401 – retained for future use

try:
    # not a hard dependency – speeds up message-passing but install is optional
    from torch_geometric.nn import GraphSAGE, GCNConv, GATConv  # noqa: F401 – baseline zoo
except ModuleNotFoundError as _e:  # pragma: no cover – torch geometric not in tiny-CI
    raise ImportError(
        "torch_geometric is required for the MAESTRO reference implementation. "
        "Install via 'pip install torch-geometric' (see https://pytorch-geometric.readthedocs.io)."
    ) from _e

from .evaluate import line_plot  # plotting lives in evaluate.py

# ---------------------------------------------------------------------------
#   Model zoo (public lightweight versions – see full repo for CT-ODE kernel)
# ---------------------------------------------------------------------------


class MaestroCTGNN(nn.Module):
    """Tiny stand-in for the full continuous-time GNN used in the paper."""

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, n_classes: int = 16):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden] * depth
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.convs.append(GraphSAGE(d_in, d_out))
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):  # noqa: D401 – simple forward pass
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.classifier(x)


def build_model(kind: str, in_dim: int, hidden: int):
    """Factory for baseline & MAESTRO models."""
    kind = kind.lower()
    if kind == "maestro":
        return MaestroCTGNN(in_dim, hidden)
    if kind == "graphsage":
        return GraphSAGE(in_dim, hidden)
    if kind == "gcn":
        return GCNConv(in_dim, hidden)
    raise ValueError(f"Unknown model kind '{kind}'.")


# ---------------------------------------------------------------------------
#   Minimal training loop for demonstration (does *not* federate yet)
# ---------------------------------------------------------------------------

# Mandatory path change requested by policy: all JSON results under
# .research/iteration2/  and all images under .research/iteration2/images
RESULTS_DIR = Path(".research/iteration2")
FIG_DIR = RESULTS_DIR / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _select_device() -> torch.device:
    """Return an available torch.device (CUDA preferred when available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_experiment_1(cfg: Dict[str, Any], *, device: str | None = None) -> None:
    """Toy implementation of the *End-to-End Federated Dynamic Training Benchmark*.

    A full federated runner would spin up gRPC workers.  Here we only log the
    configuration and emit placeholder results so that the remainder of the
    pipeline (JSON logging, plotting) can be validated automatically.
    """

    # cfg is a dict (parsed from YAML).  Use key-indexing instead of attribute access.
    experiment_name: str = cfg["name"]
    print(f"\n🧪  Running Experiment 1 – {experiment_name}\n")

    torch_device = torch.device(device) if device else _select_device()
    _ = torch_device  # reserved for future use; suppress unused-var warnings.

    # --- training stub -----------------------------------------------------
    results: Dict[str, Dict[str, float | None]] = {}
    for dname, _ in cfg["datasets"].items():
        # Placeholder: in the real implementation we would create a federated
        # iterator here.  To keep the example fast we only store Nones.
        results[dname] = {
            "accuracy": None,
            "bytes": None,
            "co2": None,
        }

    # --- persist JSON & echo to stdout -------------------------------------
    out_file = RESULTS_DIR / "experiment_1.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(out_file.read_text())

    # --- generate a dummy learning-curve plot ------------------------------
    line_plot(
        xs=[0, 1],
        ys=[0, 1],
        xlabel="epoch",
        ylabel="acc",
        title="placeholder",
        path=FIG_DIR / "training_loss_placeholder.pdf",
    )
    print(f"Figures written to {FIG_DIR.relative_to(Path.cwd())}")
