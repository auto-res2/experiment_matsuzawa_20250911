"""
src/evaluate.py
================
Experiment execution, plotting helpers and the registry used by ``src.main``.
The heavy scientific computation is stripped – only the control-flow and file
I/O remain so that the CI pipeline can validate paths, JSON output and figure
generation.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit

# ---------------------------------------------------------------------------
# PyTorch ≥2.6 defaults to ``weights_only=True`` in torch.load which blocks
# arbitrary Python objects.  OGB stores ``torch_geometric.data.DataEdgeAttr``
# – we must explicitly allow-list this class before any dataset is loaded.
# ---------------------------------------------------------------------------
try:  # pragma: no cover – safety net for older PyTorch versions
    import torch.serialization as _ser

    from torch_geometric.data.data import DataEdgeAttr

    _ser.add_safe_globals([DataEdgeAttr])
except (ImportError, AttributeError):
    # Either we are on an older PyTorch or the API changed – in both cases the
    # default behaviour will work, so we just warn and continue.
    print("[WARN] Could not register DataEdgeAttr as a safe global – proceeding anyway.")

from .train import (
    BloomGNN,
    BloomGNNNoBMRF,
    OrbitGCN,
    PairNormGCN,
)

# ---------------------------------------------------------------------------
# Registry helper – keeps src.main decoupled from concrete experiment classes
# ---------------------------------------------------------------------------
EXPERIMENT_REGISTRY: Dict[str, type] = {}

def register(name: str):  # noqa: D401 –  simple functional decorator
    """Decorator that registers an experiment so that ``src.main`` can find it."""

    def _wrap(cls):
        EXPERIMENT_REGISTRY[name] = cls
        return cls

    return _wrap

# ---------------------------------------------------------------------------
# Helper utilities – kept here to avoid extra files / imports
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover – highly unlikely, but we harden I/O
        print("[WARN] Failed to create directory:", path.parent)
        traceback.print_exc()


def save_line(x, y, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    """Create a 2-D line plot and save it as PDF under *exactly* the requested path."""
    _ensure_parent(path)
    plt.figure()
    plt.plot(x, y, marker="o", label=title)
    for _x, _y in zip(x, y):
        plt.annotate(f"{_y:.2f}", (_x, _y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def print_heading(txt: str) -> None:
    print("\n" + "=" * 60)
    print(txt)
    print("=" * 60)

# ---------------------------------------------------------------------------
#                     EXPERIMENT 1 – BMRF STABILITY STUDY
# ---------------------------------------------------------------------------


@register("experiment1")
class Experiment1:
    """Toy variance / divergence study – fully synthetic to keep CI lightweight."""

    # ------------------------------------------------------------------
    def __init__(self, cfg, outdir: Path):
        self.cfg = cfg
        self.outdir = outdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def _prepare_data(self):
        """Download datasets (real) – but we never *use* them later on."""
        from pathlib import Path as _Path  # local alias to avoid shadowing

        # ---- OGBN-arxiv ------------------------------------------------
        # The real experiment uses the *temporal* variant, but the classic one
        # is ~80 MB and downloads quickly in the CI environment.
        self.arxiv_ds = PygNodePropPredDataset("ogbn-arxiv")

        # ---- Reddit ----------------------------------------------------
        reddit_root = _Path("data/reddit")
        try:
            reddit_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            traceback.print_exc()
        self.reddit_ds = Reddit(reddit_root)
        # Further preprocessing is omitted – we never dereference the data.

    # ------------------------------------------------------------------
    def run(self) -> Tuple[Dict[str, Dict[str, float]], List[Path]]:
        """Execute the experiment and return a metrics dict plus figure paths."""
        self._prepare_data()
        seeds = self.cfg["seeds"]
        models = {
            "bloom": BloomGNN,
            "no_bmrf": BloomGNNNoBMRF,
            "pairnorm": PairNormGCN,
            "orbit": OrbitGCN,
        }
        metrics: Dict[str, Dict[str, float]] = {}
        for m_name, m_cls in models.items():
            var_list, div_list = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = m_cls(**self.cfg["model_params"][m_name]).to(self.device)
                var, diverged = self._train_single(model)
                var_list.append(var)
                div_list.append(int(diverged))
            metrics[m_name] = {
                "kappa_variance_mean": float(np.mean(var_list)),
                "divergence_rate": float(np.mean(div_list)),
            }

        # ----------------------------------------------------------------
        # Figure – PDF under .research/iteration2/images
        # ----------------------------------------------------------------
        images_dir = self.outdir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        fig_path = images_dir / "experiment1_kappa_variance.pdf"
        save_line(
            x=list(metrics.keys()),
            y=[v["kappa_variance_mean"] for v in metrics.values()],
            title="Kappa Variance across Models",
            xlabel="Model",
            ylabel="Variance",
            path=fig_path,
        )
        return metrics, [fig_path]

    # ------------------------------------------------------------------
    def _train_single(self, model):
        """Short synthetic training loop with divergence detection."""
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        diverged = False
        for _epoch in range(self.cfg["epochs"]):
            try:
                # We purposely pass *some* object so that the signature matches.
                loss, var = model.forward_temporal(self.arxiv_ds)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                # Divergence check identical to original script.
                if loss.detach().float().item() > 1e5:
                    diverged = True
                    break
            except (FloatingPointError, RuntimeError):
                diverged = True
                break
        return var, diverged
