"""
src/evaluate.py
================
Evaluation logic, figure helpers and the experiment registry live here.  All
logic is copied verbatim from the original single-file script except for path
adjustments and minor robustness fixes (try / except blocks, safer filesystem
handling, etc.).
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
    """Decorator that makes an experiment discoverable via EXPERIMENT_REGISTRY."""

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
    """Save a simple line plot as PDF (always)."""
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
    """Variance / divergence study for the Bayesian curvature filter (BMRF)."""

    # ------------------------------------------------------------------
    def __init__(self, cfg, outdir: Path):
        self.cfg = cfg
        self.outdir = outdir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def _prepare_data(self):
        """Load the required datasets.

        The temporal binning logic is intentionally left as *omitted for
        brevity* exactly as in the original experiment code.
        """
        from pathlib import Path as _Path  # local alias to avoid shadowing

        # ---- OGBN-arxiv -------------------------------------------------
        self.arxiv_ds = PygNodePropPredDataset("ogbn-arxiv")  # temporal slicing later

        # ---- Reddit -----------------------------------------------------
        reddit_root = _Path("data/reddit")
        try:
            reddit_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory creation failure is non-fatal but will surface later
            traceback.print_exc()
        self.reddit_ds = Reddit(reddit_root)
        # Temporal binning / preprocessing is domain-specific and hence
        # remains a stub, mirroring the original code.

    # ------------------------------------------------------------------
    def run(self) -> Tuple[Dict[str, Dict[str, float]], List[Path]]:
        """Execute the experiment and return metrics + figure paths."""
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
        # Figure – always PDF, always under .research/iteration1/images
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
        """Abbreviated training loop with built-in divergence detection."""
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        diverged = False
        # NOTE: best_val / patience are kept for parity but unused here.
        for _epoch in range(self.cfg["epochs"]):
            try:
                loss, var = model.forward_temporal(self.arxiv_ds)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                # Extremely large loss considered diverged (identical criterion)
                if loss.detach().float().item() > 1e5:
                    diverged = True
                    break
            except (FloatingPointError, RuntimeError):
                diverged = True
                break
        return var, diverged
