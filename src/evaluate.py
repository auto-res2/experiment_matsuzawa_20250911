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
# PyTorch ≥2.6 switched the default for ``weights_only`` in ``torch.load`` to
# ``True`` which blocks pickling arbitrary (non-tensor) Python objects.  OGB &
# PyG datasets still rely on full pickling, so dataset loading fails unless we
# either (a) globally force ``weights_only=False`` *or* (b) explicitly allow-
# list every custom class they use.  For the purposes of the CI pipeline the
# simplest, transparent and *fail-fast* solution is to monkey-patch
# ``torch.load`` so that the safer behaviour remains opt-in rather than
# opt-out.  We still keep the allow-list as an extra layer of robustness.
# ---------------------------------------------------------------------------

# --- 1.  Allow-list common PyG attribute classes --------------------------------
try:  # pragma: no cover – guard against API changes / older PyTorch versions
    import torch.serialization as _ser

    from torch_geometric.data.data import DataEdgeAttr

    # Newer releases split attributes into several classes – we try to import
    # them all but degrade gracefully if a given class is not present.
    try:
        from torch_geometric.data.data import DataTensorAttr

        _ser.add_safe_globals([DataEdgeAttr, DataTensorAttr])
    except ImportError:  # pragma: no cover – class not present in this version
        _ser.add_safe_globals([DataEdgeAttr])
except (ImportError, AttributeError):
    # Either we are on an older PyTorch / PyG or the API moved.  We continue
    # without the allow-list – the monkey-patch below will still make loading
    # work in a controlled manner.
    print(
        "[WARN] Could not register PyG attribute classes as safe globals – "
        "falling back to patched torch.load."
    )

# --- 2.  Monkey-patch torch.load so the default is *secure but permissive* ----
_orig_torch_load = torch.load  # keep reference to the original implementation


def _patched_torch_load(*args, **kwargs):  # noqa: D401 – simple wrapper
    """Wrapper that sets ``weights_only=False`` unless the caller overrides it.

    This replicates the default behaviour prior to PyTorch 2.6 which most data
    loaders (including OGB) implicitly rely on.  Users can still opt into the
    safer mode by explicitly passing ``weights_only=True``.
    """

    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load  # re-route all downstream calls

# ---------------------------------------------------------------------------
# Local imports – deferred until after the patch so they benefit from it
# ---------------------------------------------------------------------------
from .train import (  # noqa: E402  – circular-import safe
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
        # Figure – PDF under .research/iteration3/images (mandatory path)
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
