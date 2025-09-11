"""src/preprocess.py
Data-handling, utility helpers and global constants.  Centralising these
avoids circular-import issues across the train / evaluate / main modules.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

# ---------------- global constants & paths --------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
RESEARCH_DIR = ROOT / ".research" / "iteration1"
IMAGE_DIR = RESEARCH_DIR / "images"
RESULT_DIR = RESEARCH_DIR  # JSON files reside directly in the iteration folder

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
SEEDS = [13, 29, 47, 71, 97]

# ---------------------- generic helpers -----------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set every random seed for full reproducibility (to the extent possible)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)


def print_exp_header(name: str, description: str) -> None:
    sep = "─" * 80
    print(f"\n{sep}\n{name}\n{sep}\n{description}\n{sep}")

# Immediately ensure the required directory structure exists
for p in (DATA_DIR, IMAGE_DIR, RESULT_DIR):
    ensure_dir(p)

# ------------------------ dataset acquisition -----------------------------

# Lazy heavy imports – performed only when actually needed.
try:
    import dgl
    from ogb.nodeproppred import DglNodePropPredDataset
except ImportError:
    dgl = None
    DglNodePropPredDataset = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def _download_ogb_papers100m() -> Tuple["dgl.DGLGraph", torch.Tensor, Dict[str, torch.Tensor]]:
    if DglNodePropPredDataset is None:
        raise RuntimeError("ogb package missing; install with  pip install ogb .")
    dataset = DglNodePropPredDataset("ogbn-papers100M", root=DATA_DIR)
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]
    # Pre-processing identical to the original monolithic script
    largest_cc_nids = dgl.algos.largest_connected_components(g)
    g = dgl.node_subgraph(g, largest_cc_nids)
    labels = labels[largest_cc_nids]
    g = dgl.to_simple(dgl.to_bidirected(g))
    feats = g.ndata["feat"].float()
    if feats.shape[1] > 1024:
        from sklearn.decomposition import PCA  # local heavy import

        pca = PCA(n_components=1024, svd_solver="randomized")
        feats = torch.from_numpy(pca.fit_transform(feats.numpy()))
    else:
        feats = (feats - feats.min(0).values) / (feats.max(0).values - feats.min(0).values + 1e-9)
    g.ndata["feat"] = feats
    return g, labels.squeeze(), split_idx


def _download_hf_dataset(repo_id: str, filenames: List[str]) -> List[Path]:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub missing; install with  pip install huggingface_hub ."
        )
    local_paths = []
    for fn in filenames:
        p = hf_hub_download(repo_id=repo_id, filename=fn, cache_dir=DATA_DIR)
        local_paths.append(Path(p))
    return local_paths


def prepare_datasets(config: Dict) -> Dict[str, Dict]:
    """Download / prepare every dataset needed for the configured experiments."""
    datasets: Dict[str, Dict] = {}

    # -------------------- OGBN-Papers100M ------------------------
    if config["datasets"].get("use_ogbn_papers100m", False):
        print("Downloading ogbn-papers100M … this can take a while.")
        g, labels, split = _download_ogb_papers100m()
        datasets["ogbn-papers100M"] = dict(graph=g, labels=labels, split=split)

    # ---------------- restricted licence datasets ---------------
    restricted = {
        "macaron_social": "MACARON-Social",
        "eu_reg_credit": "EU-Reg-Credit",
        "ego_nets": "ego-nets",
    }
    for key, pretty in restricted.items():
        if config["datasets"].get(f"use_{key}", False):
            expected = DATA_DIR / pretty
            if not expected.exists():
                raise RuntimeError(
                    f"Required dataset ‘{pretty}’ not found at {expected}. "
                    "Execution stops per STRICT NO-FALLBACK RULE."
                )
            datasets[pretty] = dict(path=expected)

    # ---------------- Example HF dataset (optional) --------------
    if config["datasets"].get("use_fedgraph", False):
        repo = "FedGraph/fedgraph_ogbn-papers100M_195trainer_0hop_iid_beta_100.0_trainer_id_96"
        print(f"Downloading dataset from HuggingFace repo: {repo}")
        files = _download_hf_dataset(
            repo, ["adj.pt", "features.pt", "train_labels.pt", "test_labels.pt"]
        )
        datasets["fedgraph"] = dict(files=files)

    return datasets
