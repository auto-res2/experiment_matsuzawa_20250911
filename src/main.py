"""Main orchestration script for the refactored HYPER-WAVEGUARD demo."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List

import torch
import yaml

from src.preprocess import ensure_ogb, tensor_loader_from_feats_labels
from src.train import HyperWaveGuard, set_seed, train_one_epoch
from src.evaluate import evaluate, line_plot

# ---------------------------------------------------------------------------
#  CONFIG -------------------------------------------------------------------
# ---------------------------------------------------------------------------
CONFIG_PATH = pathlib.Path("config") / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config/config.yaml is missing – please ensure it is packaged.")

with CONFIG_PATH.open("r") as fh:
    CONFIG: Dict[str, Any] = yaml.safe_load(fh) or {}

# apply QUICK_TEST override --------------------------------------------------
CONFIG.setdefault("global", {})
CONFIG["global"]["quick_test"] = bool(
    int(os.getenv("QUICK_TEST", str(int(CONFIG["global"].get("quick_test", 0)))))
)

# ---------------------------------------------------------------------------
#  OUTPUT DIRECTORIES --------------------------------------------------------
# ---------------------------------------------------------------------------
RESEARCH_ROOT = pathlib.Path(".research") / "iteration2"
IMAGES_DIR = RESEARCH_ROOT / "images"
RESULTS_DIR = RESEARCH_ROOT  # JSON files stored at root per instructions
RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  EXPERIMENT 1 (Tensor-Spectrum Generalisation) -----------------------------
# ---------------------------------------------------------------------------

def _synthetic_dataset(num_nodes: int, num_features: int, num_classes: int):
    """Generate a small synthetic dataset – used when OGB is unavailable."""
    feats = torch.randn(num_nodes, num_features)
    labels = torch.randint(0, num_classes, (num_nodes,))
    idx = torch.randperm(num_nodes)
    n_train = int(0.8 * num_nodes)
    n_val = int(0.1 * num_nodes)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return feats, labels, {"train": train_idx, "valid": val_idx, "test": test_idx}


def run_experiment_1(cfg: Dict[str, Any], *, quick: bool) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset (ogbn-mag) --------------------------------------------------
    try:
        ogbn = ensure_ogb("ogbn-mag")
        split = ogbn.get_idx_split()
        graph = ogbn[0]
        feats = graph.ndata["feat"].float()
        labels = graph.ndata["label"].squeeze()
        num_features = feats.shape[1]
        num_classes = int(labels.max().item() + 1)
    except RuntimeError as exc:
        print("[Data] Falling back to synthetic dataset:", exc)
        num_nodes = 4096 if quick else 20000
        num_features = 128
        num_classes = 5
        feats, labels, split = _synthetic_dataset(num_nodes, num_features, num_classes)

    train_idx = split["train"]
    val_idx = split["valid"]
    test_idx = split["test"]
    if quick:
        train_idx = train_idx[:1024]
        val_idx = val_idx[:512]
        test_idx = test_idx[:512]

    loaders = {
        "train": tensor_loader_from_feats_labels(
            feats[train_idx], labels[train_idx], cfg["training"]["batch_size"]
        ),
        "val": tensor_loader_from_feats_labels(feats[val_idx], labels[val_idx], 4096, shuffle=False),
        "test": tensor_loader_from_feats_labels(feats[test_idx], labels[test_idx], 4096, shuffle=False),
    }

    # 2. Model ---------------------------------------------------------------
    model = HyperWaveGuard(num_features, num_classes, use_tsps=True).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

    best_hit, best_state = 0.0, None
    patience, epochs = 0, (1 if quick else cfg["training"]["epochs"])

    val_hits: List[float] = []
    for epoch in range(epochs):
        _ = train_one_epoch(model, optimiser, loaders["train"], device)
        val_res = evaluate(model, loaders["val"], device)
        val_hit = val_res["hit@1"]
        val_hits.append(val_hit)
        if val_hit > best_hit:
            best_hit = val_hit
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if not quick and patience >= cfg["training"].get("early_stop", 5):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate(model, loaders["test"], device)

    # 3. Persist results -----------------------------------------------------
    res = {
        "val_hit@1": best_hit,
        "test_hit@1": test_res["hit@1"],
        "epochs": epoch + 1,
    }

    exp_path = RESULTS_DIR / "exp1_results.json"
    with exp_path.open("w") as fh:
        json.dump(res, fh, indent=2)
    print("\n========== EXPERIMENT 1 – Tensor-Spectrum Generalisation ==========")
    print(cfg.get("description", ""))
    print(json.dumps(res, indent=2))

    # 4. Plot ---------------------------------------------------------------
    fig_path = line_plot(
        xs=list(range(1, len(val_hits) + 1)),
        ys=val_hits,
        xlabel="Epoch",
        ylabel="Val hit@1",
        title="Exp-1 Val Accuracy",
        fname="accuracy_exp1",
        images_root=IMAGES_DIR,
    )
    print("Figure saved:", fig_path)
    return res


# ---------------------------------------------------------------------------
#  STUBS FOR EXP-2 / EXP-3 ---------------------------------------------------
# ---------------------------------------------------------------------------

def run_experiment_2(cfg: Dict[str, Any]):  # noqa: D401 – stub
    raise RuntimeError("HyperStream-Bench dataset not accessible – cannot run Experiment 2.")


def run_experiment_3(cfg: Dict[str, Any]):  # noqa: D401 – stub
    raise RuntimeError("HyperStream-Bench dataset not accessible – cannot run Experiment 3.")


# ---------------------------------------------------------------------------
#  MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    quick = CONFIG["global"].get("quick_test", False)
    set_seed(42)

    results_master: Dict[str, Any] = {}

    # EXP 1 -----------------------------------------------------------------
    res1 = run_experiment_1(CONFIG["experiment_1"], quick=quick)
    results_master["experiment_1"] = res1

    # EXP 2 -----------------------------------------------------------------
    try:
        res2 = run_experiment_2(CONFIG.get("experiment_2", {}))
        results_master["experiment_2"] = res2
    except RuntimeError as exc:
        print("[Exp-2] ABORTED:", exc)

    # EXP 3 -----------------------------------------------------------------
    try:
        res3 = run_experiment_3(CONFIG.get("experiment_3", {}))
        results_master["experiment_3"] = res3
    except RuntimeError as exc:
        print("[Exp-3] ABORTED:", exc)

    # SUMMARY ---------------------------------------------------------------
    summary_path = RESULTS_DIR / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(results_master, fh, indent=2)

    print("\n================ OVERALL SUMMARY =================")
    print(json.dumps(results_master, indent=2))


if __name__ == "__main__":
    main()
