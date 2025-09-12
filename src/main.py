"""src/main.py
Entry-point orchestrating smoke-test and full experiments.

Usage
-----
Smoke-test only:
    uv run python -m src.main --smoke-test
Full experiment only:
    uv run python -m src.main --full-experiment
Both (default – runs smoke-test first, then full):
    uv run python -m src.main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml
import torch

from .preprocess import get_dataloaders
from .train import SimpleCNN, train_one_epoch
from .evaluate import evaluate, plot_confusion_matrix, plot_training_curves

# -----------------------------------------------------------------------------
# DIRECTORIES (relative to project root)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
RESEARCH_DIR = ROOT_DIR / ".research" / "iteration2"
FIG_DIR = RESEARCH_DIR / "images"
RESULT_DIR = RESEARCH_DIR

for _d in (CONFIG_DIR, DATA_DIR, FIG_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SMOKE_CFG_PATH = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG_PATH = CONFIG_DIR / "full_experiment.yaml"


# -----------------------------------------------------------------------------
# DEFAULT YAML GENERATION (one-time)
# -----------------------------------------------------------------------------

def _create_default_configs() -> None:
    """Create the two YAML configuration files if they do not yet exist."""

    if not SMOKE_CFG_PATH.exists():
        smoke = {
            "experiment_name": "fashion_mnist_smoke",
            "dataset": {
                "name": "FashionMNIST",
                "url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                "root": str(DATA_DIR / "fashion_mnist"),
                "train_subset": 1000,
                "test_subset": 1000,
            },
            "model": {"type": "SimpleCNN", "num_classes": 10},
            "training": {"epochs": 1, "batch_size": 64, "lr": 0.01, "momentum": 0.9},
            "evaluation": {"metrics": ["accuracy"]},
            "output": {"results_path": str(RESULT_DIR), "figures_path": str(FIG_DIR)},
        }
        with open(SMOKE_CFG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(smoke, fh)

    if not FULL_CFG_PATH.exists():
        full = {
            "experiment_name": "fashion_mnist_full",
            "dataset": {
                "name": "FashionMNIST",
                "url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                "root": str(DATA_DIR / "fashion_mnist"),
                "train_subset": None,
                "test_subset": None,
            },
            "model": {"type": "SimpleCNN", "num_classes": 10},
            "training": {"epochs": 10, "batch_size": 64, "lr": 0.01, "momentum": 0.9},
            "evaluation": {"metrics": ["accuracy"]},
            "output": {"results_path": str(RESULT_DIR), "figures_path": str(FIG_DIR)},
        }
        with open(FULL_CFG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(full, fh)


# -----------------------------------------------------------------------------
# SINGLE EXPERIMENT WORKFLOW
# -----------------------------------------------------------------------------

def _run_experiment(cfg: Dict) -> None:  # noqa: C901 – keep unified logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Data ----------------
    train_loader, test_loader = get_dataloaders(cfg)

    # ---------------- Model --------------
    if cfg["model"]["type"] != "SimpleCNN":
        raise ValueError("Unsupported model type in configuration – only SimpleCNN is available.")
    model = SimpleCNN(num_classes=cfg["model"]["num_classes"]).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
    )

    all_losses: List[float] = []
    all_accs: List[float] = []

    # ------------- Training loop ---------
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss_epoch = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc_epoch, _, _ = evaluate(model, test_loader, device)
        all_losses.append(loss_epoch)
        all_accs.append(acc_epoch)
        print(f"Epoch {epoch}/{cfg['training']['epochs']}  Loss: {loss_epoch:.4f}  Acc: {acc_epoch:.4f}")
        sys.stdout.flush()

    # ------------- Final evaluation ------
    final_acc, preds, labels = evaluate(model, test_loader, device)

    # ------------- Figures ---------------
    fig_loss, fig_acc = plot_training_curves(all_losses, all_accs, FIG_DIR)
    fig_cm = plot_confusion_matrix(labels, preds, FIG_DIR)

    # ------------- Results ---------------
    results = {
        "experiment_name": cfg["experiment_name"],
        "final_test_accuracy": final_acc,
        "loss_per_epoch": all_losses,
        "accuracy_per_epoch": all_accs,
        "figures": [fig_loss, fig_acc, fig_cm],
    }

    result_path = RESULT_DIR / f"results_{cfg['experiment_name']}.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ----------- Console summary ---------
    print("\n=====================  EXPERIMENT DESCRIPTION  =====================")
    print(
        f"Experiment '{cfg['experiment_name']}' – Classification on {cfg['dataset']['name']}\n"
        f"Training epochs : {cfg['training']['epochs']}\n"
        f"Batch size      : {cfg['training']['batch_size']}\n"
        f"Learning rate   : {cfg['training']['lr']}\n"
        f"Momentum        : {cfg['training']['momentum']}\n"
        f"Train subset    : {cfg['dataset']['train_subset']}\n"
        f"Test  subset    : {cfg['dataset']['test_subset']}"
    )
    print("\n=====================  NUMERICAL RESULTS  ==========================")
    print(json.dumps(results, indent=2))
    print("\n=====================  FIGURE FILES  ===============================")
    for f in results["figures"]:
        print(f)


# -----------------------------------------------------------------------------
# ARGPARSE & CONTROL FLOW
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN experiment runner")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--smoke-test", action="store_true", help="run the quick smoke-test only")
    grp.add_argument("--full-experiment", action="store_true", help="run the full experiment only")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    _create_default_configs()
    args = _parse_args()

    if args.smoke_test:
        cfg_paths = [SMOKE_CFG_PATH]
    elif args.full_experiment:
        cfg_paths = [FULL_CFG_PATH]
    else:  # default – both, smoke-test first
        cfg_paths = [SMOKE_CFG_PATH, FULL_CFG_PATH]

    for cfg_path in cfg_paths:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        print("\n#######################  RUNNING", cfg["experiment_name"].upper(), "#######################\n")
        _run_experiment(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
