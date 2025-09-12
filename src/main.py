"""Main orchestration script – provides CLI for smoke/full experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch.nn as nn
import torch.optim as optim

from .evaluate import evaluate, save_confusion_matrix, save_line_plot
from .preprocess import get_data_loaders, load_yaml, set_seed
from .train import Metrics, SimpleCNN, train_epoch

# ------------------------- paths / config -------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SMOKE_CFG = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG = CONFIG_DIR / "full_experiment.yaml"

RESEARCH_DIR = BASE_DIR / ".research" / "iteration2"
IMAGES_DIR = RESEARCH_DIR / "images"

# Ensure top-level research directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------ experiment logic ------------------------- #


def run_experiment(cfg: Dict):
    """Runs a *single* experiment as described by *cfg*."""

    # reproducibility
    set_seed(cfg["seed"])

    # --------------- data ----------------
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # --------------- model --------------
    model = SimpleCNN(cfg["model"]["input_channels"], cfg["model"]["num_classes"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = None
    if cfg["training"]["lr_scheduler"]["name"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"]["lr_scheduler"]["step_size"],
            gamma=cfg["training"]["lr_scheduler"]["gamma"],
        )

    metrics = Metrics(train_loss=[], train_acc=[], val_loss=[], val_acc=[], test_loss=0.0, test_acc=0.0)

    # --------------- training loop ---------------
    for _ in range(cfg["training"]["epochs"]):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        metrics.train_loss.append(t_loss)
        metrics.train_acc.append(t_acc)
        metrics.val_loss.append(v_loss)
        metrics.val_acc.append(v_acc)
        if scheduler is not None:
            scheduler.step()

    # --------------- test phase ------------------
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    metrics.test_loss = test_loss
    metrics.test_acc = test_acc

    # --------------- persistence -----------------
    exp_name = cfg["experiment_name"]
    result_path = RESEARCH_DIR / f"{exp_name}.json"
    with open(result_path, "w") as fp:
        json.dump(metrics.__dict__, fp, indent=2)

    # figures
    save_line_plot(metrics.train_loss, "Training Loss", IMAGES_DIR / f"{exp_name}_train_loss.pdf", f"{exp_name}: Training Loss")
    save_line_plot(metrics.val_loss, "Validation Loss", IMAGES_DIR / f"{exp_name}_val_loss.pdf", f"{exp_name}: Validation Loss")
    save_line_plot(metrics.train_acc, "Training Accuracy", IMAGES_DIR / f"{exp_name}_train_acc.pdf", f"{exp_name}: Training Accuracy")
    save_line_plot(metrics.val_acc, "Validation Accuracy", IMAGES_DIR / f"{exp_name}_val_acc.pdf", f"{exp_name}: Validation Accuracy")
    save_confusion_matrix(y_true, y_pred, IMAGES_DIR / f"{exp_name}_confusion_matrix.pdf")

    # ------------ stdout summary ---------------
    print("\n================ EXPERIMENT SUMMARY ================")
    print(json.dumps(metrics.__dict__, indent=2))
    print("===================================================\n")


# ----------------------------- CLI --------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST-CNN experiment runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true", help="Run only the smoke test experiment")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment (smoke test will be executed first)")
    return parser.parse_args()


def main():  # noqa: D401 – simple main
    args = parse_args()

    if not SMOKE_CFG.exists() or not FULL_CFG.exists():
        raise FileNotFoundError("Configuration YAML files not found under ./config/")

    smoke_cfg = load_yaml(SMOKE_CFG)
    full_cfg = load_yaml(FULL_CFG)

    # -------- execution path --------
    if args.smoke_test:
        run_experiment(smoke_cfg)
    elif args.full_experiment:
        print("Running preliminary smoke test ...")
        run_experiment(smoke_cfg)
        print("Smoke test passed – running full experiment ...")
        run_experiment(full_cfg)
    else:
        # default – only smoke test so that CI jobs remain quick
        run_experiment(smoke_cfg)


if __name__ == "__main__":
    main()
