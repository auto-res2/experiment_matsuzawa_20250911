# src/main.py
"""Unified entry point orchestrating PHOENIX-Mem experiments.

All artefacts (figures, JSON) are written to the mandatory
`.research/iteration3` hierarchy as required by the task description.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Local imports (relative to "src")
from .preprocess import build_loader
from .train import PhoenixMem, train_model
from .evaluate import evaluate_model

# =========================
#  Paths
# =========================
RESEARCH_ROOT = Path(".research/iteration3")
IMG_DIR = RESEARCH_ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = RESEARCH_ROOT  # JSON files live directly here per instructions
RES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("config/config.yaml")

# =========================
#  Helpers
# =========================

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Configuration file {CONFIG_PATH} missing – abort.")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# =========================
#  Experiment 1
# =========================

def run_experiment_1(cfg: dict):
    print("\n=== Experiment 1 – Component-level Compression & Reliability ===")

    # ---------- Data ----------
    root = Path("data")
    loader, n_classes = build_loader(root, cfg)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoenixMem(use_causal=True, ecc_r=14, sram_protect=True, num_classes=n_classes).to(device)

    # ---------- Training ----------
    try:
        model, train_logs = train_model(model, loader, cfg, device)
    except RuntimeError as e:
        raise RuntimeError("Training failed – dataset unavailable or corrupt.") from e

    # ---------- Evaluation ----------
    metrics = evaluate_model(model, loader, device)

    # ---------- Serialise results ----------
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        **metrics,
        "train_logs": train_logs,
        "config": {**model.metadata, **cfg},
    }
    out_file = RES_DIR / "experiment1_component_level.json"
    out_file.write_text(json.dumps(result, indent=2))
    print("\nExperimental numerical data:")
    print(json.dumps(result, indent=2))

    # ---------- Dummy figure ----------
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        epochs = range(len([k for k in train_logs if k.startswith("epoch_")]))
        losses = [train_logs[f"epoch_{e}_loss"] for e in epochs]
        plt.plot(epochs, losses, marker="o")
        plt.title("Training loss (CIFAR-10 stub)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        fig_path = IMG_DIR / "training_loss_phoenix_mem.pdf"
        plt.savefig(fig_path, bbox_inches="tight")
        print("Names of figures summarising the numerical data:")
        print(fig_path.relative_to(RESEARCH_ROOT))
    except Exception as e:
        print(f"Figure generation failed: {e}")


# =========================
#  Main
# =========================

def main():
    try:
        cfg = load_config()
        start = time.time()
        run_experiment_1(cfg["experiment1"])
        # Potential: run_experiment_2(cfg["experiment2"], ...)
        print(f"\nAll requested experiments completed in {(time.time() - start) / 60:.2f} min.")
    except Exception:
        print("\nFATAL: experiment pipeline aborted. Reason:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
