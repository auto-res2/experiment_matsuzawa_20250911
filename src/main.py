# src/main.py
"""Project entry-point with CLI flags for smoke/full experiments."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

# Local imports
from .preprocess import load_tiny_imagenet_splits
from .train import FederatedTrainer

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_cfg(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r") as fp:
        return yaml.safe_load(fp)


# ------------------------------------------------------------------
# CLI orchestrator
# ------------------------------------------------------------------

def _run(cfg: Dict) -> None:  # noqa: D401
    # ------------------------------------------------------------------
    # Prepare output directories (research iteration structure)
    # ------------------------------------------------------------------
    base_research_dir = Path(".research") / "iteration1"
    images_dir = base_research_dir / "images"
    results_dir = base_research_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    # CUDA visibility (respect the number of GPUs requested in config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(int(cfg["global"]["num_gpus"]))))

    # Determinism
    _set_seeds(cfg["global"]["seed_list"][0])

    # --------------------------------------------------------------
    # EXPERIMENT 1  –  Asynchronous FL on Tiny-ImageNet
    # --------------------------------------------------------------
    print("Preparing Tiny-ImageNet sequential tasks …")
    tasks = load_tiny_imagenet_splits(cfg)
    trainer = FederatedTrainer(tasks, cfg, results_dir=str(results_dir), figure_dir=str(images_dir))
    trainer.run()


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="HERMES-MEM experimental runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick validation experiment (1-2 epochs)")
    group.add_argument("--full-experiment", action="store_true", help="Run complete experimental suite")
    args = parser.parse_args()

    cfg_dir = Path("config")
    if args.smoke_test:
        cfg = _load_cfg(cfg_dir / "smoke_test_config.yaml")
    else:  # --full-experiment
        cfg = _load_cfg(cfg_dir / "full_experiment_config.yaml")

    try:
        _run(cfg)
    except Exception as exc:
        print(f"[ERROR] Experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
