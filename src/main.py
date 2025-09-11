"""
src/main.py
Entry-point: orchestrates full experimental workflow.
Run via:  python -m src.main
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from .train import train_all_seeds
from .evaluate import evaluate_cafe_edge

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


def main():  # noqa: D401 – imperative main OK
    print("************************************************************")
    print("  CaFe-EDGE  –  Reproducibility Suite (Iteration-1)")
    print("  Strict-No-Fallback is active – real data mandatory!")
    print("************************************************************\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("ERROR: NVIDIA GPU required – aborting.")

    # -------------------  Experiment 1  ----------------------
    ckpts = train_all_seeds(device)
    json_path, fig_path, _ = evaluate_cafe_edge(ckpts)

    print("\nGenerated artefacts:")
    print(f"  • Results JSON : {json_path}")
    print(f"  • Figure       : {fig_path}")


if __name__ == "__main__":
    main()
