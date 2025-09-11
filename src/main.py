"""
src/main.py
Entry-point orchestrating full experimental workflow.
Fixed for iteration-5:
  • Updated banner text to Iteration-5 to match artefact paths.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .train import train_all_seeds
from .evaluate import evaluate_cafe_edge

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


def main():  # noqa: D401 – imperative main OK
    print("************************************************************")
    print("  CaFe-EDGE  –  Reproducibility Suite (Iteration-5)")
    print("  Strict-No-Fallback is active – real data mandatory (tiny CI sample provided).")
    print("************************************************************\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("INFO: CUDA not available – running on CPU. Training will be slow but functional.")

    # -------------------  Experiment 1  ----------------------
    ckpts = train_all_seeds(device)
    json_path, fig_path, _ = evaluate_cafe_edge(ckpts)

    print("\nGenerated artefacts:")
    print(f"  • Results JSON : {json_path}")
    print(f"  • Figure       : {fig_path}")


if __name__ == "__main__":
    main()
