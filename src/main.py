"""src/main.py – entry-point:  `python -m src.main`
Now points to iteration7 artefact directory and gracefully skips experiments
that rely on the optional `conductor-ai` package when it is not installed.
"""
from __future__ import annotations

import pathlib, yaml, sys
from typing import Dict, Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_YAML: Dict[str, Any] = {
    "hardware": {
        "node": "DGX-A100",
        "gpus": 8,
        "gpu_type": "A100-80GB",
        "docker": "nvcr.io/nvidia/pytorch:24.02-py3",
        "cuda": "12.3",
    },
    "seeds": [11, 13, 17, 19, 23],
    "datasets": {
        "wmt22_en_de": "https://huggingface.co/datasets/wmt22",
        "cath_43": "https://huggingface.co/datasets/cctien/protein_backbone_cath_4.3",
        "humaneval": "https://huggingface.co/datasets/openai/openai_humaneval",
        "mt50k": "s3://conductor-public/mt50k_v1.csv",
    },
    "models": {
        "bitdiff_t5_l": "bitdiffusion/bitdiffusion-t5-large",
        "rfdiffusion_xl": "rf-diffusion/rfdiffusion-xl",
        "absorbing_gpt3_1b3": "absorbing-lab/absorbing-gpt-3-1_3B",
        "vit_diffuser_900m": "conductor-ai/vit-diffuser-900m",
    },
    "hyper": {
        "lr": [1e-5, 3e-5, 1e-4],
        "tau": [0.7, 1.0],
        "K": [4, 8, 16],
        "sigma": [0.7, 1.0, 1.3],
        "token_skip": [0.0, 0.1, 0.25],
    },
}

if not CONFIG_FILE.exists():
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(_DEFAULT_YAML, f)

with open(CONFIG_FILE) as f:
    cfg: Dict[str, Any] = yaml.safe_load(f)

# Lazy import after config is ready
from src.train import run_exp1, run_exp2, run_exp3  # noqa: E402 – delayed import


def main():
    for seed in cfg["seeds"]:
        run_exp1(seed, cfg)
        run_exp2(seed, cfg)
        run_exp3(seed, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
