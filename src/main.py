from __future__ import annotations
"""src/main.py
Entry-point that orchestrates the three studies and stores every artefact in
``.research/iteration14`` exactly as required by the specification.
Run with e.g.::

    python -m tiny_taco.main  # runs all studies on CPU
"""
import argparse
from pathlib import Path

import torch

from .evaluate import run_study1, run_study2, run_study3

# ---------------------------------------------------------------------------
# Default human-readable config (used if ``config.yaml`` is empty/missing)
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {
    "models": {
        "vision_trunk": "mobilevit_s.cvnets_in1k",  # timm name only
        "audio_trunk": "openai/whisper-tiny",
        "text_trunk": "distilbert-base-uncased",
    },
    "hyper_params": {
        "probe_dim": 32,
        "lr": 1e-4,
    },
}


def _parse_args():
    p = argparse.ArgumentParser(description="Run Tiny-TACO studies")
    p.add_argument("--study", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():  # noqa: D403
    args = _parse_args()

    # All artefacts – JSON logs and figures – must live here per spec
    out_dir = Path(".research/iteration14")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    if args.study in {"1", "all"}:
        run_study1(device, _DEFAULT_CFG, out_dir)
    if args.study in {"2", "all"}:
        run_study2(out_dir)
    if args.study in {"3", "all"}:
        run_study3(out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
