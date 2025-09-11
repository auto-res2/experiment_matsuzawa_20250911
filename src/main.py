"""Entry-point helper so `python -m raptor.run +config=...` in the spec
maps to a single file for this lightweight repo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .train import fit, load_yaml


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run miniature RAPTOR experiment")
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to Hydra-style YAML")
    p.add_argument("--variant", type=str, default="v0", help="Experiment variant id (v0, v1, …)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()


def main(argv=None):  # noqa: D401 – CLI wrapper
    args = _parse() if argv is None else _parse(argv)
    exp_conf = load_yaml(args.config)
    fit(exp_conf, args.variant, args.seed)


if __name__ == "__main__":  # pragma: no cover – manual execution
    main(sys.argv[1:])