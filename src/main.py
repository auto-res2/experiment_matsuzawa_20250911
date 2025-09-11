"""Entry-point helper so `python -m raptor.run --config=...` works.

Bug-fixes (iteration14)
-----------------------
1. `_parse` now accepts an *optional* `argv` list allowing callers to
   supply a pre-tokenised argument vector.  This resolves the TypeError:
      _parse() takes 0 positional arguments but 1 was given
   that occurred when `main(sys.argv[1:])` passed the CLI arguments.
2. Delegates path handling to the updated training code which now obeys
   the iteration15 directory policy.
"""
from __future__ import annotations

import argparse
import sys

from .train import fit, load_yaml


def _parse(argv: list[str] | None = None):
    """Parse CLI arguments.

    Parameters
    ----------
    argv : list[str] | None
        When *None* (default) the current `sys.argv[1:]` is parsed.
        Otherwise the provided list is treated as the argument vector –
        this is crucial for programmatic calls such as unit tests.
    """
    p = argparse.ArgumentParser(description="Run miniature RAPTOR experiment")
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to Hydra-style YAML")
    p.add_argument("--variant", type=str, default="v0", help="Experiment variant id (v0, v1, …)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None):  # noqa: D401 – CLI wrapper
    args = _parse(argv)
    exp_conf = load_yaml(args.config)
    fit(exp_conf, args.variant, args.seed)


if __name__ == "__main__":  # pragma: no cover – manual execution
    main(sys.argv[1:])
