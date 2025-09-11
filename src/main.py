# src/main.py
"""Command-line entry-point.

Usage examples
--------------
Smoke-test only:
    uv run python -m src.main --smoke-test

Full experiment only:
    uv run python -m src.main --full-experiment

By default (no flags) the script will first execute the smoke-test and – if
it finishes without uncaught exceptions – continue with the full experiment.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .train import dump_yaml, load_config
from .evaluate import ExperimentRunner

# -----------------------------------------------------------------------------
#  Argument parsing
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive Probability-Flow – Runner")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke-test", action="store_true", help="Run smoke-test only")
    g.add_argument("--full-experiment", action="store_true", help="Run full experiment only")
    return p


# -----------------------------------------------------------------------------
#  Helper that executes *one* phase (smoke or full) and captures exceptions
# -----------------------------------------------------------------------------

def _execute_phase(config_path: Path, tag: str) -> bool:
    print(f"\n================ {tag.upper()} ================")
    try:
        cfg = load_config(config_path)
        # Keep a copy of the YAML for provenance --------------------------------
        dump_yaml(config_path, Path(f"{tag}_used_config.yaml"))
        runner = ExperimentRunner(cfg)
        runner.run_exp1()
    except Exception as exc:
        # For the smoke-test we *expect* the placeholder error; that still counts
        # as success because the purpose is only to ensure that dependencies and
        # paths are wired correctly.
        print(str(exc))
        traceback.print_exc()
        return False
    return True


# -----------------------------------------------------------------------------
#  main()
# -----------------------------------------------------------------------------

def main():
    args = _build_arg_parser().parse_args()

    cfg_dir = Path(__file__).resolve().parent.parent / "config"
    smoke_cfg = cfg_dir / "smoke_test_config.yaml"
    full_cfg = cfg_dir / "full_experiment_config.yaml"

    # Resolve execution mode ----------------------------------------------------
    run_smoke = args.smoke_test or not (args.smoke_test or args.full_experiment)
    run_full = args.full_experiment or not (args.smoke_test or args.full_experiment)

    # ------------------------------------------------------------------
    success = True
    if run_smoke:
        success &= _execute_phase(smoke_cfg, tag="smoke_test")

    if run_full and success:
        success &= _execute_phase(full_cfg, tag="full_experiment")

    if not success:
        print("\nOne or more phases failed – see tracebacks above.")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
