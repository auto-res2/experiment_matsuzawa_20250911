"""Command-line interface for the refactored experiment.

Supported execution patterns:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment

The script performs minimal structural validation exactly as the
original single-file implementation did while delegating the three core
stages (pre-processing, training, evaluation) to their respective
modules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

import src.preprocess as _pre
import src.train as _train
import src.evaluate as _eval

# ---------------------------------------------------------------------------
# Configuration helpers (adapted from the original src/config.py)
# ---------------------------------------------------------------------------

def _validate_config(cfg: Dict[str, Any], cfg_path: Path) -> None:
    missing: list[str] = []
    if not cfg.get("dataset", {}).get("url"):
        missing.append("dataset.url")
    if not cfg.get("model", {}).get("name"):
        missing.append("model.name")
    if missing:
        sys.stderr.write(
            f"[FATAL] Configuration error in {cfg_path} – missing required field(s): "
            f"{', '.join(missing)}\n"
        )
        sys.stderr.write(
            "The STRICT NO-FALLBACK RULE prohibits the use of synthetic or "
            "placeholder data.  Please provide real dataset URLs before "
            "re-running.\n"
        )
        sys.exit(1)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        sys.stderr.write(f"[FATAL] Configuration file not found: {path}\n")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg, path)
    return cfg

# ---------------------------------------------------------------------------
# Main orchestration helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental pipeline runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="run full experiment")
    return parser.parse_args()


def _config_path_from_args(args: argparse.Namespace) -> Path:
    if args.smoke_test:
        return Path("config/smoke_test.yaml")
    if args.full_experiment:
        return Path("config/full_experiment.yaml")
    # argparse guarantees one flag, but keep mypy happy
    raise AssertionError("Unreachable")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover – CLI entry-point
    args = _parse_args()
    cfg_path = _config_path_from_args(args)
    cfg = _load_yaml(cfg_path)

    # ---------------------------------------------------------
    # Phase 1 – Pre-processing (download / extract)
    # ---------------------------------------------------------
    try:
        dataset_path = _pre.download_and_prepare(cfg["dataset"])
    except Exception as e:  # noqa: BLE001 – generic handler (network, checksum …)
        sys.stderr.write(f"[FATAL] Dataset preparation failed: {e}\n")
        sys.exit(1)

    # ---------------------------------------------------------
    # Phase 2 – Training (stub)
    # ---------------------------------------------------------
    train_outputs = _train.train(cfg, dataset_path)

    # ---------------------------------------------------------
    # Phase 3 – Evaluation (stub)
    # ---------------------------------------------------------
    results = _eval.evaluate(cfg, train_outputs)

    # ---------------------------------------------------------
    # Persist & report
    # ---------------------------------------------------------
    mode = "smoke" if args.smoke_test else "full"
    out_path = Path(".research/iteration2") / f"{mode}_results.json"
    _eval.save_json(results, out_path)

    # Print to STDOUT for verification (keeps CI happy)
    import json  # local import keeps global namespace tidy

    sys.stdout.write(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
