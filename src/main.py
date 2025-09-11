"""
src/main.py
===========
Entry-point that orchestrates all experiments.  The script searches for
configuration under `config/config.yaml`, dispatches to the experiment registry
and stores results / figures under `.research/iteration1`.

Run via:  python -m src.main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import yaml

from .evaluate import EXPERIMENT_REGISTRY, print_heading


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[1] / "config" / "config.yaml"


def _load_cfg() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "config/config.yaml missing – please ensure the repository is up to date."
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple procedural entry-point
    cfg = _load_cfg()

    # Root directory mandated by the assignment
    results_root = Path(".research/iteration1")
    images_root = results_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    # Iterate over all experiments declared in YAML – this is future-proof
    for exp_name, exp_cfg in cfg.items():
        print_heading(f"RUNNING {exp_name.upper()}")
        runner_cls = EXPERIMENT_REGISTRY.get(exp_name)
        if runner_cls is None:
            print(f"[WARN] No experiment runner registered for '{exp_name}'. Skipping…")
            continue

        exp_dir = results_root  # Figures are already saved to images_root inside the runner
        try:
            runner = runner_cls(exp_cfg, results_root)
            metrics, figure_files = runner.run()
        except Exception as exc:
            print(f"[ERROR] Experiment '{exp_name}' failed: {exc}")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Persist results – each experiment gets its own JSON file
        # ------------------------------------------------------------------
        json_path = results_root / f"{exp_name}.json"
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

        # Mandated stdout for verification
        print("\nEXPERIMENT DESCRIPTION:")
        print(exp_cfg["description"])
        print("\nNUMERICAL RESULTS:")
        print(json.dumps(metrics, indent=2))
        print("\nFIGURE FILES:")
        for f in figure_files:
            print(f" - {f.relative_to(results_root)}")
        print("=" * 60 + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
