# src/main.py
"""Entry-point orchestrating the MAESTRO experiments.
Run via `python -m src.main` (package mode ensures relative imports work).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

# Local imports – pylint/ruff might complain about relative-import style when
# "src" is treated as a top-level package, but RuntimeLoader uses package mode.
from .train import run_experiment_1

# ---------------------------------------------------------------------------
#   Configuration loader (reads YAML into a nested dict of dicts)
# ---------------------------------------------------------------------------
_CFG_PATH = Path("config/config.yaml").resolve()
if not _CFG_PATH.exists():
    raise FileNotFoundError(
        "Expected config/config.yaml next to the source tree.  Please make "
        "sure it is generated during the repo bootstrap stage."
    )


def _load_cfg() -> Dict[str, Any]:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:  # noqa: D401 – simple script
    cfg = _load_cfg()
    # Only experiment 1 is executed in the public stub.
    run_experiment_1(cfg["experiment_1"])


if __name__ == "__main__":  # pragma: no cover – CLI guard
    main()
