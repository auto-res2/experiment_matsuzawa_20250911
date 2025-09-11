"""src/main.py
-------------------------------------------------------------------------
Entry-point that orchestrates preparation and then delegates to
`src.train.run_full_training`.

Key changes in this revision
----------------------------
1.  All artefacts are now written to `.research/iteration7/` as required by
    the latest task specification.
2.  The train interface now expects the explicit `ci_mode` boolean so that
    it can *fail-fast* outside CI.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .preprocess import _SKIP as PREPROCESS_SKIP  # CI flag from preprocess.py
from .train import ExperimentConfig, run_full_training

# ---------------------------------------------------------------------
# 0.  Paths & constants
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

# Directory mandated by the instructions for result JSONs
RESEARCH_DIR = ROOT_DIR / ".research" / "iteration7"


# ---------------------------------------------------------------------
# 1.  Hardware sanity check helpers
# ---------------------------------------------------------------------
_REQUIRED_ENV_VARS = [
    "EDGE_HARDWARE_AVAILABLE",  # must be "1" when boards are connected
]


def _hardware_available() -> bool:  # noqa: D401
    """Return *True* iff all required environment flags are present."""

    return all(os.getenv(var, "0") == "1" for var in _REQUIRED_ENV_VARS)


# ---------------------------------------------------------------------
# 2.  Main launcher
# ---------------------------------------------------------------------

def main() -> None:  # noqa: D401
    # CI mode is active when either the usual CI variables *or* the
    # SKIP_DATA_DOWNLOAD flag (shared with preprocess.py) is set.
    ci_mode = (
        os.getenv("CI", "0") == "1"
        or os.getenv("CI_TEST_ENV", "0") == "1"
        or PREPROCESS_SKIP
    )

    # --------------------------------------------------------------
    # Load configuration (YAML → dataclass)
    # --------------------------------------------------------------
    if not DEFAULT_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Configuration file {DEFAULT_CONFIG_PATH} missing.  Please place your YAML there."
        )

    cfg = ExperimentConfig.from_yaml(DEFAULT_CONFIG_PATH)

    # --------------------------------------------------------------
    # Dataset acquisition (forced no-op in CI)
    # --------------------------------------------------------------
    data_root = ROOT_DIR / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Hardware guard – only warn in CI, fail otherwise
    # --------------------------------------------------------------
    if not _hardware_available() and not ci_mode:
        raise RuntimeError(
            "Required mixed-signal hardware not detected.  Set "
            "‘EDGE_HARDWARE_AVAILABLE=1’ when boards are physically connected."
        )

    # --------------------------------------------------------------
    # Launch training (analytical estimator when in CI)
    # --------------------------------------------------------------
    run_full_training(cfg, ci_mode=ci_mode)

    # Show artefacts generated during CI
    if ci_mode:
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        contents = sorted(p.name for p in RESEARCH_DIR.iterdir())
        print("\n[INFO] .research/iteration7 contents after run: " + json.dumps(contents), flush=True)

    sys.exit(0)


# ---------------------------------------------------------------------
# 3.  Module guard
# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
