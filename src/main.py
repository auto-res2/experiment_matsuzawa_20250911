"""src/main.py
-------------------------------------------------------------------------
Entry-point that orchestrates the preparation steps and then delegates to
`src.train.run_full_training`.  Must be called as

    python -m src.main
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .preprocess import download_file, extract_tar
from .train import ExperimentConfig, run_full_training

# ---------------------------------------------------------------------
# 0.  Paths & constants
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

# ---------------------------------------------------------------------
# 1.  Hardware sanity check helpers
# ---------------------------------------------------------------------
_REQUIRED_ENV_VARS = [
    "EDGE_HARDWARE_AVAILABLE",  # must be "1" when boards are connected
]


def _check_hardware_available() -> None:
    for var in _REQUIRED_ENV_VARS:
        if os.getenv(var, "0") != "1":
            raise RuntimeError(
                "Required mixed-signal hardware not detected.  Environment "
                f"variable ‘{var}=1’ must be set when InP-PCM boards and GAP9 "
                "SoCs are physically connected."
            )
    print("[INFO] Required edge hardware detected via environment flags.")


# ---------------------------------------------------------------------
# 2.  Main launcher
# ---------------------------------------------------------------------

def main() -> None:  # noqa: D401
    # ------------------------------------------------------------------
    # Load configuration (YAML → dataclass)
    # ------------------------------------------------------------------
    if not DEFAULT_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Configuration file {DEFAULT_CONFIG_PATH} missing.  Please place your YAML there."
        )

    cfg = ExperimentConfig.from_yaml(DEFAULT_CONFIG_PATH)

    # ------------------------------------------------------------------
    # Hardware availability guard (STRICT NO-FALLBACK)
    # ------------------------------------------------------------------
    _check_hardware_available()

    # ------------------------------------------------------------------
    # Dataset acquisition
    # ------------------------------------------------------------------
    data_root = ROOT_DIR / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_root / f"{cfg.dataset.name.lower()}-{cfg.dataset.version}"

    if not dataset_dir.exists():
        archive_path = data_root / "edgebench-48.tar.gz"
        if not archive_path.exists():
            download_file(cfg.dataset.url, archive_path, cfg.dataset.sha256)
        extract_tar(archive_path, data_root)

    # Final guard – still missing → abort.
    if not dataset_dir.exists():
        raise RuntimeError(
            f"Dataset directory {dataset_dir} still missing after download attempt. "
            "Execution cannot proceed without the real EdgeBench-48 dataset."
        )

    # ------------------------------------------------------------------
    # Launch training & evaluation (will abort when hardware is absent)
    # ------------------------------------------------------------------
    run_full_training(cfg, dataset_dir)


# ---------------------------------------------------------------------
# 3.  Module guard
# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
