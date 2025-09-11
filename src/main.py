# src/main.py
# -------------------------------------------------------------
# Orchestrates the full experimental pipeline via relative imports.
# -------------------------------------------------------------
from __future__ import annotations

import yaml

from .preprocess import (
    ExperimentConfig,
    download_edgebench,
    check_hardware_available,
)
from .train import run_experiment_1
from .evaluate import save_results  # plotting utilities can be imported if needed


def _load_cfg() -> ExperimentConfig:
    """Load YAML into ExperimentConfig dataclass."""
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not cfg_path.exists():
        # Fall back to defaults
        return ExperimentConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return ExperimentConfig(**data)


def main() -> None:  # noqa: D401 – imperative style
    cfg = _load_cfg()

    # 1. Verify hardware presence
    check_hardware_available()

    # 2. Ensure dataset is present locally
    download_edgebench(cfg)

    # 3. Run Experiment-1 (will abort if dataset parsing is not implemented)
    results = run_experiment_1(cfg)

    # 4. Persist raw results JSON in .research/iteration8/
    save_results("experiment_1", results)


if __name__ == "__main__":
    main()
