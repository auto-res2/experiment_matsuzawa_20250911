"""src/main.py
Entry-point orchestrating the HydraSketch-Φ experimental workflow.
All heavy-weight logic has been delegated to the corresponding helper modules
so that this file mostly performs configuration I/O and high-level control-flow
as required by the task description.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .evaluate import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
)
from .preprocess import ensure_directories, download_dataset, extract_dataset

###############################################################################
# Configuration handling – serialised to config/config.yaml on first run
###############################################################################

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.yaml"

@dataclass
class ExperimentConfig:
    # Dataset ----------------------------------------------------------------
    dataset_name: str = "EdgeBench-48"
    dataset_version: str = "v1.2"
    dataset_url: str = (
        "https://edgebench.org/download/edgebench-48-v1.2.tar"
    )
    dataset_archive_name: str = "edgebench-48-v1.2.tar"

    # Models -----------------------------------------------------------------
    vision_backbone: str = "microsoft/resnet-18"
    radar_backbone: str = "timm/pointnetlite"
    gas_backbone: str = "custom_gru_128"
    ecg_backbone: str = "custom_resnet1d_10"

    # Hyper-parameters -------------------------------------------------------
    learning_rates: list[float] = (1e-3, 3e-4, 1e-4)
    fractional_sde_alpha: list[float] = (0.25, 0.5, 0.75)
    teleport_threshold: list[float] = (0.5, 1.0, 2.0)
    latency_grid: list[str] = ("lambda1", "lambda2", "lambda3", "lambda4")

    # Budgets ----------------------------------------------------------------
    ram_caps_kb: list[int] = (10, 50, 100)
    latency_caps_ms: list[int] = (15, 30)

    # Re-usable paths --------------------------------------------------------
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")
    figure_root: Path = Path("figures")

    # ---------------------------------------------------------------------
    def as_yaml(self) -> str:  # convenience helper
        return yaml.dump(asdict(self), sort_keys=False)

def _load_or_create_cfg() -> ExperimentConfig:
    if CONFIG_PATH.exists():
        try:
            cfg_dict = yaml.safe_load(CONFIG_PATH.read_text())
            return ExperimentConfig(**cfg_dict)
        except Exception as err:  # pragma: no cover – config corruption
            raise RuntimeError(f"Failed to parse configuration: {err}") from err
    # ------------------------------------------------------------------
    cfg = ExperimentConfig()
    CONFIG_PATH.write_text(cfg.as_yaml())
    print(f"Configuration written to {CONFIG_PATH.resolve()}")
    return cfg

###############################################################################
# Pipeline – mirrors control-flow from the original monolithic script
###############################################################################

def main() -> None:  # pragma: no cover (run via `python -m src.main`)
    cfg = _load_or_create_cfg()
    ensure_directories(cfg)

    # Dataset acquisition ----------------------------------------------------
    archive_path = download_dataset(cfg)
    dataset_root = extract_dataset(cfg, archive_path)

    # Sequentially execute the three experiments ----------------------------
    experiments: list[tuple[str, Any]] = [
        ("experiment1_results.json", run_experiment_1),
        ("experiment2_results.json", run_experiment_2),
        ("experiment3_results.json", run_experiment_3),
    ]

    for json_name, fn in experiments:
        json_path = Path(cfg.output_root) / json_name
        print("\n============================================================")
        print(f"Running {fn.__name__} – results will be saved to {json_path}")
        print("============================================================\n")
        try:
            results: Dict[str, Any] = fn(dataset_root)
        except RuntimeError as err:
            # Immediate termination as per STRICT NO-FALLBACK RULE
            sys.stderr.write(str(err) + "\n")
            sys.exit(1)

        json_path.write_text(json.dumps(results, indent=2))

        # Echo JSON to stdout for verification ------------------------------
        print("Experiment description:")
        print(fn.__doc__ or "<no description>")
        print("\nExperimental numerical data:")
        print(json.dumps(results, indent=2))
        print("\nNames of figures summarising the numerical data:")
        for key in sorted(results.keys()):
            if key.endswith("_figure"):
                print(results[key])

if __name__ == "__main__":
    main()
