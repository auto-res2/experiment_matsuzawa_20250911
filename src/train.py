"""src/train.py
-------------------------------------------------------------------------
Training-related logic for the HydraSketch-Φ experiments.
Because the real mixed-signal hardware and the 320 GB EdgeBench-48 dataset
are unavailable in a generic execution environment, the full training
routine intentionally aborts (STRICT NO-FALLBACK).  This module therefore
only contains the configuration dataclasses and a placeholder
`run_full_training` function which will be invoked from `src/main.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------
# 1.  Dataclass helpers
# ---------------------------------------------------------------------
try:
    import yaml  # PyYAML is required to parse the config file
except ImportError as e:  # pragma: no cover – fail fast if missing
    raise RuntimeError(
        "Required dependency ‘pyyaml’ missing – please install it in the execution environment."
    ) from e


@dataclass
class DatasetConfig:
    name: str
    version: str
    url: str
    sha256: str


@dataclass
class ExperimentConfig:
    name: str
    seeds: List[int]
    ram_caps_mb: List[float]
    latency_caps_ms: List[int]
    dataset: DatasetConfig

    # yaml-reader helper (used from src/main.py)
    @staticmethod
    def from_yaml(path: Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        ds = raw["experiment"]["dataset"]
        dataset_cfg = DatasetConfig(**ds)
        cfg = ExperimentConfig(
            name=raw["experiment"]["name"],
            seeds=raw["experiment"]["seeds"],
            ram_caps_mb=raw["experiment"]["ram_caps_mb"],
            latency_caps_ms=raw["experiment"]["latency_caps_ms"],
            dataset=dataset_cfg,
        )
        # Pretty-print for human verification
        print("\nLoaded configuration:\n" + json.dumps(asdict(cfg), indent=2), flush=True)
        return cfg


# ---------------------------------------------------------------------
# 2.  Full training pipeline placeholder
# ---------------------------------------------------------------------

def run_full_training(cfg: ExperimentConfig, data_root: Path) -> None:  # noqa: D401
    """Launch the full HydraSketch-Φ training-and-evaluation pipeline.

    The real implementation integrates:
      • Fractional-SDE retention on mixed-signal CDSC & InP-PCM memory;
      • Temperature-aware cell migration executed on 20× GAP-9 SoCs;
      • Online latency control & energy / thermo logging.

    All of those components *require* physical edge hardware plus the
    320 GB EdgeBench-48 dataset.  In compliance with the project’s
    STRICT NO-FALLBACK rule we abort when the necessary resources are
    not present.
    """
    raise RuntimeError(
        "Full-scale HydraSketch-Φ experiment requires physical edge hardware, "
        "thermal chamber and the EdgeBench-48 dataset (~320 GB).  Execution is "
        "aborted because the necessary resources are not present in the current "
        "environment (STRICT NO-FALLBACK RULE)."
    )
