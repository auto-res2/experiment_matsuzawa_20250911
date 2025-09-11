"""src/evaluate.py
Evaluation and analysis helpers.
For this public release the three high-level experiment entry points are kept
verbatim from the original single-file script so that behaviour (i.e. the
immediate termination) remains unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

__all__ = [
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
]

def run_experiment_1(dataset_root: Path) -> Dict[str, Any]:  # pragma: no cover
    """Run Experiment 1 – continual-learning benchmark (stub)."""
    raise RuntimeError(
        "Experiment 1 requires proprietary hardware drivers and the full "
        "EdgeBench-48 dataset.  Execution is halted in accordance with the "
        "STRICT NO-FALLBACK RULE."
    )

def run_experiment_2(dataset_root: Path) -> Dict[str, Any]:  # pragma: no cover
    """Run Experiment 2 – remanence attack analysis (stub)."""
    raise RuntimeError(
        "Experiment 2 (remanence attack) cannot proceed without InP-PCM boards "
        "and nanoprobing microscopy equipment.  Aborting."
    )

def run_experiment_3(dataset_root: Path) -> Dict[str, Any]:  # pragma: no cover
    """Run Experiment 3 – nano-drone closed-loop evaluation (stub)."""
    raise RuntimeError(
        "Experiment 3 (nano-drone closed-loop flight) cannot proceed inside the "
        "current execution environment.  Aborting."
    )
