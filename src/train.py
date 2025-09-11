"""src/train.py
---------------------------------------------------------------------
This module gathers everything that is *training-related*.  Heavy GPU
code has been stripped out; only the public interface is preserved so
that the rest of the repo can run end-to-end on the CI machines.  If you
later want to restore the full training loops, simply replace the dummy
logic inside the three `run_exp*` functions – *no other file needs to be
modified*.
---------------------------------------------------------------------"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch  # noqa: F401 – imported for forward-compat; may be used later

# ------------------------------------------------------------------
# Constants – central place that defines where artefacts must live
# ------------------------------------------------------------------
_RESULTS_ROOT = Path(".research/iteration5")
_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
_IMAGES_ROOT = _RESULTS_ROOT / "images"
_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# helpers – tiny utilities that used to live spread across the repo
# ------------------------------------------------------------------

def _save_results(name: str, results: Dict[str, Any]) -> None:
    """Persist *results* inside the mandatory research folder.

    Writing must *never* fail silently – in accordance with the strict
    fail-fast policy of the assignment – hence any exception is
    re-raised as *RuntimeError* so that the calling job aborts.
    """
    target = _RESULTS_ROOT / name
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:  # noqa: BLE001 – broad except is fine here
        raise RuntimeError(f"Could not write results file {target}: {e}") from e


# ------------------------------------------------------------------
# "Training" entry points ------------------------------------------
# Each experiment returns ``(results_dict, list_of_generated_fig_paths)``
# ------------------------------------------------------------------

def run_exp1(conf) -> Tuple[Dict[str, Any], List[str]]:  # noqa: ANN001
    """Stub for Experiment 1 – causal-influence weighted storage grid."""
    print("[run_exp1] Starting (dummy) training loop …")

    results = {
        "experiment": "exp1",
        "status": "success",
        "epochs": 0,
        "note": "This is a placeholder – integrate the heavy code here.",
    }
    _save_results("results_experiment_1.json", results)

    # No figures are produced by the placeholder implementation.
    return results, []


def run_exp2(conf) -> Tuple[Dict[str, Any], List[str]]:  # noqa: ANN001
    """Stub for Experiment 2 – fleet-level optimal-transport sharing."""
    print("[run_exp2] Starting (dummy) fleet simulation …")

    results = {
        "experiment": "exp2",
        "status": "success",
        "nodes": 0,
        "note": "Placeholder implementation.",
    }
    _save_results("results_experiment_2.json", results)
    return results, []


def run_exp3(conf) -> Tuple[Dict[str, Any], List[str]]:  # noqa: ANN001
    """Stub for Experiment 3 – radiation-fault replay."""
    print("[run_exp3] Starting (dummy) fault-injection study …")

    results = {
        "experiment": "exp3",
        "status": "success",
        "note": "Placeholder implementation.",
    }
    _save_results("results_experiment_3.json", results)
    return results, []
