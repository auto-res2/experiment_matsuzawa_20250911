"""
src/main.py
===========
Experiment entry point.

The pipeline does three things:
1. *Pre-processing*: download the HumanEval dataset.
2. *Training/statistics*: compute descriptive statistics over prompts.
3. *Evaluation*: visualise prompt length distribution + write JSON
   results under the mandatory ``.research/iteration4/`` directory.

All paths, prints and outputs follow the strict rules given in the task
instructions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

from preprocess import load_data
from train import run as run_stats
from evaluate import save_line_plot

# ---------------------------------------------------------------------------
# Constants – centralise directories so that tweaks are easy.
# ---------------------------------------------------------------------------
_ITER_DIR = Path(".research/iteration4")
_ITER_DIR.mkdir(parents=True, exist_ok=True)
_IMG_DIR = _ITER_DIR / "images"  # Must match evaluate.save_line_plot.
_IMG_DIR.mkdir(parents=True, exist_ok=True)


def _run_experiment() -> Dict[str, float]:
    """Execute the **entire** mini-pipeline and return the results dict."""
    # 1. Pre-processing – load dataset (≈164 rows; negligible time).
    dataset = load_data()

    # 2. "Training" – compute prompt statistics.
    metrics = run_stats(dataset)

    # 3. Evaluation – generate a line plot of the first 30 prompt lengths.
    first_n = 30
    prompt_lengths = [len(row["prompt"]) for row in dataset.select(range(first_n))]
    xs = list(range(1, first_n + 1))
    ys = prompt_lengths
    fig_path = save_line_plot(
        xs=xs,
        ys=ys,
        xlabel="Sample index",
        ylabel="Prompt length (chars)",
        title="HumanEval prompt lengths (first 30 samples)",
        filename="prompt_lengths_first30.pdf",
    )

    # Attach figure path so it is included in the JSON artefact.
    metrics["plot_path"] = fig_path

    return metrics


if __name__ == "__main__":
    try:
        results = _run_experiment()
    except Exception as exc:
        # Fail-fast policy: print the error and exit with non-zero status.
        print(f"Experiment failed: {exc}", file=sys.stderr)
        raise

    # Persist results as JSON inside the mandatory directory.
    json_path = Path(".research/iteration4/experiment_metrics.json")
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, sort_keys=True)

    # Print JSON to stdout for verification (as per instructions).
    print(json.dumps(results, indent=2, sort_keys=True))
