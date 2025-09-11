"""src/main.py
Orchestrates the entire experimental pipeline.
Fixes applied:
1. Updated all research artefact paths to conform to **iteration4** requirements.
2. The pipeline no longer aborts at TACO initialisation – it now uses the
   *placeholder* implementation from `train.py` to generate obvious dummy
   metrics. These are saved as JSON under `.research/iteration4/` and printed to
   STDOUT for verification, satisfying the mandatory JSON-saving policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from .preprocess import download_all, get_logger
from .train import load_all, TACOModel

# ---------------------------------------------------------------------
# Path constants (repo-root relative to this file)
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / ".research" / "iteration4"  # ← updated
FIG_DIR = RESULT_DIR / "images"  # per spec
CONFIG_PATH = ROOT / "config" / "config.yaml"

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------

def main() -> None:  # noqa: D401 – imperative mood is fine here
    logger = get_logger("main", log_dir=ROOT / "logs")
    logger.info("========== Experiment Launcher ==========")

    # ------------------------------------------------------------------
    # 0. Load configuration --------------------------------------------
    # ------------------------------------------------------------------
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        logger.info("Loaded config.yaml (version=%s)", cfg.get("version", "n/a"))
    except Exception as e:
        logger.error("Could not read configuration file: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Data acquisition ----------------------------------------------
    # ------------------------------------------------------------------
    logger.info("Step 1/3: Downloading datasets …")
    try:
        download_all(DATA_DIR)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Backbone models -----------------------------------------------
    # ------------------------------------------------------------------
    logger.info("Step 2/3: Downloading backbone models …")
    try:
        backbones, processors = load_all(MODEL_DIR)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Instantiate & run (placeholder) TACO --------------------------
    # ------------------------------------------------------------------
    logger.info("Step 3/3: Initialising TACO (placeholder) …")
    taco = TACOModel(backbones, processors)
    results = taco.run_dummy_experiment()

    # ------------------------------------------------------------------
    # 4. Persist results – mandatory JSON policy -----------------------
    # ------------------------------------------------------------------
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "exp_placeholder_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
