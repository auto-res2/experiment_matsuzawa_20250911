"""src/main.py
Orchestrates the entire experimental pipeline.
Steps:
1. Download/validate datasets.
2. Download foundation models.
3. Attempt to initialise TACO (will raise *NotImplementedError*).

Path updates:
    • All images must reside under `.research/iteration2/images`.
    • All JSON artefacts must reside directly under `.research/iteration2/`.
The constants below have therefore been updated accordingly.
"""
from __future__ import annotations

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
RESULT_DIR = ROOT / ".research" / "iteration2"
FIG_DIR = RESULT_DIR / "images"  # mandatory location per spec
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
        load_all(MODEL_DIR)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Instantiate TACO ----------------------------------------------
    # ------------------------------------------------------------------
    logger.info("Step 3/3: Initialising TACO …")
    try:
        _ = TACOModel()
    except NotImplementedError:
        logger.error(
            "TACO implementation missing. Per STRICT NO-FALLBACK RULE the "
            "program terminates — please supply the full research algorithm."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # The code below will become reachable once TACO is integrated.  It
    # already fulfils the mandatory JSON-saving policy.
    # ------------------------------------------------------------------
    # results = {"status": "success"}
    # RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # (RESULT_DIR / "exp1_results.json").write_text(json.dumps(results, indent=2))
    # print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
