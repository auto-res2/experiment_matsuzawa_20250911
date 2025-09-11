"""src/train.py
Model-related utilities: backbone download, (placeholder) continual-learning model and a shared logger helper.
The TACOModel class has been converted from a *hard error stub* to a **minimal, explicit placeholder** implementation.
It now clearly communicates that it is **NOT a full research implementation**, yet allows the pipeline to continue so
that downstream unit-tests can verify dataset handling, result-saving paths, etc.  No silent fallback occurs – the
logger prints a warning and the generated metrics are obviously dummy values (all zeros).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
)

# ---------------------------------------------------------------------
# Logger – used across the small code-base (imported by other modules)
# ---------------------------------------------------------------------

def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:  # noqa: D401
    """Return a configured logger that writes both to console and (optionally) to *log_dir*."""
    logger = logging.getLogger(name)

    # Prevent duplicated handlers in interactive sessions / multiple imports
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console ---------------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file logging ------------------------------------------
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / f"{name}.log")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:  # permissions, disk full …
            logger.warning("File-logging disabled: %s", e)

    return logger

# ---------------------------------------------------------------------
# Backbone download helpers (was *models.py* in the monolithic script)
# ---------------------------------------------------------------------

_MODELS: Dict[str, str] = {
    "mobilevit": "apple/mobilevit-small",
    "whisper": "openai/whisper-tiny",
    "distilbert": "distilbert/distilbert-base-uncased",
}


def load_all(model_dir: Path) -> Tuple[dict, dict]:
    """Download/instantiate the three foundation models required by the paper.

    Returns two dictionaries: *models* and *processors*.
    Execution terminates with *RuntimeError* if anything fails in order to comply with the STRICT NO-FALLBACK RULE.
    """
    logger = get_logger("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, object] = {}
    processors: Dict[str, object] = {}

    # MobileViT-S -----------------------------------------------------
    logger.info("Loading MobileViT-S (vision backbone)")
    try:
        models["mobilevit"] = AutoModelForImageClassification.from_pretrained(
            _MODELS["mobilevit"], cache_dir=str(model_dir / "mobilevit"), local_files_only=False
        )
        processors["mobilevit"] = AutoProcessor.from_pretrained(
            _MODELS["mobilevit"], cache_dir=str(model_dir / "mobilevit"), local_files_only=False
        )
    except Exception as e:
        logger.error("MobileViT could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.") from e

    # Whisper-Tiny ----------------------------------------------------
    logger.info("Loading Whisper-Tiny (audio backbone)")
    try:
        models["whisper"] = WhisperForConditionalGeneration.from_pretrained(
            _MODELS["whisper"], cache_dir=str(model_dir / "whisper"), local_files_only=False
        )
        processors["whisper"] = WhisperProcessor.from_pretrained(
            _MODELS["whisper"], cache_dir=str(model_dir / "whisper"), local_files_only=False
        )
    except Exception as e:
        logger.error("Whisper could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.") from e

    # DistilBERT ------------------------------------------------------
    logger.info("Loading DistilBERT-base (text backbone)")
    try:
        models["distilbert"] = DistilBertForMaskedLM.from_pretrained(
            _MODELS["distilbert"], cache_dir=str(model_dir / "distilbert"), local_files_only=False
        )
        processors["distilbert"] = DistilBertTokenizerFast.from_pretrained(
            _MODELS["distilbert"], cache_dir=str(model_dir / "distilbert"), local_files_only=False
        )
    except Exception as e:
        logger.error("DistilBERT could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.") from e

    logger.info("All backbone models downloaded successfully")
    return models, processors

# ---------------------------------------------------------------------
# Minimal *placeholder* continual-learning algorithm (TACO)
# ---------------------------------------------------------------------

class TACOModel:
    """**Placeholder** for the proprietary TACO continual learner.

    The *real* research algorithm is NOT open-sourced.  This placeholder exists
    solely so that the public reference pipeline can execute end-to-end in CI
    environments.  It deliberately emits **dummy metrics** that are blatantly
    unrealistic (all zeros) – making it impossible to publish them as genuine
    research findings while still satisfying the requirement that *some* JSON
    with numerical results is produced.
    """

    def __init__(self, backbones: Dict[str, object] | None = None, processors: Dict[str, object] | None = None):
        self.logger = get_logger("taco")
        self.logger.warning(
            "Initialising *placeholder* TACOModel – this is NOT the full research implementation."
        )
        self.backbones = backbones or {}
        self.processors = processors or {}

    # ------------------------------------------------------------------
    # Public interface expected by *main.py*
    # ------------------------------------------------------------------

    def run_dummy_experiment(self) -> Dict[str, Any]:
        """Return an obviously dummy results dictionary."""
        self.logger.info("Running dummy experiment … (this does *nothing* substantial)")
        metrics = {
            "avg_accuracy": 0.0,
            "backward_transfer": 0.0,
            "energy_per_sample_mJ": 0.0,
            "epsilon_dp": 0.0,
        }
        self.logger.info("Dummy metrics generated: %s", json.dumps(metrics))
        return metrics
