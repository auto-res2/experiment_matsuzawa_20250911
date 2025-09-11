"""src/train.py
Model-related utilities: backbone download, (stub) continual-learning model and a shared logger helper.
Only minimal refactoring has been applied – no new logic introduced.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple, Dict

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

def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Return a configured logger that writes both to console and (optionally)
    to *log_dir*. The function is intentionally placed in *train.py* so that all
    other modules can simply `from .train import get_logger` without creating
    extra files (STRICT FILE CONSTRAINT).
    """
    logger = logging.getLogger(name)

    # Prevent duplicated handlers in interactive sessions / multiple imports
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / f"{name}.log")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:  # permissions, disk full …
            # Do not crash the whole experiment because file logging fails.
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
    Execution terminates with *RuntimeError* if anything fails in order to
    comply with the STRICT NO-FALLBACK RULE.
    """
    logger = get_logger("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, object] = {}
    processors: Dict[str, object] = {}

    # MobileViT-S ---------------------------------------------------------
    logger.info("Loading MobileViT-S (vision backbone)")
    try:
        models["mobilevit"] = AutoModelForImageClassification.from_pretrained(
            _MODELS["mobilevit"], cache_dir=str(model_dir / "mobilevit")
        )
        processors["mobilevit"] = AutoProcessor.from_pretrained(
            _MODELS["mobilevit"], cache_dir=str(model_dir / "mobilevit")
        )
    except Exception as e:
        logger.error("MobileViT could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.")

    # Whisper-Tiny --------------------------------------------------------
    logger.info("Loading Whisper-Tiny (audio backbone)")
    try:
        models["whisper"] = WhisperForConditionalGeneration.from_pretrained(
            _MODELS["whisper"], cache_dir=str(model_dir / "whisper")
        )
        processors["whisper"] = WhisperProcessor.from_pretrained(
            _MODELS["whisper"], cache_dir=str(model_dir / "whisper")
        )
    except Exception as e:
        logger.error("Whisper could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.")

    # DistilBERT ----------------------------------------------------------
    logger.info("Loading DistilBERT-base (text backbone)")
    try:
        models["distilbert"] = DistilBertForMaskedLM.from_pretrained(
            _MODELS["distilbert"], cache_dir=str(model_dir / "distilbert")
        )
        processors["distilbert"] = DistilBertTokenizerFast.from_pretrained(
            _MODELS["distilbert"], cache_dir=str(model_dir / "distilbert")
        )
    except Exception as e:
        logger.error("DistilBERT could not be loaded: %s", e)
        raise RuntimeError("Model download failure – terminating execution.")

    logger.info("All backbone models downloaded successfully")
    return models, processors

# ---------------------------------------------------------------------
# Continual-learning algorithm stub (formerly *taco.py*)
# ---------------------------------------------------------------------

class TACOModel:  # noqa: D101 – documented in docstring below
    """Stub for the proprietary TACO continual learner.

    The actual algorithm is *not* part of the public reference. Any attempt to
    instantiate the class will raise *NotImplementedError* – thereby enforcing
    the STRICT NO-FALLBACK RULE that forbids silent degradation of results.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TACO research implementation is proprietary and not included in this "
            "public reference. Please integrate the full algorithm before use."
        )
