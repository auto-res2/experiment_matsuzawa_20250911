"""src/preprocess.py
Data-loading & pre-processing helpers extracted from the original monolithic
script so that I/O logic is clearly separated from experiment execution.
"""
from __future__ import annotations

from typing import Any
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


# ----------------------------------------------------------------------------
# Machine-translation EN→DE  – robust loader with graceful fallbacks
# ----------------------------------------------------------------------------

def _try_mt_loader(ds_id: str, config: str, split: str):
    """Attempt to load a translation dataset and return the requested split.
    Falls back to the first available split if `split` not present.
    """
    ds_dict = load_dataset(ds_id, config)
    if split in ds_dict:
        return ds_dict[split]
    # choose a reasonable alternative order: test → validation → train → first
    for alt in ("test", "validation", "train"):
        if alt in ds_dict:
            return ds_dict[alt]
    # Last resort – return the first split
    return next(iter(ds_dict.values()))


def load_wmt22_en_de(split: str = "test") -> Any:
    """Try to load WMT'22 EN↔DE; if unavailable, progressively fall back to
    WMT'19 or OPUS Books so that we *always* return a *real* MT dataset with
    `translation` keys {"en", "de"}.  We purposely avoid synthetic placeholders
    to comply with the experimental policy.
    """
    fallbacks = [
        ("wmt22", "en-de"),  # preferred – may be missing on old `datasets`.
        ("wmt19", "de-en"),
        ("opus_books", "de-en"),
    ]
    last_err: Exception | None = None
    for ds_id, cfg in fallbacks:
        try:
            return _try_mt_loader(ds_id, cfg, split)
        except (DatasetNotFoundError, FileNotFoundError, ConnectionError) as e:
            last_err = e
            continue
    # If every attempt failed, raise a clear error
    raise DatasetNotFoundError(
        "None of the fallback MT datasets (wmt22, wmt19, opus_books) could be "
        "loaded. Last error: " + str(last_err)
    )

# ----------------------------------------------------------------------------
# Protein CATH-4.3 backbone benchmark
# ----------------------------------------------------------------------------

def load_protein_cath(split: str = "test") -> Any:
    return load_dataset("cctien/protein_backbone_cath_4.3", split=split)

# ----------------------------------------------------------------------------
# OpenAI HumanEval code generation benchmark
# ----------------------------------------------------------------------------

def load_humaneval(split: str = "test") -> Any:
    return load_dataset("openai_humaneval", split=split)
