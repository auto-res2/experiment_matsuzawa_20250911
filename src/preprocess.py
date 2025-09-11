"""
src/preprocess.py
=================
Download / load the evaluation dataset.  We rely on the HuggingFace
``datasets`` library because it is lightweight and the *HumanEval* split
(≈200 KB) is tiny, so the download easily fits within the resource and
network constraints of the execution sandbox.
"""
from __future__ import annotations

from datasets import load_dataset
from typing import Any

_DATASET_ID = "openai_humaneval"
_SPLIT = "test"


def load_data() -> Any:
    """Load the HumanEval *test* split and return it as a Dataset object."""
    # `trust_remote_code` has been removed from the datasets API as of v4.0.
    # The HumanEval dataset is now packaged as standard metadata, so a plain
    # call without the flag suffices.
    ds = load_dataset(_DATASET_ID, split=_SPLIT)
    return ds
