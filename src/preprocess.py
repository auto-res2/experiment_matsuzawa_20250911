"""src/preprocess.py
Data-loading & pre-processing helpers extracted from the original monolithic
script so that I/O logic is clearly separated from experiment execution.
"""
from __future__ import annotations

from typing import Any
from datasets import load_dataset


# ----------------------------------------------------------------------------
# Machine-translation EN→DE WMT-22 test split
# ----------------------------------------------------------------------------

def load_wmt22_en_de(split: str = "test") -> Any:
    return load_dataset("wmt22", "en-de", split=split)

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
