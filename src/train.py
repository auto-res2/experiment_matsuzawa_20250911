"""
train.py – model-centric utilities (CERTIFLOW reference repo)
-------------------------------------------------------------
This file is intentionally lean because the public CERTIFLOW
experiments do not actually *train* new networks – they only
load pre-trained checkpoints, run accelerated inference and
collect statistics.  Still, we keep model helpers here so that
all model–related functionality lives in a single place.
"""
from pathlib import Path

import torch
from transformers import AutoModel

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def ensure_model(model_id: str):
    """Download (or load from cache) a HuggingFace model checkpoint.
    Returns the *local* identifier that downstream code can hand to HF APIs.
    The helper lives in train.py because it is the only piece that directly
    touches neural-network weights.
    """
    try:
        _ = AutoModel.from_pretrained(model_id, cache_dir=str(CACHE_DIR),
                                      trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"[FATAL] Could not load model '{model_id}': {e}")
    return model_id

# -----------------------------------------------------------------------------
# Device helper (small but handy)
# -----------------------------------------------------------------------------

def get_best_device() -> torch.device:
    """Return CUDA if available, otherwise CPU.  Call this once and reuse."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
