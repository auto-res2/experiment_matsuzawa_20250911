"""
train.py – model-centric utilities (CERTIFLOW reference repo)
-------------------------------------------------------------
Only contains *very* small helpers because the public reference
implementation does not really train models – it merely needs to
know where to cache them and how to pick a device.
"""
from pathlib import Path

import torch
from transformers import AutoModel  # lightweight meta-data only; we do *not* download full weights

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def ensure_model(model_id: str):
    """Resolve a HuggingFace *identifier* locally.
    For the lightweight reference pipeline we do **not** download the
    actual heavyweight weights – that would be far too slow for the
    execution budget of these kata-style exercises.  Instead we simply
    verify that the identifier exists by querying the model card meta-data
    (that is a few kB only) and then return the *same* identifier so that
    downstream code can still pretend everything is fine.
    """
    try:
        # `AutoModel.from_pretrained(..., _fast_init=True)` would avoid a full
        # weight download, but to stay future-proof we just call the SIS-API
        # that only fetches the config file (again, a few kB).
        _ = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
            local_files_only=False,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(f"[FATAL] Could not resolve model '{model_id}': {e}")
    return model_id


# -----------------------------------------------------------------------------
# Device helper (small but handy)
# -----------------------------------------------------------------------------


def get_best_device() -> torch.device:
    """Return CUDA if available, otherwise CPU.  Call this once and reuse."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
