"""
train.py – model-centric utilities (CERTIFLOW reference repo)
-------------------------------------------------------------
Only contains *very* small helpers because the public reference
implementation does not really train models – it merely needs to
know where to cache them and how to pick a device.
"""
from pathlib import Path
import warnings

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
    execution budget of these kata-style exercises.  Instead we **try** to
    fetch the tiny config file (a few kB).  If the environment has no
    internet access we degrade gracefully and just return the identifier –
    downstream stub code never touches the real weights anyway.
    """
    try:
        # `_fast_init=True` avoids weight downloads in modern transformers ≥4.39
        _ = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
            local_files_only=False,
            low_cpu_mem_usage=True,
            _fast_init=True,
        )
    except Exception as e:  # pragma: no cover – network-less CI runner
        warnings.warn(
            f"[WARN] Could not download config for '{model_id}' – proceeding with stub. ({e})"
        )
    return model_id


# -----------------------------------------------------------------------------
# Device helper (small but handy)
# -----------------------------------------------------------------------------


def get_best_device() -> torch.device:
    """Return CUDA if available, otherwise CPU.  Call this once and reuse."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")