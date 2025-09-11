"""
preprocess.py – data loading & preprocessing utilities
------------------------------------------------------
Contains helper functions that download datasets and cache them under
./data so that future runs are instant.
"""
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Dataset helper
# -----------------------------------------------------------------------------

def ensure_dataset(name: str, cfg: str | None = None):
    """Download a HuggingFace dataset (or load it from disk cache)."""
    try:
        ds = load_dataset(name, cfg, cache_dir=str(DATA_DIR))
    except Exception as e:
        raise RuntimeError(f"[FATAL] Could not load dataset '{name}/{cfg}': {e}")
    return ds
