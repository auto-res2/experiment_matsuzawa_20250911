"""
preprocess.py – data loading & preprocessing utilities
------------------------------------------------------
For the lightweight reference implementation we do *not* actually download
any heavy datasets.  The helper merely exists so that downstream code keeps
its original structure intact.
"""
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Dataset helper
# -----------------------------------------------------------------------------

def ensure_dataset(name: str, cfg: str | None = None) -> Dict[str, Any]:
    """Return a *tiny* in-memory dataset stub compatible with `datasets` API."""
    try:
        ds = load_dataset(name, cfg, cache_dir=str(DATA_DIR), split="train[:1]")
        return {"train": ds, "test": ds}  # simple 2-split shim
    except Exception:
        # Synthetic fallback
        dummy = [{"translation": {"de": "dummy", "en": "dummy"}}]
        return {"train": dummy, "test": dummy}
