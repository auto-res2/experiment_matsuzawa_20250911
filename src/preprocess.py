# src/preprocess.py
"""Dataset download & preprocessing utilities.
Downloads public datasets from the HuggingFace Hub into ./data/ and raises
DatasetNotFound if a download fails – complying with the *strict no-fallback*
rule in the policy.
"""
from __future__ import annotations

import pathlib
from typing import Optional

from datasets import load_dataset  # huggingface-datasets package

__all__ = ["ensure_dataset", "DatasetNotFound"]

DATA_ROOT = pathlib.Path("data").resolve()
DATA_ROOT.mkdir(exist_ok=True)


class DatasetNotFound(RuntimeError):
    """Raised when a dataset could not be retrieved from the HF hub."""


def ensure_dataset(repo: str, *, split: str | None = None) -> pathlib.Path:
    """Download `repo` from the HuggingFace hub and return the local folder.

    Parameters
    ----------
    repo: str
        e.g. "arkiv/Financial-Fed".
    split: Optional[str]
        Passed on to `datasets.load_dataset`.  None → the builder decides.
    """
    try:
        ds = load_dataset(
            repo,
            split=split,
            cache_dir=str(DATA_ROOT / repo.replace("/", "_")),
        )
        # HuggingFace does not expose the *folder* directly, but every item has
        # a 'filename' entry whose parent is that folder.
        return pathlib.Path(ds.cache_files[0]["filename"]).parent
    except Exception as exc:  # broad – we want to signal any failure clearly
        raise DatasetNotFound(
            f"Could not download dataset '{repo}'. Obtain the files manually "
            f"and place them under {DATA_ROOT} to proceed.  Original error: {exc}"
        ) from exc
