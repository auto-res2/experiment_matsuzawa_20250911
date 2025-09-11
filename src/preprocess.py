# src/preprocess.py
"""Dataset download & preprocessing utilities.
Downloads public datasets from the HuggingFace Hub into ./data/ and raises
DatasetNotFound if a download fails – complying with the *strict no-fallback*
rule in the policy.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import datasets as hf_datasets  # Ensure module is available for type hints
from datasets import load_dataset, DatasetDict, Dataset  # huggingface-datasets package

__all__ = ["ensure_dataset", "DatasetNotFound"]

DATA_ROOT = pathlib.Path("data").resolve()
DATA_ROOT.mkdir(exist_ok=True)


class DatasetNotFound(RuntimeError):
    """Raised when a dataset could not be retrieved from the HF hub."""


def _extract_cache_folder(ds: Dataset) -> pathlib.Path:
    """Return the physical cache folder that stores *ds* on disk.

    Parameters
    ----------
    ds : datasets.Dataset
        A single split of a HuggingFace dataset.
    """
    if not getattr(ds, "cache_files", None):
        raise RuntimeError(
            "Dataset object has no cache_files attribute – cannot locate files on disk."
        )
    return pathlib.Path(ds.cache_files[0]["filename"]).parent


def ensure_dataset(repo: str, *, split: str | None = None) -> pathlib.Path:
    """Download `repo` from the HuggingFace hub and return the local folder.

    Parameters
    ----------
    repo: str
        e.g. "ag_news" or "SocialGrep/one-year-of-tsla-on-reddit".
    split: Optional[str]
        Passed on to `datasets.load_dataset`.  None → the builder decides.
    """
    try:
        ds = load_dataset(
            repo,
            split=split,
            cache_dir=str(DATA_ROOT / repo.replace("/", "_")),
        )

        # `load_dataset` returns either a Dataset (when split is given) or
        # a DatasetDict.  We need a *single* split to locate the cache folder.
        if isinstance(ds, DatasetDict):
            # Take the first available split deterministically.
            first_split = ds[next(iter(ds))]
        else:
            first_split = ds  # already a Dataset

        return _extract_cache_folder(first_split)
    except Exception as exc:  # broad – we want to signal any failure clearly
        raise DatasetNotFound(
            f"Could not download dataset '{repo}'. Obtain the files manually "
            f"and place them under {DATA_ROOT} to proceed.  Original error: {exc}"
        ) from exc
