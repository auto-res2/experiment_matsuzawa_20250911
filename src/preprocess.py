"""src/preprocess.py
Dataset acquisition and directory-management utilities extracted from the
original single-file pipeline.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

__all__ = [
    "ensure_directories",
    "download_dataset",
    "extract_dataset",
]

def _root(p: Any) -> Path:
    """Convert attribute or mapping entry to ``Path`` consistently."""
    return p if isinstance(p, Path) else Path(p)

def ensure_directories(cfg: Any) -> None:
    """Create ``data``, ``outputs`` and ``figures`` directories declared in cfg."""
    for attr in ("data_root", "output_root", "figure_root"):
        _root(getattr(cfg, attr)).mkdir(parents=True, exist_ok=True)

def download_dataset(cfg: Any) -> Path:
    """Download EdgeBench-48 archive unless already present.

    STRICT NO-FALLBACK:  any failure during download leads to an exception that
    must be handled by the caller (usually resulting in immediate exit).
    """
    archive_path = _root(cfg.data_root) / cfg.dataset_archive_name
    if archive_path.is_file():
        return archive_path

    import requests
    from tqdm import tqdm

    print(
        f"Dataset archive not found locally – attempting download from\n  {cfg.dataset_url}\n\n(This may take a long time; the file is >300 GB.)"
    )

    try:
        response = requests.get(cfg.dataset_url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    except Exception as exc:  # pragma: no cover – fatal path
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            "Unable to download EdgeBench-48 dataset.  Execution aborted as "
            "dictated by the STRICT NO-FALLBACK RULE.\n"
            f"Original error: {exc}"
        ) from exc

    return archive_path

def extract_dataset(cfg: Any, archive_path: Path) -> Path:
    """Extract dataset archive into a versioned sub-directory."""
    target_dir = _root(cfg.data_root) / f"{cfg.dataset_name}-{cfg.dataset_version}"
    if target_dir.exists():
        return target_dir

    print(f"Extracting dataset to {target_dir} (this may consume >1 TB disk)")
    try:
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=_root(cfg.data_root))
    except Exception as exc:  # pragma: no cover – fatal path
        raise RuntimeError(
            "Failed while extracting EdgeBench-48 dataset – aborting."
        ) from exc

    if not target_dir.exists():  # pragma: no cover – sanity check
        raise RuntimeError(
            "Dataset extraction completed but expected directory is missing."
        )
    return target_dir
