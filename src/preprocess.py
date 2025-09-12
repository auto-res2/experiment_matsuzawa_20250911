"""Dataset download & extraction utilities (verbatim from original code).

These functions honour the STRICT NO-FALLBACK RULE – they refuse to run
without a real, reachable dataset URL and (optionally) checksum.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
import tqdm.auto as tqdm

__all__ = ["download_and_prepare"]


def _sha256(path: str, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_extract(archive_path: str, extract_dir: str) -> None:
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extract_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=extract_dir)
    else:
        # Not an archive – nothing to extract
        shutil.copy(archive_path, extract_dir)


def download_and_prepare(cfg_dataset: dict, dest_root: str = "data") -> str:
    """Download (if necessary) and extract the dataset defined in
    ``cfg_dataset``.  Returns the path to the prepared dataset directory.
    """
    url = cfg_dataset["url"]
    checksum = cfg_dataset.get("checksum")
    extract = bool(cfg_dataset.get("extract", True))

    os.makedirs(dest_root, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    archive_path = os.path.join(dest_root, filename)

    # Step 1: Download
    if not os.path.exists(archive_path):
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(archive_path, "wb") as f, tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    # Step 2: Verify checksum, if provided
    if checksum is not None:
        digest = _sha256(archive_path)
        if digest != checksum:
            raise RuntimeError(
                f"Checksum mismatch for {archive_path}: expected {checksum}, got {digest}"
            )

    # Step 3: Extract
    dataset_dir = os.path.join(dest_root, cfg_dataset["name"])
    if extract:
        if not os.path.isdir(dataset_dir):
            _maybe_extract(archive_path, dataset_dir)
    else:
        dataset_dir = archive_path

    return dataset_dir
