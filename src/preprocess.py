# src/preprocess.py
"""Data download & preparation helpers – independent of model code so they can
be reused by multiple experiments.
"""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  ERRORS
# -----------------------------------------------------------------------------
class DataUnavailableError(RuntimeError):
    """Raised when a dataset cannot be downloaded or verified."""


class ModelUnavailableError(RuntimeError):
    """Placeholder for future use (e.g. missing checkpoints)."""


# -----------------------------------------------------------------------------
#  DIRECTORIES
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  INTERNAL UTILITIES
# -----------------------------------------------------------------------------
CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def http_download(url: str, target: Path, expected_sha256: str | None = None) -> None:
    """Download *url* into *target* with resume support and optional SHA-256
    verification.
    """
    tmp = target.with_suffix(".part")
    headers: Dict[str, str] = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        if r.status_code >= 400:
            raise DataUnavailableError(f"HTTP {r.status_code}: {url}")
        total = int(r.headers.get("Content-Length", 0)) + tmp.stat().st_size if headers else int(r.headers.get("Content-Length", 0))
        mode = "ab" if headers else "wb"
        with open(tmp, mode) as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Download {target.name}") as bar:
            if headers:
                bar.update(tmp.stat().st_size)
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(target)

    # integrity check (optional)
    if expected_sha256 and sha256sum(target) != expected_sha256:
        raise DataUnavailableError(f"SHA-256 mismatch for {target}")


# -----------------------------------------------------------------------------
#  SAFE TAR EXTRACTION (prevents path traversal attacks)
# -----------------------------------------------------------------------------

def _is_within_directory(directory: Path, target: Path) -> bool:  # noqa: D401
    try:
        target.relative_to(directory)
        return True
    except ValueError:
        return False


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path.resolve()):
            raise DataUnavailableError("Blocked path traversal in tar file")
    tar.extractall(path)


# -----------------------------------------------------------------------------
#  PUBLIC DATASET PREP FUNCTIONS
# -----------------------------------------------------------------------------

def prepare_imagenette(root: Path, cfg: Dict[str, str]) -> None:
    """Download (if necessary) and extract the *Imagenette* dataset.  The
    configuration dictionary *cfg* is expected to contain keys: ``url``,
    ``filename`` and optional ``sha256``.
    """
    root.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / cfg["filename"]

    if not (root / "train").exists():
        http_download(cfg["url"], tar_path, cfg.get("sha256"))
        with tarfile.open(tar_path) as tar:
            _safe_extract(tar, root)
