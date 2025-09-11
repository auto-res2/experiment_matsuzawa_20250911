"""
src/preprocess.py
=================
Dataset download & integrity verification utilities.  This is a direct lift
from the original script with added error handling and Path import fixes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

__all__ = ["DatasetDownloadError", "download"]


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or fails checksum test."""


# ---------------------------------------------------------------------------
# Helper – SHA-256 checksum
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API – download with optional checksum verification
# ---------------------------------------------------------------------------

def download(url: str, tgt: Path, sha256: Optional[str] = None) -> Path:
    try:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.exists():
            if sha256 and _sha256(tgt) == sha256:
                return tgt  # already up-to-date
            tgt.unlink()  # checksum mismatch → force re-download
        print(f"[INFO] Downloading {url} → {tgt} …")
        urlretrieve(url, tgt)
        if sha256 and _sha256(tgt) != sha256:
            raise DatasetDownloadError(f"Checksum mismatch for {url}")
        return tgt
    except Exception as exc:  # pragma: no cover – wrapped anyway
        raise DatasetDownloadError(str(exc)) from exc
