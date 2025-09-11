"""src/preprocess.py
-------------------------------------------------------------------------
Dataset acquisition and preprocessing utilities.  Handles secure download
with SHA-256 verification and extraction of the EdgeBench-48 archive.

In a CI environment we **do not** download the 320 GB real dataset.  When
`SKIP_DATA_DOWNLOAD=1` is set (the default), the helper functions become
no-ops so that the pipeline terminates quickly.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path
from typing import Final

try:
    import requests  # external HTTP client
except ImportError as e:  # pragma: no cover – fail fast if missing
    raise RuntimeError(
        "Required dependency ‘requests’ missing – please install it in the execution environment."
    ) from e

try:
    from tqdm import tqdm  # progress bar
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Required dependency ‘tqdm’ missing – please install it in the execution environment."
    ) from e

__all__: Final[list[str]] = [
    "download_file",
    "extract_tar",
]


# ---------------------------------------------------------------------
# Helper flags – CI always sets SKIP_DATA_DOWNLOAD=1
# ---------------------------------------------------------------------
_SKIP = os.getenv("SKIP_DATA_DOWNLOAD", "1") == "1"


def download_file(url: str, dest: Path, expected_sha256: str, chunk_size: int = 1 << 20) -> None:  # noqa: D401
    """Download *url* to *dest* verifying the SHA-256 digest.

    When `_SKIP` is *True* the function returns immediately so that no
    network traffic is generated inside the sandbox.
    """

    if _SKIP:
        print("[INFO] download_file() skipped in CI environment.", flush=True)
        return

    print(f"[INFO] Downloading dataset from {url} …", flush=True)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="EdgeBench-48") as pbar:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
    except requests.RequestException as e:  # pragma: no cover – network/downlink issues
        raise RuntimeError("Dataset download failed – network unavailable or URL invalid.") from e

    # ---------------- SHA-256 verification --------------------------
    sha = hashlib.sha256()
    with open(dest, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            sha.update(blk)
    digest = sha.hexdigest().lower()
    if digest != expected_sha256.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            "SHA-256 mismatch for dataset download.\n"
            f"Expected: {expected_sha256}\nActual:   {digest}\n"
            "Download aborted – integrity compromised."
        )
    print("[INFO] Dataset archive downloaded & verified.")


def extract_tar(archive: Path, target_dir: Path) -> None:  # noqa: D401
    """Extract *.tar.gz* archive to *target_dir* with basic error handling.

    In CI mode (`_SKIP` == True) this is a no-op.
    """

    if _SKIP:
        print("[INFO] extract_tar() skipped in CI environment.", flush=True)
        return

    print(f"[INFO] Extracting {archive} to {target_dir} …", flush=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=target_dir)
    except (tarfile.TarError, EOFError) as e:  # pragma: no cover – corrupt file or disk full
        raise RuntimeError(
            "Failed to extract EdgeBench-48 archive – file corrupted or insufficient disk space."
        ) from e
    print("[INFO] Extraction complete.")
