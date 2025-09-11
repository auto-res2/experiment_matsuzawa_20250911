"""src/preprocess.py
---------------------------------------------------------------------
Dataset download, caching and very light pre-processing utilities.  The
original repository used a dedicated sub-package with SHA-256 checks –
that would be overkill for this demonstration, but we still maintain the
*no-fallback* policy: if a required file is missing, execution stops.
---------------------------------------------------------------------"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import requests


class DatasetMissingError(RuntimeError):
    """Raised when a required dataset cannot be found or downloaded."""


# ------------------------------------------------------------------
# public helper – mimics the original `src.data.download.ensure_all_*`
# ------------------------------------------------------------------

def ensure_all_datasets_exist(conf: Any) -> None:  # noqa: ANN401
    """Verify every dataset entry in *conf.datasets* is present on disk.

    Expected YAML structure::

        datasets:
          - name: ego4d_m
            url:  https://…/ego4d_m.zip
          - name: har_100
            url:  https://…/har_100.tar

    If a file is missing we *attempt* to download it.  Should the HTTP
    request fail, the function aborts the whole program as mandated by
    the strict "no fallback" rule in the assignment.
    """
    data_root = Path(conf.get("data_dir", "data")).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    for entry in conf.get("datasets", []):
        fname = data_root / Path(entry["url"]).name
        if fname.exists():
            continue
        print(f"[preprocess] Dataset {entry['name']} missing – downloading …")
        try:
            _download_file(entry["url"], fname)
        except Exception as exc:  # noqa: BLE001 – convert to hard abort
            print(f"[preprocess] FATAL: could not download {entry['url']}: {exc}", file=sys.stderr)
            raise DatasetMissingError from exc


def _download_file(url: str, target: Path) -> None:
    """Stream *url* to *target* with a basic progress indicator."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                pct = (done / total * 100) if total else 0.0
                sys.stdout.write(f"\r  ↳ {pct:5.1f}%")
                sys.stdout.flush()
    sys.stdout.write("\n")

    # Very small safeguard: delete partially downloaded files on failure
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Download produced an empty file for {url}")

    # Optionally – future-proof: extract archives automatically
    if target.suffix in {".zip", ".tar", ".gz"}:
        print(f"[preprocess] Extracting {target.name} …")
        _extract_archive(target)


def _extract_archive(archive: Path) -> None:
    try:
        shutil.unpack_archive(str(archive), extract_dir=str(archive.parent))
    except (shutil.ReadError, ValueError) as err:
        print(f"[preprocess] Could not extract {archive}: {err}", file=sys.stderr)
        raise
