from __future__ import annotations

import hashlib
import shutil
import urllib.request as urlreq
from pathlib import Path

# -------------------- DIRECTORY CONSTANTS --------------------
ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration5"  # UPDATED TO ITERATION-5
DATA_DIR = ROOT / "data"
RESULTS_DIR = RESEARCH_DIR  # JSON metrics live directly in the iteration folder
FIG_DIR = RESEARCH_DIR / "images"
for _p in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# -------------------- EXCEPTIONS -----------------------------
class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or checksum mismatches."""


# -------------------- HELPERS --------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(name: str, url: str, sha256: str) -> Path:
    """Download a file into DATA_DIR and verify SHA-256 checksum."""
    out_path = DATA_DIR / Path(url).name
    if out_path.exists() and _sha256(out_path) == sha256:
        return out_path

    print(f"Downloading {name} from {url} …", flush=True)
    try:
        with urlreq.urlopen(url) as resp, open(out_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        raise DatasetDownloadError(f"Failed to download {name}: {e}") from e

    if _sha256(out_path) != sha256:
        raise DatasetDownloadError(f"Checksum mismatch for {name}")
    return out_path
