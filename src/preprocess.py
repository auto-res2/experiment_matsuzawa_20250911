# src/preprocess.py
# -------------------------------------------------------------
# Data acquisition, hardware checks and dataset stubs.
# -------------------------------------------------------------
from __future__ import annotations

import os
import sys
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm

# ---------------------------
# CONSTANTS & DIRECTORIES
# ---------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_ARCHIVE = DATA_DIR / "edgebench-48-v1.2.tar"
EDGE_BENCH_URL = "https://edgebench.org/datasets/edgebench-48-v1.2.tar"

# Make sure base directories exist so that CI does not fail on missing dirs.
for _p in (DATA_DIR,):
    _p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# HELPER: Abort with descriptive message (NO FALLBACK)
# ------------------------------------------------------------

def abort(msg: str) -> None:  # noqa: D401 – imperative style
    """Terminate execution immediately with a spec-compliant message."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------
# DATACLASS: YAML-serialisable configuration
# ------------------------------------------------------------


@dataclass
class ExperimentConfig:
    # Dataset
    dataset_url: str = EDGE_BENCH_URL
    dataset_archive: str = str(RAW_DATA_ARCHIVE)
    dataset_root: str = str(DATA_DIR / "edgebench-48")

    # Models
    vision_backbone: str = "microsoft/resnet-18"
    radar_backbone: str = "pointnet"
    gas_radiation_backbone: str = "gru128"
    ecg_backbone: str = "resnet1d-10"

    # Hyper-parameters
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 300

    # Retention / policy parameters
    alpha: float = 0.5
    teleport_delta_bits: float = 1.0
    latency_deadline_ms: int = 30
    ram_cap_mb: float = 0.05

    # Misc
    num_workers: int = 8
    seeds: List[int] = (0, 1, 2)


# Persist default YAML so users can edit externally.
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CONFIG_PATH.exists():
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(ExperimentConfig()), fh, sort_keys=False)


# ------------------------------------------------------------
# HARDWARE CHECKS
# ------------------------------------------------------------

def check_hardware_available() -> None:
    """Verify that EDGE_HARDWARE_AVAILABLE=1 is set."""
    if os.environ.get("EDGE_HARDWARE_AVAILABLE", "0") != "1":
        abort("Mixed-signal edge hardware not detected – set EDGE_HARDWARE_AVAILABLE=1 once hardware is attached.")


# ------------------------------------------------------------
# DATA ACQUISITION
# ------------------------------------------------------------

def download_edgebench(cfg: ExperimentConfig) -> None:
    """Download & extract EdgeBench-48 if absent."""

    import requests  # local import avoids mandatory dep in dry runs

    archive_path = Path(cfg.dataset_archive)
    dataset_root = Path(cfg.dataset_root)

    if dataset_root.exists() and any(dataset_root.iterdir()):
        print("EdgeBench-48 already present – skipping download.")
        return

    if not archive_path.exists():
        print("Downloading EdgeBench-48 (≈320 GB)…")
        try:
            with requests.get(cfg.dataset_url, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    abort(f"Failed to download EdgeBench-48 (HTTP {r.status_code}).")
                total = int(r.headers.get("content-length", 0))
                chunk_sz = 1 << 20  # 1 MiB
                with open(archive_path, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_sz):
                        if chunk:  # filter keep-alive
                            fh.write(chunk)
                            pbar.update(len(chunk))
        except Exception as e:  # noqa: BLE001
            abort(f"Download failed: {e}")

    # Extract
    print("Extracting EdgeBench-48…")
    try:
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=DATA_DIR)
    except tarfile.TarError as e:
        abort(f"Extraction failed: {e}")

    if not dataset_root.exists():
        abort("Extraction finished but dataset root missing – archive corrupted?")


# ------------------------------------------------------------
# DATASET PLACEHOLDER (Strict NO-FALLBACK)
# ------------------------------------------------------------
from torch.utils.data import Dataset  # local import to keep heavy deps optional


class EdgeBenchPlaceholder(Dataset):
    """Stub that aborts – loading dummy data is forbidden."""

    def __init__(self, *_args, **_kwargs):
        abort("EdgeBench-48 parser not implemented – real dataset required.")

    def __len__(self) -> int:  # pragma: no cover – never executed
        return 0

    def __getitem__(self, _idx):  # pragma: no cover – never executed
        return {}
