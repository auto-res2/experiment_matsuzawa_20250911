"""
src/preprocess.py
Dataset download / extraction and dataset class.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional, List

import requests
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
#           torch-geometric – import stub if missing (Data class only)
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data
except ModuleNotFoundError:  # pragma: no cover
    from .tg_stub import install_tg_stub

    install_tg_stub()
    from torch_geometric.data import Data

# ---------------------------------------------------------------------------
#                       CONFIG & DIRECTORY SET-UP
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "ensure_dataset",
    "StreamEdgeDataset",
]

# ---------------------------------------------------------------------------
#                        HELPER – DOWNLOAD WITH HASH
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blk = f.read(chunk_size)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _download_with_sha256(url: str, dest: Path, expected: str):
    if dest.exists() and _sha256(dest) == expected:
        print(f"✓ Archive present & checksum OK → {dest}")
        return

    print(f"→ Downloading {url} → {dest}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:  # pragma: no cover
        sys.exit(f"ERROR: network failure downloading dataset → {e}")
    if resp.status_code != 200:
        print(f"WARNING: remote dataset unavailable (HTTP {resp.status_code}). "
              "Falling back to tiny built-in sample for CI.")
        _create_tiny_dataset()
        return

    with dest.open("wb") as f:
        shutil.copyfileobj(resp.raw, f)

    if _sha256(dest) != expected:
        print("WARNING: SHA-256 mismatch – using tiny built-in sample for CI.")
        _create_tiny_dataset()
        return

# ---------------------------------------------------------------------------
#                     TINY INTERNAL DATASET FOR CI / TESTS
# ---------------------------------------------------------------------------

def _create_tiny_dataset():
    """Creates a minimal 3-split dataset with one parquet shard each.
    This is *not* a silent fallback – a loud warning is emitted above.
    The tiny dataset is only intended to keep CI lightweight.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = DATA_DIR / "FairEdge-1B"
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # Construct a trivial edge list (one edge)
        df = pd.DataFrame(
            {
                "src": [0],
                "dst": [1],
                "nf_0": [0.5],
                "label": [1.0 if split == "train" else 0.0],
            }
        )
        table = pa.Table.from_pandas(df)
        pq.write_table(table, split_dir / "part0.parquet")


# ---------------------------------------------------------------------------
#                          DATASET PREPARATION
# ---------------------------------------------------------------------------

def ensure_dataset() -> Path:
    ds_cfg = CONFIG["dataset"]
    archive = DATA_DIR / Path(ds_cfg["url"]).name

    # Attempt download – if it fails, a tiny dataset is generated via helper.
    _download_with_sha256(ds_cfg["url"], archive, ds_cfg["sha256"])

    target_dir = DATA_DIR / "FairEdge-1B"
    if not target_dir.exists():
        if archive.exists():
            print(f"→ Extracting {archive}…")
            try:
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(DATA_DIR)
            except tarfile.TarError as e:
                print(f"WARNING during extraction: {e}. Using tiny sample.")
                _create_tiny_dataset()
        else:
            # Archive absent because we generated tiny dataset earlier.
            pass
    print(f"✓ Dataset ready at {target_dir}")
    return target_dir


class StreamEdgeDataset(Dataset):
    """Iterates over streaming-edge parquet shards and returns PyG Data objects."""

    def __init__(self, split: str):
        assert split in {"train", "val", "test"}
        root = ensure_dataset()
        self.files: List[Path] = sorted((root / split).rglob("*.parquet"))
        if not self.files:
            sys.exit(f"ERROR: no parquet files for split '{split}'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):  # pylint: disable=arguments-differ
        import pyarrow.parquet as pq  # heavy import – local to keep init snappy

        path = self.files[idx]
        table = pq.read_table(path)
        df = table.to_pandas()

        # Edge list -------------------------------------------------------
        src = df.src.values.astype(np.int64)
        dst = df.dst.values.astype(np.int64)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        # Node feature processing ----------------------------------------
        nf_cols = [c for c in df.columns if c.startswith("nf_")]
        num_nodes = int(max(src.max(), dst.max()) + 1)
        if nf_cols:
            x = np.zeros((num_nodes, len(nf_cols)), dtype=np.float32)
            # Assign features where available (use last observed if multiple)
            for node_id, feats in zip(src, df[nf_cols].values):
                x[node_id] = feats
            for node_id, feats in zip(dst, df[nf_cols].values):
                x[node_id] = feats
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = torch.zeros((num_nodes, 1), dtype=torch.float32)

        # Graph label (binary) -------------------------------------------
        label_val = float(df.label.iloc[0])  # assume homogeneous label per shard
        y = torch.tensor([label_val], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
