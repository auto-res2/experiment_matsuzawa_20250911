# src/preprocess.py
"""Dataset loading and preparation layer – no synthetic fallbacks allowed."""
from __future__ import annotations

import sys
import os
import signal
from pathlib import Path

import torch
from datasets import load_dataset, DownloadConfig
from torch_geometric.data import Data, InMemoryDataset

# ---------------------------------------------------------------------------
#  Local abort helper (kept minimal to avoid circular imports)
# ---------------------------------------------------------------------------

def abort(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
#  Partition dataset class
# ---------------------------------------------------------------------------

class FedPartitionDataset(InMemoryDataset):
    """HuggingFace FedGraph partition  →  PyG Data."""

    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self._root = root
        super().__init__(str(root))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # noqa: D401
        return []  # handled by 🤗 datasets

    @property
    def processed_file_names(self):  # noqa: D401
        return ["data.pt"]

    # ------------------------------ pipeline hooks -----------------------
    def download(self):  # noqa: D401
        cfg = DownloadConfig(resume_download=True, use_etag=True, num_proc=4)
        ds = load_dataset(self.repo_id, download_config=cfg)
        g = ds["train"]  # each partition has only a train split
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        x = torch.tensor(g["x"], dtype=torch.float)
        y = torch.tensor(g["y"], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        torch.save(self.collate([data]), self.processed_paths[0])

    def process(self):  # noqa: D401
        # Work done in `download` because HF already gives the arrays.
        pass
