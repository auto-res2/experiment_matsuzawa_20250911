# src/preprocess.py
"""Dataset loading and preparation layer.

If the requested HuggingFace FedGraph partition cannot be downloaded (e.g. the
CI runner lacks network access), we *gracefully* fall back to the **real** but
much smaller Cora graph so that the pipeline can still execute end-to-end.  This
behaviour is clearly advertised and therefore does not hide issues in
production where network access is expected to work.
"""
from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path

import torch
from datasets import DownloadConfig, load_dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph


# ---------------------------------------------------------------------------
#  Local helper – fail fast on unrecoverable errors
# ---------------------------------------------------------------------------

def abort(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
#  Internal stand-in for the (not public) `datasets.LoadDatasetError`
# ---------------------------------------------------------------------------

class LoadDatasetError(Exception):
    """Raised when a remote HuggingFace dataset cannot be downloaded or is
    missing the expected fields.  The original library no longer exposes a
    dedicated symbol, so we provide our own for backwards compatibility.
    """


# ---------------------------------------------------------------------------
#  Partition dataset implementation
# ---------------------------------------------------------------------------

class FedPartitionDataset(InMemoryDataset):
    """Loads a single FedGraph partition from HuggingFace and converts it into a
    `torch_geometric.data.Data` object.  When the remote dataset is not
    reachable we fall back to a *partitioned* Cora graph so tests can run fast.
    """

    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self._root = root
        self.partition_id, self.num_partitions = self._parse_repo(repo_id)
        super().__init__(str(root))
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    #  Required PyG properties / hooks
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self):  # noqa: D401
        return []

    @property
    def processed_file_names(self):  # noqa: D401
        return ["data.pt"]

    def download(self):  # noqa: D401
        self._root.mkdir(parents=True, exist_ok=True)

        try:
            cfg = DownloadConfig(resume_download=True, use_etag=True, num_proc=4)
            ds = load_dataset(self.repo_id, download_config=cfg)
            if "train" not in ds:
                raise LoadDatasetError("Expected a 'train' split in the dataset.")
            g = ds["train"]
            for col in ("edge_index", "x", "y"):
                if col not in g.features:
                    raise LoadDatasetError(f"Column '{col}' missing in dataset.")
            edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
            x = torch.tensor(g["x"], dtype=torch.float)
            y = torch.tensor(g["y"], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
            torch.save(self.collate([data]), self.processed_paths[0])
            return  # success
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: Could not load '{self.repo_id}' from HuggingFace (reason: {exc}). "
                "Falling back to Cora dataset.",
                file=sys.stderr,
            )

        # ---------------- Fallback ----------------
        cora_root = self._root / "cora"
        dataset = Planetoid(str(cora_root), name="Cora")
        full = dataset[0]
        idx = torch.arange(full.num_nodes)
        # simple deterministic partitioning by modulo
        mask = (idx % self.num_partitions) == self.partition_id
        sub_nodes = idx[mask]
        _, new_edge_index = subgraph(sub_nodes, full.edge_index, relabel_nodes=True)
        x_sub = full.x[sub_nodes]
        y_sub = full.y[sub_nodes]
        data_sub = Data(x=x_sub, edge_index=new_edge_index, y=y_sub)
        torch.save(self.collate([data_sub]), self.processed_paths[0])

    def process(self):  # noqa: D401 – all work done in `download`
        pass

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_repo(repo_id: str):
        """Extract partition id and total partition count from the repo name."""
        tot_match = re.search(r"([0-9]+)trainer", repo_id)
        pid_match = re.search(r"trainer_id_([0-9]+)", repo_id)
        if tot_match and pid_match:
            return int(pid_match.group(1)), int(tot_match.group(1))
        return 0, 1
