import pathlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader

# Graph / datasets ----------------------------------------------------------
try:
    from ogb.lg import LGDataset
except ImportError:  # pragma: no cover
    LGDataset = None  # will be checked at call-time

import dgl  # noqa: F401 – needed for some dataset loaders

__all__ = [
    "ensure_ogb",
    "tensor_loader_from_feats_labels",
    "load_imdb_multi",
]

# ---------------------------------------------------------------------------
#  HELPERS ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def ensure_ogb(dataset_name: str):
    if LGDataset is None:
        raise RuntimeError("OGB is not installed.  Please install 'ogb' to continue.")
    try:
        return LGDataset(name=dataset_name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"[Data] Failed to load OGB dataset {dataset_name}: {exc}") from exc


def tensor_loader_from_feats_labels(
    feats: torch.Tensor, labels: torch.Tensor, batch_size: int, *, shuffle: bool = True
) -> DataLoader:
    ds = torch.utils.data.TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
#  IMDB-MULTI EXAMPLE (not used in minimal smoke-test) -----------------------
# ---------------------------------------------------------------------------

def load_imdb_multi() -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from dgl.data import IMDBDataset

        ds = IMDBDataset("movie")
        g = ds[0]
        feats = g.ndata["feat"]
        labels = g.ndata["label"]
        return torch.tensor(feats), torch.tensor(labels)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"IMDB-multi dataset unavailable: {exc}") from exc
