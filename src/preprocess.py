import pathlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader

# Graph / datasets ----------------------------------------------------------
try:
    from ogb.lg import LGDataset  # noqa: F401 – optional dependency
except ImportError:  # pragma: no cover
    LGDataset = None  # will be checked at call-time

# ---------------------------------------------------------------------------
#  OPTIONAL DGL IMPORT ------------------------------------------------------
# ---------------------------------------------------------------------------
# DGL is a heavy dependency and not available on all CI systems.  It is only
# required for certain real datasets.  We therefore treat it as *optional* and
# fail fast with descriptive errors if a function actually needs it.
try:
    import dgl  # noqa: F401 – optional, may not exist on all platforms
except ImportError:  # pragma: no cover
    dgl = None

__all__ = [
    "ensure_ogb",
    "tensor_loader_from_feats_labels",
    "load_imdb_multi",
]

# ---------------------------------------------------------------------------
#  HELPERS ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def ensure_ogb(dataset_name: str):
    """Safely load an OGB dataset if the library is available."""
    if LGDataset is None:
        raise RuntimeError("OGB is not installed.  Please install 'ogb' to continue.")
    try:
        return LGDataset(name=dataset_name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"[Data] Failed to load OGB dataset {dataset_name}: {exc}") from exc


def tensor_loader_from_feats_labels(
    feats: torch.Tensor, labels: torch.Tensor, batch_size: int, *, shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader directly from feature and label tensors."""
    ds = torch.utils.data.TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
#  IMDB-MULTI EXAMPLE (not used in minimal smoke-test) -----------------------
# ---------------------------------------------------------------------------

def load_imdb_multi() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load IMDB-multi dataset via DGL if the library is present."""
    if dgl is None:
        raise RuntimeError("DGL is not installed – cannot load IMDB-multi dataset.")
    try:
        from dgl.data import IMDBDataset

        ds = IMDBDataset("movie")
        g = ds[0]
        feats = g.ndata["feat"]
        labels = g.ndata["label"]
        return torch.tensor(feats), torch.tensor(labels)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"IMDB-multi dataset unavailable: {exc}") from exc
