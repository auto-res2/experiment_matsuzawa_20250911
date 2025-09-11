import pathlib
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

__all__ = ["evaluate", "line_plot"]

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
#  EVALUATION ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)
    return {"hit@1": correct / max(1, n)}


# ---------------------------------------------------------------------------
#  PLOTTING -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def line_plot(
    *,
    xs: List[int],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    fname: str,
    images_root: pathlib.Path,
) -> str:
    """Save a simple line plot and return the file path (PDF)."""
    _ensure_dir(images_root)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.3f}", fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    out_path = images_root / f"{fname}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return str(out_path)
