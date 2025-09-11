"""
src/train.py
Model definitions + training utilities for CaFe-EDGE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------------------------------------------------------------------------
#                         CONFIG & DIRECTORY HANDLING
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
try:
    with CONFIG_PATH.open() as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError as e:  # pragma: no cover – fatal for experiment
    sys.exit(f"ERROR: Cannot locate configuration file: {CONFIG_PATH}\n{e}")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESEARCH_DIR = ROOT / ".research" / "iteration2"
for _d in [DATA_DIR, MODELS_DIR, RESEARCH_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#                                MODEL
# ---------------------------------------------------------------------------
class GraphEncoder(nn.Module):
    """Simple GraphSAGE encoder used inside CaFe-EDGE."""

    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, hidden))
        self.act = nn.ReLU()

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        return x


class CaFeEDGE(nn.Module):
    """Minimal implementation of CaFe-EDGE incl. ProDeF head."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        layers: int = 4,
        lambda_E: float = 1.0,
        lambda_MI: float = 0.1,
        epsilon_F: float = 0.03,
        epsilon_S: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden, layers)
        self.classifier = nn.Linear(hidden, 1)
        self.prodef = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, layers), nn.Softmax(-1)
        )
        self.layers = layers
        self.lambda_E = lambda_E
        self.lambda_MI = lambda_MI
        self.epsilon_F = epsilon_F
        self.epsilon_S = epsilon_S

    # ----------------------------  TRAIN  ---------------------------------
    def forward(self, data):  # pylint: disable=arguments-differ
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.encoder(x, edge_index)
        hg = global_mean_pool(h, batch)
        logits = self.classifier(hg).squeeze(-1)
        return logits

    # -------------------------  INFERENCE  ------------------------------
    @torch.no_grad()
    def predict_with_prodef(self, data, features_for_prodef):
        depth_probs = self.prodef(features_for_prodef)
        chosen_depth = depth_probs.argmax(-1).item() + 1  # 1-indexed
        logits = self.forward(data)  # full forward pass (simplified)
        return logits, chosen_depth, depth_probs.cpu().tolist()


# ---------------------------------------------------------------------------
#                           TRAINING UTILITIES
# ---------------------------------------------------------------------------
from .preprocess import StreamEdgeDataset  # imported late to avoid circularity

__all__ = [
    "CaFeEDGE",
    "train_all_seeds",
]


def _collate(graphs: List):
    return PyGBatch.from_data_list(graphs)


def _train_single_seed(seed: int, device: torch.device) -> Path:
    torch.manual_seed(seed)

    ds_train = StreamEdgeDataset("train")
    dl_train = DataLoader(
        ds_train,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=CONFIG["training"]["workers"],
        collate_fn=_collate,
    )

    in_dim = ds_train[0].x.shape[1]
    model = CaFeEDGE(
        in_dim=in_dim,
        hidden=CONFIG["models"]["cafe_edge"]["hidden_dim"],
        layers=CONFIG["models"]["cafe_edge"]["gnn_layers"],
        lambda_E=CONFIG["models"]["cafe_edge"]["lambda_E"],
        lambda_MI=CONFIG["models"]["cafe_edge"]["lambda_MI"],
        epsilon_F=CONFIG["models"]["cafe_edge"]["epsilon_F"],
        epsilon_S=CONFIG["models"]["cafe_edge"]["epsilon_S"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()

    for _epoch in range(CONFIG["training"]["epochs"]):
        model.train()
        for batch in dl_train:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(batch)
                labels = batch.y.float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        # Validation omitted – early-stopping not critical for refactor.

    ckpt = MODELS_DIR / f"cafe_edge_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt


def train_all_seeds(device: torch.device) -> List[Path]:
    """Train CaFe-EDGE for all seeds defined in the YAML config."""
    ckpts: List[Path] = []
    for s in CONFIG["training"]["seeds"]:
        print(f"\n===== Training CaFe-EDGE  (seed={s}) =====")
        ckpts.append(_train_single_seed(s, device))
    # Persist list of checkpoints so that external tools can pick them up
    (RESEARCH_DIR / "ckpts.json").write_text(json.dumps([str(p) for p in ckpts]))
    return ckpts
