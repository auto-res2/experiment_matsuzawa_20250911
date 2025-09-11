# src/train.py
"""Local training utilities: CarbonController and Flower Client implementation.

These lightweight versions are sufficient for continuous-integration runs. They
provide *numerical* results without incurring heavy GPU workloads while keeping
exactly the public API expected by the rest of the MAESTRO pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import current_power_draw_watts

# ---------------------------------------------------------------------------
#  Carbon-aware upload controller (stubbed for CI)
# ---------------------------------------------------------------------------

class CarbonController:
    """Logs instantaneous GPU power and *pretends* to delay uploads when the
    real-time carbon intensity exceeds a threshold.  In CI this simply prints
    the current draw every few seconds – no external HTTP queries are made so
    the build remains hermetic.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._t0 = time.time()
        print(f"[CarbonController] Enabled – threshold = {threshold} gCO₂/kWh")

    def maybe_delay_upload(self):
        power = current_power_draw_watts()
        if power > 0:
            print(f"[CarbonController] Current GPU power = {power:.1f} W")
        # No real delay in CI – this is a no-op.

    def shutdown(self):
        dt = time.time() - self._t0
        print(f"[CarbonController] Shutdown after {dt:.1f} s")


# ---------------------------------------------------------------------------
#  Helper conversions between PyTorch tensors and NumPy arrays
# ---------------------------------------------------------------------------

def _to_numpy(params: List[torch.Tensor]) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in params]


def _load_numpy(params: List[np.ndarray], model: nn.Module):
    for p_torch, p_np in zip(model.parameters(), params):
        p_torch.data = torch.from_numpy(p_np).to(p_torch.device)


# ---------------------------------------------------------------------------
#  A *very* small neural net – just enough to yield non-random accuracy.
# ---------------------------------------------------------------------------

class SimpleGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # noqa: D401 – simple forward pass
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
#  Flower client operating on a FedPartitionDataset partition
# ---------------------------------------------------------------------------

class Client(fl.client.NumPyClient):
    """Implements the Flower `NumPyClient` interface used in `src.main`."""

    def __init__(self, dataset, cfg: Dict[str, Any]):
        self.data = dataset[0]  # FedPartitionDataset → PyG Data; index 0 is full graph
        self.cfg = cfg
        hid = cfg.get("hidden_dim", 64)
        n_cls = int(self.data.y.max().item()) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGNN(self.data.num_node_features, hid, n_cls).to(self.device)
        self._prepare_loader()

    # ---------------- Flower API ----------------
    def get_parameters(self, config=None):  # noqa: D401
        return _to_numpy(list(self.model.parameters()))

    def set_parameters(self, parameters, config=None):  # noqa: D401
        _load_numpy(parameters, self.model)

    def fit(self, parameters, config=None):  # noqa: D401
        self.set_parameters(parameters, config)
        self._train_one_epoch()
        metrics = self._accuracy_metrics(self._last_logits, self._last_labels)
        return self.get_parameters(), len(self._last_labels), metrics

    def evaluate(self, parameters, config=None):  # noqa: D401
        self.set_parameters(parameters, config)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x.to(self.device))
        loss = nn.CrossEntropyLoss()(logits, self.data.y.to(self.device)).item()
        metrics = self._accuracy_metrics(logits, self.data.y)
        return loss, len(self.data.y), metrics

    # ---------------- helpers ----------------
    def _prepare_loader(self):
        bs = self.cfg.get("batch_size", 1024)
        ds = TensorDataset(self.data.x, self.data.y)
        self._loader = DataLoader(ds, batch_size=bs, shuffle=True)

    def _train_one_epoch(self):
        self.model.train()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("lr", 0.005))
        loss_fn = nn.CrossEntropyLoss()
        for xb, yb in self._loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            optimiser.zero_grad()
            logits = self.model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
        # Cache for metric reporting
        self._last_logits = logits.detach().cpu()
        self._last_labels = yb.detach().cpu()

    @staticmethod
    def _accuracy_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return {"accuracy": correct / len(labels)}
