# src/train.py
"""Local training utilities: CarbonController and Flower Client implementation.

These lightweight versions are sufficient for continuous-integration runs. They
provide *numerical* results without incurring heavy GPU workloads while keeping
exactly the public API expected by the rest of the MAESTRO pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

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

def _to_numpy(params: Sequence[torch.Tensor]) -> List[np.ndarray]:
    """Detach and move tensors to CPU, returning NumPy arrays."""
    return [p.detach().cpu().numpy() for p in params]


def _load_numpy(params: Sequence[object], model: nn.Module):
    """Load parameters into *model*.

    The *params* argument may contain either ``np.ndarray`` objects (the usual
    Flower <-> Client interface) **or** raw ``bytes`` (when we manually pass
    ``strategy.final_parameters.tensors``).  This helper transparently handles
    both cases, performing strict size/shape checks so that any mismatch is
    caught immediately.
    """

    params_iter = iter(params)
    for p_torch in model.parameters():
        try:
            p_src = next(params_iter)
        except StopIteration as exc:  # pragma: no cover – defensive
            raise ValueError("Not enough tensors when loading NumPy weights") from exc

        # ------------------------------------------------------------------
        # 1) Decode – bytes → ndarray if needed. We must be robust to dtype
        #     disparities introduced during aggregation (e.g. float32 → float64).
        # ------------------------------------------------------------------
        if isinstance(p_src, bytes):
            # Determine dtype by inspecting byte length; aggregated parameters
            # might be float64 even if the original model used float32.
            n_elems = p_torch.numel()
            expected_bytes_fp32 = n_elems * 4
            expected_bytes_fp64 = n_elems * 8
            if len(p_src) == expected_bytes_fp32:
                dtype = np.float32
            elif len(p_src) == expected_bytes_fp64:
                dtype = np.float64
            else:
                raise ValueError(
                    "Parameter size mismatch when loading NumPy weights (raw-bytes)"
                )
            p_np = np.frombuffer(p_src, dtype=dtype)
        elif isinstance(p_src, np.ndarray):
            p_np = p_src
        else:
            raise TypeError(
                "Expected elements of parameters to be either bytes or np.ndarray, "
                f"got {type(p_src)}"
            )

        # ------------------------------------------------------------------
        # 2) Sanity checks – size/shape must match target tensor
        # ------------------------------------------------------------------
        if p_torch.numel() != p_np.size:
            raise ValueError("Parameter size mismatch when loading NumPy weights")

        # 3) Copy data (casting dtype if necessary) --------------------------
        tensor = torch.from_numpy(p_np.reshape(p_torch.shape))
        if tensor.dtype != p_torch.dtype:
            tensor = tensor.to(p_torch.dtype)
        p_torch.data.copy_(tensor.to(p_torch.device))


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
        metrics = self._accuracy_metrics(logits.cpu(), self.data.y.cpu())
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