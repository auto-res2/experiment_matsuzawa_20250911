# src/train.py
"""Model architectures, federated client and carbon controller.
All heavy-weight logic that touches the GPU or the network lives here so
that other modules can import lightweight utilities only.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Dict, List

import requests
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
import flwr as fl

# ---------------------------------------------------------------------------
#  Safety helpers (NO-FALLBACK philosophy)
# ---------------------------------------------------------------------------

def abort(msg: str):
    """Terminate immediately – external callers must handle clean-up."""
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
#  Continuous-time GNN
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """Right-hand side  dh/dt = f(h,A)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = SAGEConv(in_dim, in_dim)
        self.nfe: int = 0  # number of function evaluations

    def forward(self, t: Tensor, h: Tensor, edge_index: Tensor):  # noqa: N802
        self.nfe += 1
        return F.relu(self.conv(h, edge_index))


class CTGNN(nn.Module):
    """Encoder – ODE – Decoder architecture used in all experiments."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, step_size: float):
        super().__init__()
        from torchdiffeq import odeint_adjoint as odeint  # local import, avoids global pollut.

        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_classes)
        self.step_size = step_size
        self._odeint = odeint

    def forward(self, data):  # `data` is torch_geometric.data.Data
        x = self.encoder(data.x)
        t = torch.tensor([0, self.step_size], device=x.device)
        z = self._odeint(
            self.odefunc,
            x,
            t,
            method="dopri5",
            options={"step_size": self.step_size},
            args=(data.edge_index,),
        )[-1]
        return self.decoder(z)


# ---------------------------------------------------------------------------
#  FLwr client wrapper
# ---------------------------------------------------------------------------


def get_model_state(model: nn.Module):
    return {k: v.cpu() for k, v in model.state_dict().items()}


def set_model_state(model: nn.Module, state):
    model.load_state_dict(state)


class Client(fl.client.NumPyClient):
    """Flower client around a single data partition."""

    def __init__(self, dataset, cfg_exp1: Dict):
        from torch_geometric.loader import NeighborLoader  # late import keeps start-up light
        self.data = dataset
        self.cfg = cfg_exp1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CTGNN(
            in_dim=dataset.num_node_features,
            hidden_dim=cfg_exp1["hidden_dim"],
            num_classes=int(dataset.data.y.max().item()) + 1,
            step_size=cfg_exp1["ode_step_size"],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg_exp1["lr"])
        # One huge batch per local epoch – identical to the monolithic script.
        self.loader = NeighborLoader(dataset, batch_size=dataset.data.num_nodes, num_neighbors=[-1])
        self.criterion = nn.CrossEntropyLoss()

    #  ------ FLwr interface -------------------------------------------------
    def get_parameters(self, *args, **kwargs):  # noqa: D401  (Flower API)
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters, *args, **kwargs):  # noqa: D401
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        set_model_state(self.model, state_dict)

    def fit(self, parameters, config):  # noqa: D401
        self.set_parameters(parameters, config)
        self.model.train()
        for _ in range(self.cfg["local_epochs"]):
            for batch in self.loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.data), {}

    def evaluate(self, parameters, config):  # noqa: D401
        self.set_parameters(parameters, config)
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.loader)).to(self.device)
            logits = self.model(batch)
            pred = logits.argmax(dim=-1)
            acc = (pred == batch.y).float().mean().item()
            loss = self.criterion(logits, batch.y).item()
        return float(loss), len(self.data), {"accuracy": float(acc)}


# ---------------------------------------------------------------------------
#  Carbon intensity controller (Exp-1)
# ---------------------------------------------------------------------------

class CarbonController:  # pylint: disable=too-few-public-methods
    """Continuously fetch carbon intensity and decide if uploads are allowed."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._last_val: float | None = None
        self._stop = False
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    # ------------------------- internal helpers ---------------------------
    def _poll(self):
        url = "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=US"
        headers = {"auth-token": os.getenv("ELECTRICITYMAP_TOKEN", "")}
        if not headers["auth-token"]:
            abort("ELECTRICITYMAP_TOKEN environment variable not set – cannot fetch carbon data.")
        while not self._stop:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                intensity = response.json()["carbonIntensity"]
                with self._lock:
                    self._last_val = intensity
            except Exception as exc:  # noqa: BLE001
                abort(f"ElectricityMap API unreachable: {exc}")
            time.sleep(300)  # poll every 5 minutes

    # ------------------------ public API ---------------------------------
    def ok_to_upload(self) -> bool:  # noqa: D401
        with self._lock:
            val = self._last_val
        return val is not None and val < self.threshold

    def shutdown(self):
        self._stop = True
        self.thread.join()
