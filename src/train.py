from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class BMRFKalman(torch.nn.Module):
    """1-D Kalman filter for online curvature estimation (per-edge)."""

    mu: torch.Tensor  # registered as buffer
    sigma2: torch.Tensor  # registered as buffer

    def __init__(self, process_noise: float = 1.0, obs_noise: float = 1.0):
        super().__init__()
        # register "learned" state as non-trainable buffers so they travel with .to(device)
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma2", torch.tensor(1.0))
        self.process_noise = float(process_noise)
        self.obs_noise = float(obs_noise)

    def forward(self, kappa_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prediction
        sigma_pred = self.sigma2 + self.process_noise
        # Kalman gain
        K = sigma_pred / (sigma_pred + self.obs_noise)
        # Update
        self.mu = self.mu + K * (kappa_hat - self.mu)
        self.sigma2 = (1 - K) * sigma_pred
        return self.mu, self.sigma2


class CurvatureGatedGCN(nn.Module):
    """GCN stack with optional curvature gating & BMRF."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        use_bmrf: bool = True,
        process_noise: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_bmrf = use_bmrf
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, weight=True))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, weight=True))
        # output layer
        self.layers.append(GraphConv(hidden_dim, out_dim, weight=True))

        if self.use_bmrf:
            self.bmrf = BMRFKalman(process_noise=process_noise, obs_noise=1.0)
        self.register_buffer("gate_threshold", torch.tensor(0.0))

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        h = feat
        for layer in self.layers[:-1]:
            if self.use_bmrf and "kappa_hat" in g.edata:
                with g.local_scope():
                    kappa_hat = g.edata["kappa_hat"]
                    self.bmrf(kappa_hat.mean())  # posterior mean not directly used
                    mask = (kappa_hat > self.gate_threshold).float()
                    g.edata["w"] = mask
                    h = layer(g, h, edge_weight=g.edata["w"])
            else:
                h = layer(g, h)
            h = F.relu(h)
        logits = self.layers[-1](g, h)
        return logits


def train_epoch(
    model: nn.Module,
    g: dgl.DGLGraph,
    idx: torch.Tensor,
    labels: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> float:
    """Single training epoch – returns cross-entropy loss."""
    model.train()
    opt.zero_grad()
    logits = model(g, g.ndata["feat"])
    loss = F.cross_entropy(logits[idx], labels[idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())
