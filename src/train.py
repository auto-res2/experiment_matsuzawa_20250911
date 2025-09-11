# src/train.py
"""Training-related classes and helpers extracted from the monolithic
HERMES-MEM reference script.

NOTE  No behavioural changes were introduced – only minimal
refactoring (imports / path handling / device guards / type safety)
necessary to make the code runnable as a stand-alone module inside the
new project layout.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .evaluate import plot_accuracy_curve  # relative import – safe after refactor

# -----------------------------------------------------------
# Model building blocks
# -----------------------------------------------------------


class HermesMemHead(nn.Module):
    """Toy HERMES-MEM head – identical interface, no crypto."""

    def __init__(self, num_keys: int, dim: int):
        super().__init__()
        self.keys = nn.Embedding(num_keys, dim)
        nn.init.normal_(self.keys.weight, std=0.02)
        self.classifier = nn.Linear(dim, 10)  # 10-way per task

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        kid = torch.randint(0, self.keys.num_embeddings, (x.size(0),), device=x.device)
        kv = self.keys(kid)
        fused = (x + kv) / 2.0
        return self.classifier(fused)


class HoloZipHead(HermesMemHead):
    pass  # identical in this compact reference implementation


class GrailHead(HermesMemHead):
    pass


# -----------------------------------------------------------
# Backbone + factory
# -----------------------------------------------------------

def _get_backbone(cfg: Dict) -> nn.Module:
    mb = torchvision.models.mobilenet_v2(
        width_mult=cfg["models"]["backbone"]["width_mult"],
        pretrained=cfg["models"]["backbone"]["pretrained"],
    )
    mb.classifier = nn.Identity()
    return mb


def build_model(cfg: Dict, head_variant: str = "hermes") -> nn.Module:
    """Build MobileNetV2 + memory head as single nn.Module."""

    feature_dim = 1280  # fixed for MobileNetV2
    bb = _get_backbone(cfg)

    if head_variant == "hermes":
        head_cfg = cfg["models"]["memory"]["hermes"]
        head = HermesMemHead(head_cfg["num_keys"], head_cfg["dim"])
    elif head_variant == "holozip":
        head_cfg = cfg["models"]["memory"]["holozip"]
        head = HoloZipHead(head_cfg["num_keys"], 128)
    elif head_variant == "grail":
        head_cfg = cfg["models"]["memory"]["grail"]
        head = GrailHead(head_cfg["num_keys"], 128)
    else:
        raise ValueError(f"Unknown head {head_variant}")

    net = nn.Sequential(
        bb,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(feature_dim, 128),
        head,
    )
    return net


# -----------------------------------------------------------
# Federated-learning simulator (Experiment-1)
# -----------------------------------------------------------


class _Device:
    """Single simulated client device."""

    def __init__(
        self,
        idx: int,
        dataloader: DataLoader,
        model_fn: Callable[[], nn.Module],
        cfg: Dict,
        device: torch.device,
    ) -> None:
        self.idx = idx
        self.loader = dataloader
        self.model = model_fn().to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg["train"]["lr"])

    def train_one_epoch(self) -> Dict[str, torch.Tensor]:
        self.model.train()
        for batch in self.loader:
            img = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            out = self.model(img)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            self.opt.step()
        # Return detached copy to prevent CUDA IPC issues
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


class FederatedTrainer:
    """Asynchronous FL simulator – supports a single sequential stream."""

    # Type annotation added for static analysis tools
    results: DefaultDict[str, List[float]]

    def __init__(self, tasks: List[Subset], cfg: Dict, results_dir: str, figure_dir: str):
        self.tasks = tasks
        self.cfg = cfg
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.global_model = build_model(cfg, head_variant="hermes").to(self.device)
        self.results = defaultdict(list)
        self._results_dir = pathlib.Path(results_dir)
        self._fig_dir = pathlib.Path(figure_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------
    def _aggregate(self, deltas: List[Dict[str, torch.Tensor]]) -> None:
        with torch.no_grad():
            for name, param in self.global_model.state_dict().items():
                stacked = torch.stack([d[name].to(self.device) for d in deltas], dim=0)
                param.copy_(stacked.mean(dim=0))

    def _eval_task(self, t_idx: int) -> float:
        self.global_model.eval()
        loader = DataLoader(self.tasks[t_idx], batch_size=64)
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
                out = self.global_model(img)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def run(self) -> None:
        rounds = self.cfg["train"]["rounds"]
        devices_per_round = max(1, int(self.cfg["global"]["device_world_size"] * 0.1))
        rng = random.Random(self.cfg["global"]["seed_list"][0])

        for rnd in tqdm(range(rounds), desc="Federated Rounds"):
            t_idx = (rnd // 2) % len(self.tasks)
            ds = self.tasks[t_idx]
            indices = rng.sample(range(len(ds)), k=min(len(ds), 32 * devices_per_round))
            split_indices = torch.chunk(torch.tensor(indices), devices_per_round)
            deltas = []
            for didx, idx_subset in enumerate(split_indices):
                loader = DataLoader(
                    Subset(ds, idx_subset.tolist()),
                    batch_size=self.cfg["train"]["batch_size"],
                )
                device = _Device(
                    didx,
                    loader,
                    lambda: build_model(self.cfg, "hermes"),
                    self.cfg,
                    self.device,
                )
                deltas.append(device.train_one_epoch())
            self._aggregate(deltas)
            acc = self._eval_task(t_idx)
            self.results["accuracy"].append(float(acc))

        self._finalise()

    # -------------------------------------------------------
    def _finalise(self) -> None:
        res_file = self._results_dir / "exp1_results.json"
        with open(res_file, "w") as fp:
            json.dump(self.results, fp, indent=2)
        print("\nEXPERIMENT 1 – Asynchronous FL with verifiable deletion")
        print(json.dumps(self.results, indent=2))

        fig_path = self._fig_dir / "accuracy_training_curve.pdf"
        plot_accuracy_curve(self.results["accuracy"], str(fig_path))
        print(f"Figures saved: {fig_path}")
