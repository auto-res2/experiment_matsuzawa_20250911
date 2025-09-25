"""
src/train.py
Specialized training logic after placeholder replacement.  Added
1. DialoGPT-medium causal-LM support
2. Language-model aware training loop (per-token loss + ppl)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy (not used for LM)."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_units: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# ---------------------------------------------------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------------------------------------------------

def get_model(config: Dict, input_dim: int, num_classes: int):
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "mlp").lower()

    if model_type == "mlp":
        hidden = model_cfg.get("hidden_units", [128, 64])
        return MLPClassifier(input_dim, num_classes, hidden)

    # DialoGPT-medium causal LM
    if model_type in {"dialogpt", "dialogpt-medium", "microsoft/dialoGPT-medium".lower()}:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    raise NotImplementedError(f"Model type '{model_type}' not implemented.")


# ---------------------------------------------------------------------------------------------------------------------
# Training loop (classification + LM)
# ---------------------------------------------------------------------------------------------------------------------

def _step_lm(model, inputs: torch.Tensor, device: torch.device):
    inputs = inputs.to(device)
    outputs = model(input_ids=inputs, labels=inputs)
    loss = outputs.loss
    logits = outputs.logits.detach()
    # Per-token accuracy for monitoring only
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == inputs).float().mean().item()
    return loss, acc


def train_and_validate(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
):
    set_seed(config.get("seed", 42))

    model = get_model(config, input_dim, num_classes).to(device)

    model_cfg_type = config["model"]["type"].lower()
    is_lm = model_cfg_type.startswith("dialogpt")

    lr = config.get("optimizer", {}).get("lr", 1e-4)
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.get("scheduler", {}).get("step_size", 5), gamma=0.5
    )

    criterion = nn.CrossEntropyLoss() if not is_lm else None
    epochs = config.get("training", {}).get("epochs", 3)

    history: Dict[str, List[float]] = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best_val = float("inf") if is_lm else 0.0  # minimise loss for LM, maximise acc for CLS
    best_ckpt = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        t_loss = t_acc = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            optimizer.zero_grad()
            if is_lm:
                loss, acc = _step_lm(model, batch[0], device)
            else:
                inputs, targets = (b.to(device) for b in batch)
                logits = model(inputs)
                loss = criterion(logits, targets)
                acc = compute_accuracy(logits.detach(), targets)
            loss.backward()
            optimizer.step()

            bs = batch[0].size(0)
            t_loss += loss.item() * bs
            t_acc += acc * bs
        t_loss /= len(train_loader.dataset)
        t_acc /= len(train_loader.dataset)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)

        # ---------- val ----------
        model.eval()
        v_loss = v_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_lm:
                    loss, acc = _step_lm(model, batch[0], device)
                else:
                    inputs, targets = (b.to(device) for b in batch)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    acc = compute_accuracy(logits, targets)
                bs = batch[0].size(0)
                v_loss += loss.item() * bs
                v_acc += acc * bs
        v_loss /= len(val_loader.dataset)
        v_acc /= len(val_loader.dataset)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        scheduler.step()

        if is_lm:
            metric = v_loss
            improved = metric < best_val
        else:
            metric = v_acc
            improved = metric > best_val
        if improved:
            best_val = metric
            torch.save({"model_state": model.state_dict(), "config": config, "history": history}, best_ckpt)

        print(
            f"Epoch {epoch:02d}: train_loss={t_loss:.4f} val_loss={v_loss:.4f} "
            f"train_acc={t_acc:.4f} val_acc={v_acc:.4f}"
        )

    return {"history": history, "best_ckpt_path": str(best_ckpt)}
