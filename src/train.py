"""src/train.py
Core training logic specialised with real Hugging-Face models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# -----------------------------------------------------------------------------
# Model instantiation helpers --------------------------------------------------
# -----------------------------------------------------------------------------


def _build_hf_sequence_classifier(config: Dict) -> nn.Module:
    """Create a *sequence-classification* wrapper around any HF backbone.

    The function is purposely light-weight so that *any* AutoModel backbone can
    be used while still fitting into the generic Trainer loop (i.e. it accepts
    only a single *input_ids* tensor and returns raw *logits*).
    """

    model_name = config["model"]["name"]
    num_labels = int(config["dataset"].get("num_classes", 3))

    # --------------- Load backbone & classification head ----------------
    hf_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    backbone = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=hf_config
    )

    # --------------- Build tiny inference wrapper -----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id

    class _HFWrapper(nn.Module):  # noqa: D401 – internal helper
        """Thin wrapper so that *forward(x)* is enough for the core Trainer."""

        def __init__(self, base_model: nn.Module, pad_token_id: int):
            super().__init__()
            self.base = base_model
            self.pad_id = pad_token_id

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            attention_mask = (input_ids != self.pad_id).long()
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    return _HFWrapper(backbone, pad_id)


# -----------------------------------------------------------------------------
# Public factory ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def instantiate_model(config: Dict) -> nn.Module:
    """Instantiate a model based on *config*.

    Supported *model.name* options:
    • "dummy" – built-in toy MLP for smoke tests
    • Any valid 🤗 model identifier (e.g. "microsoft/DialoGPT-medium") which
      will be loaded as a *sequence-classification* model via
      :func:`_build_hf_sequence_classifier`.
    """

    name = config["model"]["name"].lower()

    # ---------------- Dummy ---------------------------------------------------
    if name == "dummy":
        input_dim = int(config["dataset"].get("input_dim", 20))
        num_classes = int(config["dataset"].get("num_classes", 3))
        hidden_dim = int(config["model"].get("hidden_dim", 32))
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        return model

    # ---------------- Hugging-Face backbone ----------------------------------
    return _build_hf_sequence_classifier(config)


# -----------------------------------------------------------------------------
# Trainer (unchanged except for type comments) ---------------------------------
# -----------------------------------------------------------------------------


class Trainer:
    """Universal training wrapper that is *entirely* dataset-agnostic."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler | None,
        device: torch.device,
        output_dir: Path,
        config: Dict,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.config = config
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_path = self.output_dir / "best_model.pth"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Helpers

    @staticmethod
    def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)

    def _run_epoch(self, train: bool = True) -> Tuple[float, float]:
        data_loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        running_loss = 0.0
        running_acc = 0.0
        total = 0

        with torch.set_grad_enabled(train):
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                acc = self._accuracy(outputs, targets)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += acc * inputs.size(0)
                total += inputs.size(0)

        return running_loss / total, running_acc / total

    # ------------------------------------------------------------------ Public

    def train(self) -> Dict:
        num_epochs = int(self.config["training"].get("epochs", 10))
        print(f"Starting training for {num_epochs} epochs …")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._run_epoch(train=True)
            val_loss, val_acc = self._run_epoch(train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)

            print(
                f"Epoch [{epoch}/{num_epochs}] – "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} – "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

        total_time = time.time() - start_time
        print(f"Training finished in {total_time/60:.2f} minutes.")

        metrics_path = self.output_dir / "training_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(self.history, fp, indent=2)

        return {
            "metrics_path": str(metrics_path),
            "best_model_path": str(self.best_model_path),
            "history": self.history,
        }
