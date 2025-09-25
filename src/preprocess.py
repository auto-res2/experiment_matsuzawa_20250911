"""src/preprocess.py
Specialised preprocessing pipeline for *SQuAD* as a light-weight
classification task (answer-length buckets).
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# ----------------------------------------------------------------------------
# Dataset loader (specialises the previous placeholder)
# ----------------------------------------------------------------------------

def _load_dataset_placeholder(config: Dict):  # noqa: N802 – keep original name
    """Load *SQuAD* and turn it into a 3-class classification dataset.

    Label schema (based on *answer token length*):
    • 0 – *short*  (<5 tokens)
    • 1 – *medium* (5–15 tokens)
    • 2 – *long*   (>15 tokens)
    """

    # ---------------- Configuration -------------------------------------
    model_name = config["model"]["name"]
    max_len = int(config["dataset"].get("max_length", 64))
    rng = random.Random(int(config["training"].get("seed", 42)))

    # ---------------- Load raw dataset ----------------------------------
    ds_raw = load_dataset("squad")

    # We down-sample aggressively so that the experiment fits into the 500 MB
    # RAM envelope of the execution environment.
    def _sample(split):
        indices = list(range(len(ds_raw[split])))
        rng.shuffle(indices)
        return indices[:1000]  # ≤1 k samples per split

    sampled = {
        "train": ds_raw["train"].select(_sample("train")),
        "validation": ds_raw["validation"].select(_sample("validation")),
    }

    # ---------------- Tokeniser -----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _vectorise(example):
        text = example["question"] + " " + example["context"]
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        # --------------- Build label (answer length bucket) --------------
        ans_len = len(tokenizer(example["answers"]["text"][0])["input_ids"])
        if ans_len < 5:
            label = 0
        elif ans_len <= 15:
            label = 1
        else:
            label = 2
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

    # Vectorise splits ----------------------------------------------------
    proc_train = sampled["train"].map(_vectorise)
    proc_val = sampled["validation"].map(_vectorise)

    # Convert to *TensorDataset* -----------------------------------------
    def _to_tensor_dataset(split):
        inputs = torch.stack(split["input_ids"])
        labels = torch.stack(split["label"])
        return TensorDataset(inputs, labels)

    train_ds = _to_tensor_dataset(proc_train)
    val_ds = _to_tensor_dataset(proc_val)

    # We reuse validation set as test – SQuAD has no dedicated *test* part
    return train_ds, val_ds, val_ds


# ----------------------------------------------------------------------------
# Data Pre-Processor (public API unchanged)
# ----------------------------------------------------------------------------

class DataPreprocessor:
    """Factory to create train/val/test dataloaders irrespective of dataset."""

    @staticmethod
    def _create_dummy_dataset(config: Dict):
        # unchanged (see common core)
        input_dim = int(config["dataset"].get("input_dim", 20))
        num_classes = int(config["dataset"].get("num_classes", 3))
        n_samples = int(config["dataset"].get("n_samples", 1000))

        rng = np.random.default_rng(seed=int(config["training"].get("seed", 42)))
        X = rng.normal(size=(n_samples, input_dim)).astype(np.float32)
        y = rng.integers(low=0, high=num_classes, size=n_samples, dtype=np.int64)

        indices = np.arange(n_samples)
        rng.shuffle(indices)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        idx_train, idx_val, idx_test = (
            indices[:train_end],
            indices[train_end:val_end],
            indices[val_end:],
        )

        def make_dataset(idxs):
            return TensorDataset(
                torch.from_numpy(X[idxs]), torch.from_numpy(y[idxs])
            )

        return make_dataset(idx_train), make_dataset(idx_val), make_dataset(idx_test)

    # ----------------------- Public API ----------------------------------

    @staticmethod
    def get_data_loaders(config: Dict):
        dataset_name = config["dataset"]["name"].lower()
        if dataset_name == "dummy":
            train_ds, val_ds, test_ds = DataPreprocessor._create_dummy_dataset(config)
        else:
            train_ds, val_ds, test_ds = _load_dataset_placeholder(config)

        batch_size = int(config["training"].get("batch_size", 32))

        def make_loader(ds, shuffle: bool):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        return (
            make_loader(train_ds, shuffle=True),
            make_loader(val_ds, shuffle=False),
            make_loader(test_ds, shuffle=False),
        )
