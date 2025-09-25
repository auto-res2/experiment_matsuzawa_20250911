"""
src/preprocess.py
Dataset loading now supports the HuggingFace SQuAD dataset for causal-LM
finetuning of DialoGPT.  We create short (<128 token) dialogue-style strings
("Question: ... Answer: ...") and tokenize them.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------------------------------------------------------------------------------------------------------------------
# DataModule abstraction
# ---------------------------------------------------------------------------------------------------------------------

class DataModule:
    def __init__(self, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, batch_size: int = 32, num_workers: int = 0, collate_fn=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        # Attribute stubs (not used by LM)
        sample = train_ds[0][0] if isinstance(train_ds[0], (tuple, list)) else train_ds[0]
        if isinstance(sample, torch.Tensor):
            self.input_dim = int(math.prod(sample.shape))
        else:
            self.input_dim = 0  # unknown / not required
        self.num_classes = 0

# ---------------------------------------------------------------------------------------------------------------------
# SQuAD builder (for causal LM)
# ---------------------------------------------------------------------------------------------------------------------

def _build_squad_lm(cfg: Dict):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    max_len = cfg.get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    raw = load_dataset("squad")

    def _convert(split):
        texts = []
        for ex in split:
            q, a = ex["question"], ex["answers"]["text"][0]
            txt = f"Question: {q} Answer: {a}"
            texts.append(txt)
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc.input_ids
        return TensorDataset(input_ids, input_ids)

    train_ds = _convert(raw["train"][0:cfg.get("train_size", 1000)])
    val_ds = _convert(raw["train"][cfg.get("train_size", 1000): cfg.get("train_size", 1000)+cfg.get("val_size", 200)])
    test_ds = _convert(raw["validation"][0:cfg.get("test_size", 200)])

    collate_fn = None  # already padded to max_length, simple stacking works
    return DataModule(train_ds, val_ds, test_ds, batch_size=cfg.get("batch_size", 8), num_workers=cfg.get("num_workers",0), collate_fn=collate_fn)

# ---------------------------------------------------------------------------------------------------------------------
# dummy dataset (kept for smoke test)
# ---------------------------------------------------------------------------------------------------------------------

def _build_dummy_dataset(cfg: Dict):
    input_dim = cfg["input_dim"]
    num_classes = cfg["num_classes"]
    def _make(size):
        return TensorDataset(torch.randn(size, input_dim), torch.randint(0, num_classes, (size,)))
    return _make(cfg.get("train_size", 1000)), _make(cfg.get("val_size", 200)), _make(cfg.get("test_size", 200))

# ---------------------------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------------------------

def build_datamodule(config: Dict) -> DataModule:
    name = config.get("dataset", {}).get("name", "dummy").lower()

    if name == "dummy":
        train, val, test = _build_dummy_dataset(config["dataset"])
        return DataModule(train, val, test, batch_size=config["dataset"].get("batch_size",32))

    if name == "squad":
        return _build_squad_lm(config["dataset"])

    raise NotImplementedError(f"Dataset '{name}' not implemented.")
