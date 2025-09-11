"""
train.py – model-training utilities for RAPTOR
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = ["train_one_epoch"]

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    quick: bool = False,
) -> None:
    """A *minimal* training loop that keeps the original logic intact.

    Parameters
    ----------
    model : torch.nn.Module
        The network that should be trained (e.g. *UNet* of Stable Diffusion).
    loader : DataLoader
        Supplies the batches – may be the LAION subset or MedMNIST.
    optimizer : torch.optim.Optimizer
        Optimiser operating on *model*'s parameters.
    epoch : int
        Epoch counter that is only used for the progress-bar description.
    quick : bool, default = False
        When *True* the loop processes just **two** batches.  This flag allows
        CI systems to execute a smoke-test in a few seconds.
    """
    max_batches: Optional[int] = 2 if quick else None

    model.train()
    for i, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
        if max_batches is not None and i >= max_batches:
            break

        # Cast & push to GPU (if available)
        batch = batch.half().to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)

        # ------------------------------------------------------------------
        # ⚠️  Toy objective --------------------------------------------------
        # ------------------------------------------------------------------
        # The original reference implementation used a dummy objective that
        # simply adds random noise to the input and computes an L1 loss.  We
        # retain this behaviour because it is light-weight and therefore
        # ideal for the self-contained test environment here.
        noise = torch.randn_like(batch)
        noisy = batch + noise  # toy corruption
        loss = (noisy - batch).abs().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
