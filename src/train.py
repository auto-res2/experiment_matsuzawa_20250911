"""
Training logic for RAPTOR experiments (single-file refactor of the original
script).  All building blocks that used to live in multiple modules are now
collapsed here so we remain within the six-file constraint.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:  # optional – will be silently skipped if fvcore is unavailable
    from fvcore.nn.flop_count import flop_count_table  # noqa: F401
except Exception:  # pragma: no cover
    flop_count_table = None

# ============================================================================
#                          CONFIGURATION OBJECTS
# ============================================================================


@dataclass
class HutchConf:
    top_k: int
    sketch: int
    refresh: int


@dataclass
class SchedConf:
    tau: float


@dataclass
class TrainConf:
    pretrain_steps: int
    finetune_epochs: int
    lr_pretrain: float
    lr_finetune: float


@dataclass
class ExperimentConf:
    id: str
    description: str
    seeds: List[int]
    variants: List[str]
    data: dict
    model: dict
    training: TrainConf
    hutch: HutchConf
    scheduler: SchedConf
    output_dir: str


# ----------------------------------------------------------------------------
#                             YAML LOADER
# ----------------------------------------------------------------------------

def load_yaml(path: str | Path) -> ExperimentConf:
    """Parse the configuration YAML into an ExperimentConf dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found.")
    with path.open() as fp:
        raw = yaml.safe_load(fp)

    exp = raw["experiment"]
    return ExperimentConf(
        id=exp["id"],
        description=exp["description"],
        seeds=exp["seeds"],
        variants=exp["variants"],
        data=exp["data"],
        model=exp["model"],
        training=TrainConf(**exp["training"]),
        hutch=HutchConf(**exp["hutch"]),
        scheduler=SchedConf(**exp["scheduler"]),
        output_dir=exp["output_dir"],
    )


# ============================================================================
#                                CORE MODULES
# ============================================================================
import torch.nn as nn  # noqa: E402
from diffusers import DiffusionPipeline  # noqa: E402


class HutchFisher(nn.Module):
    """Hutch++ Streaming approximation of the Fisher Information Matrix top-K eigenvectors."""

    eig_vec: torch.Tensor  # static type hint for mypy

    def __init__(self, model: nn.Module, topk: int = 128, sketch: int = 2048, refresh: int = 256):
        super().__init__()
        self.model = model
        self.topk = topk
        self.sketch = sketch
        self.refresh = refresh
        # register_buffer ensures tensor moves with .to(device)
        self.register_buffer("eig_vec", torch.randn(topk, self.numel()).normal_())
        self.steps = 0

    def numel(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def forward(self, grads_flat: torch.Tensor):
        self.steps += 1
        if self.steps % self.refresh != 0:
            return self.eig_vec
        device = grads_flat.device
        R = torch.randint(0, 2, (self.sketch,), device=device, dtype=torch.float32) * 2 - 1
        Y = torch.outer(R, grads_flat)
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ torch.diag_embed(grads_flat) @ Q
        eigv, _ = torch.linalg.eigh(B)
        top_ids = torch.argsort(eigv, descending=True)[: self.topk]
        # runtime assignment is safe; the buffer is updated in-place
        self.eig_vec = Q[:, top_ids].T
        return self.eig_vec


class AsyncScheduler:
    """Event-driven token scheduler."""

    def __init__(self, unet: nn.Module, fisher: HutchFisher, tau: float = 0.015):
        self.unet = unet
        self.fisher = fisher
        self.tau = tau

    def select_tokens(self, scores_before, scores_after):
        delta = (scores_after - scores_before).norm(dim=-1)
        return delta > self.tau


class ControlVariate(nn.Module):
    """Simple control-variate autoregressive model that predicts jump counts."""

    def __init__(self, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, 256)
        self.fc = nn.Linear(256, 1)

    def forward(self, ids):
        h = torch.tanh(self.embed(ids))
        return self.fc(h).squeeze(-1)


class CarbonMonitor:
    """
    Rudimentary GPU energy monitor using NVML.  The class is optional – if NVML
    is not available (e.g. on CPU boxes), monitoring is silently disabled so
    that experiments can still run.
    """

    def __init__(self, out_file: Path | str):
        self.enabled = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception:
            # NVML unavailable – disable energy tracking
            self.enabled = False
        self.start_t = time.time()
        self.j_gpu = 0.0
        self.out_file = Path(out_file)

    def sample(self):
        if not self.enabled:
            return
        p = self._pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3  # watt
        self.j_gpu += p * 0.1  # assume call every 100 ms

    def stop(self):
        res = {"joule_gpu": self.j_gpu, "wall_clock": time.time() - self.start_t}
        try:
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            with self.out_file.open("w") as fp:
                json.dump(res, fp)
        except Exception as e:  # pragma: no cover
            print(f"[CarbonMonitor] Could not write energy file: {e}", file=sys.stderr)
        return res


class RaptorDiffuser:
    """Wrapper around Stable-Diffusion XL with the RAPTOR modules attached."""

    def __init__(self, model_id: str, fisher_conf: HutchConf, sched_conf: SchedConf, control_var: bool = True):
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Unable to load model {model_id}. Ensure diffusers & model weights are available.\n{e}"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(device)

        self.fisher = HutchFisher(self.pipe.unet, **fisher_conf.__dict__)
        self.scheduler = AsyncScheduler(self.pipe.unet, self.fisher, tau=sched_conf.tau)
        self.control = ControlVariate(8192) if control_var else None


# ============================================================================
#                                   TRAIN
# ============================================================================
from .preprocess import ImageTokenDataset  # noqa: E402


def fit(exp_conf: ExperimentConf, variant: str, seed: int):
    """Fine-tune the diffusion model according to the configuration."""
    torch.manual_seed(seed)
    out_dir = Path(".research") / "iteration12" / exp_conf.id / f"{variant}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = CarbonMonitor(out_dir / "energy.json")

    # -------------------------- model ------------------------------------ #
    control_on = variant == "v0"
    model = RaptorDiffuser(
        model_id=exp_conf.model["base"],
        fisher_conf=exp_conf.hutch,
        sched_conf=exp_conf.scheduler,
        control_var=control_on,
    )

    # --------------------------- data ------------------------------------ #
    train_ds = ImageTokenDataset(hf_name=exp_conf.data["target"]["url"], split="train", vq_encoder=model.pipe.vae)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.pipe.unet.parameters(), lr=exp_conf.training.lr_finetune)

    # ------------------------ training loop ------------------------------ #
    monitor.sample()
    device = next(model.pipe.unet.parameters()).device
    for epoch in range(exp_conf.training.finetune_epochs):
        for batch in tqdm(train_dl, desc=f"{variant}-E{epoch}", leave=False):
            batch = batch.to(device)
            with autocast():
                loss = model.pipe.unet(batch)  # surrogate loss placeholder
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            monitor.sample()
        torch.save(model.pipe.unet.state_dict(), out_dir / f"ckpt_{epoch}.pt")

    energy = monitor.stop()

    # -------------------------- evaluation ------------------------------- #
    from .evaluate import evaluate_and_plot  # delayed import to avoid circularity

    metrics = evaluate_and_plot(model, exp_conf, out_dir)
    metrics.update(energy)

    with (out_dir / "result.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))


__all__ = [
    "ExperimentConf",
    "load_yaml",
    "fit",
]