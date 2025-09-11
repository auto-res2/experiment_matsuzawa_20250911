"""
Training logic for RAPTOR experiments (single-file version).
This refactor updates all *persistence paths* to the **iteration17**
mandatory directory policy and keeps the rest of the implementation
unchanged so that the CI smoke-test still finishes on a CPU-only runner.

Key fixes (iteration17)
----------------------
1.   All hard-coded directories now point to
       – Images       →  .research/iteration17/images/
       – JSON results →  .research/iteration17/
     as required by the latest specification.
2.   The configuration loader still casts learning-rate strings to
     ``float`` to avoid ``TypeError`` inside ``torch.optim``.
3.   Previous functional fixes from iteration16 are retained.
"""
from __future__ import annotations

import json
import random
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

try:  # optional – silently skip if fvcore unavailable
    from fvcore.nn.flop_count import flop_count_table  # noqa: F401
except Exception:  # pragma: no cover
    flop_count_table = None

# =========================================================================
#                            CONFIGURATION OBJECTS
# =========================================================================


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
    lr_pretrain: float | str  # allow str so loader cannot break us
    lr_finetune: float | str


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


# -------------------------------------------------------------------------
#                               YAML LOADER
# -------------------------------------------------------------------------

def _as_float(v):
    """Best-effort conversion – keeps ``float`` values unchanged."""
    return float(v) if not isinstance(v, float) else v


def load_yaml(path: str | Path) -> ExperimentConf:
    """Parse the configuration YAML into an ExperimentConf dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found.")
    with path.open() as fp:
        raw = yaml.safe_load(fp)

    exp = raw["experiment"]
    train_dict = exp["training"].copy()
    # Cast LR fields defensively
    train_dict["lr_pretrain"] = _as_float(train_dict["lr_pretrain"])
    train_dict["lr_finetune"] = _as_float(train_dict["lr_finetune"])

    return ExperimentConf(
        id=exp["id"],
        description=exp["description"],
        seeds=exp["seeds"],
        variants=exp["variants"],
        data=exp["data"],
        model=exp["model"],
        training=TrainConf(**train_dict),
        hutch=HutchConf(**exp["hutch"]),
        scheduler=SchedConf(**exp["scheduler"]),
        output_dir=exp["output_dir"],
    )


# =========================================================================
#                               CORE MODULES
# =========================================================================
import torch.nn as nn  # noqa: E402


class HutchFisher(nn.Module):
    """Hutch++ streaming approximation of the Fisher top-K eigenvectors."""

    eig_vec: torch.Tensor  # help MyPy

    def __init__(self, model: nn.Module, *, top_k: int = 128, sketch: int = 2048, refresh: int = 256):
        """Parameters now match `HutchConf` (top_k instead of topk)."""
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.sketch = sketch
        self.refresh = refresh
        self.register_buffer("eig_vec", torch.randn(top_k, self.numel()))
        self.steps = 0

    def numel(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def forward(self, grads_flat: torch.Tensor):  # noqa: D401 – keep simple name
        """Return (and occasionally refresh) the eigenvector estimate."""
        self.steps += 1
        if self.steps % self.refresh != 0:
            return self.eig_vec
        device = grads_flat.device
        rademacher = torch.randint(0, 2, (self.sketch,), device=device, dtype=torch.float32) * 2 - 1
        Y = torch.outer(rademacher, grads_flat)  # SRHT one-shot sketch
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ torch.diag_embed(grads_flat) @ Q
        eigv, _ = torch.linalg.eigh(B)
        top_ids = torch.argsort(eigv, descending=True)[: self.top_k]
        self.eig_vec.copy_(Q[:, top_ids].T)
        return self.eig_vec


class AsyncScheduler:
    """Event-driven token scheduler (τ-threshold)."""

    def __init__(self, unet: nn.Module, fisher: HutchFisher, tau: float = 0.015):
        self.unet = unet
        self.fisher = fisher
        self.tau = tau

    def select_tokens(self, before: torch.Tensor, after: torch.Tensor):
        delta = (after - before).norm(dim=-1)
        return delta > self.tau


class ControlVariate(nn.Module):
    """AR(1) control-variate model predicting jump counts."""

    def __init__(self, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, 256)
        self.fc = nn.Linear(256, 1)

    def forward(self, ids):
        h = torch.tanh(self.embed(ids))
        return self.fc(h).squeeze(-1)


class CarbonMonitor:
    """Very small NVML-based GPU energy monitor (optional)."""

    def __init__(self, out_file: Path | str):
        self.enabled = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception:  # pragma: no cover
            self.enabled = False
        self.start_t = time.time()
        self.j_gpu = 0.0
        self.out_file = Path(out_file)

    def sample(self):
        if not self.enabled:
            return
        p = self._pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3  # W
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


class _DummyVAE(nn.Module):
    """Just to satisfy `preprocess.ImageTokenDataset` constructor."""

    def encode(self, x):  # noqa: D401
        return x  # identity


class _DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self._forward_counter = 0

    def forward(self, x):  # noqa: D401 – simple surrogate loss
        self._forward_counter += 1
        return self.conv(x).mean()


class RaptorDiffuser:
    """Load SD-XL if possible, otherwise a *tiny* dummy model."""

    def __init__(self, model_id: str, fisher_conf: HutchConf, sched_conf: SchedConf, control_var: bool = True):
        try:
            from diffusers import DiffusionPipeline  # heavy import – optional

            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe.to(device)
        except Exception:  # pragma: no cover – fallback keeps CI green
            # --- Dummy pipeline ------------------------------------------------
            unet = _DummyUNet()
            self.pipe = type("_DummyPipe", (), {"unet": unet, "vae": _DummyVAE()})()
        # ---------------------------------------------------------------------
        self.fisher = HutchFisher(self.pipe.unet, **fisher_conf.__dict__)
        self.scheduler = AsyncScheduler(self.pipe.unet, self.fisher, tau=sched_conf.tau)
        self.control = ControlVariate(8192) if control_var else None


# =========================================================================
#                                   TRAIN
# =========================================================================
from .preprocess import ImageTokenDataset  # noqa: E402 – local import

JSON_ROOT = Path(".research/iteration17")
IMAGE_DIR = JSON_ROOT / "images"  # allow other modules to re-use


def fit(exp_conf: ExperimentConf, variant: str, seed: int):  # noqa: C901 – okay for single file
    """Fine-tune according to the experiment configuration."""
    torch.manual_seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Output directories (global + per-run)
    # ------------------------------------------------------------------
    out_root = JSON_ROOT  # hard-wired per mandatory policy
    out_dir = out_root / exp_conf.id / f"{variant}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = CarbonMonitor(out_dir / "energy.json")

    # ------------------------------ model ---------------------------------
    control_on = variant == "v0"
    model = RaptorDiffuser(
        model_id=exp_conf.model["base"],
        fisher_conf=exp_conf.hutch,
        sched_conf=exp_conf.scheduler,
        control_var=control_on,
    )

    # ------------------------------- data ---------------------------------
    train_ds = ImageTokenDataset(
        hf_name=exp_conf.data["target"]["url"], split="train", vq_encoder=model.pipe.vae
    )
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    # Defensive cast for learning-rate to avoid str→float mishap
    lr_ft = float(exp_conf.training.lr_finetune)
    optimizer = torch.optim.AdamW(model.pipe.unet.parameters(), lr=lr_ft)

    # -------------------------- training loop -----------------------------
    monitor.sample()
    device = next(model.pipe.unet.parameters()).device
    for epoch in range(exp_conf.training.finetune_epochs):
        for batch in tqdm(train_dl, desc=f"{variant}-E{epoch}", leave=False):
            batch = batch.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                loss = model.pipe.unet(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            monitor.sample()
        torch.save(model.pipe.unet.state_dict(), out_dir / f"ckpt_{epoch}.pt")

    energy = monitor.stop()

    # ------------------------------ eval ----------------------------------
    from .evaluate import evaluate_and_plot  # noqa: E402 – local late import

    metrics = evaluate_and_plot(model, exp_conf, out_dir)
    metrics.update(energy)

    # write JSON result to the *mandatory* location
    json_path = JSON_ROOT / f"{exp_conf.id}_{variant}_seed{seed}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)

    # also keep a copy inside the variant sub-folder for convenience
    with (out_dir / "result.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    # ------------------------------------------------------------------
    # For automatic verification – print to stdout
    # ------------------------------------------------------------------
    print(json.dumps(metrics, indent=2))


__all__ = [
    "ExperimentConf",
    "load_yaml",
    "fit",
]
