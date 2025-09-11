# src/train.py
"""Model code, configuration dataclasses and checkpoint utilities.
This file concentrates everything that is related to *models* or *model
handling* so that the rest of the project (pre-processing, evaluation,
entry-point) can simply import what is required from here.
"""
from __future__ import annotations

import pathlib
import typing as _t
from dataclasses import dataclass, field

import requests
import yaml
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
#  CONFIGURATION  
# -----------------------------------------------------------------------------

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"

# -----------------------------------------------------------------------------
#  Dataclass hierarchy that mirrors the YAML structure
# -----------------------------------------------------------------------------


@dataclass
class DatasetCfg:
    hf_id: str
    split_train: str
    split_test: str
    # Optional fields ----------------------------------------------------------
    transform: str = ""
    config: str = ""


@dataclass
class ModelCfg:
    url: str
    local_name: str


@dataclass
class SamplerCfg:
    epsilon_list: list[float] = field(default_factory=list)
    kappa_list: list[float] = field(default_factory=list)
    tau0_list: list[float] = field(default_factory=list)
    gamma_list: list[float] = field(default_factory=list)
    steps_distilled: list[int] = field(default_factory=list)
    steps_nodistill: list[int] = field(default_factory=list)


@dataclass
class GlobalConfig:
    meta: dict[str, _t.Any]
    hardware: dict[str, _t.Any]
    datasets: dict[str, DatasetCfg]
    models: dict[str, ModelCfg]
    sampler: SamplerCfg
    experiments: list[dict[str, _t.Any]]


# -----------------------------------------------------------------------------
#  YAML ⇆ Dataclass helpers
# -----------------------------------------------------------------------------


def _dict_to_dataclass(mapping: dict[str, dict], cls):
    """Utility that converts nested dictionaries to dataclass instances."""
    return {k: cls(**v) for k, v in mapping.items()}


def load_config(yaml_path: pathlib.Path) -> GlobalConfig:
    """Parse the YAML configuration file located at *yaml_path*."""
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Configuration file {yaml_path.as_posix()} could not be found.")

    with yaml_path.open() as fp:
        data = yaml.safe_load(fp)

    datasets = _dict_to_dataclass(data["datasets"], DatasetCfg)
    models = _dict_to_dataclass(data.get("models", {}), ModelCfg)
    sampler = SamplerCfg(**data["sampler"])
    return GlobalConfig(
        meta=data["meta"],
        hardware=data["hardware"],
        datasets=datasets,
        models=models,
        sampler=sampler,
        experiments=data["experiments"],
    )


def dump_yaml(src_path: pathlib.Path, dst_path: pathlib.Path):
    """Copy the YAML file to *dst_path* for provenance tracking."""
    try:
        dst_path.write_text(src_path.read_text())
    except Exception as exc:  # pragma: no cover
        print(f"[WARNING] Could not dump YAML for provenance: {exc}")


# -----------------------------------------------------------------------------
#  Model checkpoint utilities  
# -----------------------------------------------------------------------------

_MODEL_ROOT = pathlib.Path("models")
_MODEL_ROOT.mkdir(exist_ok=True)


def _download(url: str, dst: pathlib.Path):
    dst_tmp = dst.with_suffix(".tmp")
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Could not download checkpoint from {url} (HTTP {resp.status_code}).")
    total = int(resp.headers.get("content-length", 0))
    with open(dst_tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    dst_tmp.rename(dst)


def get_checkpoint(cfg: ModelCfg) -> pathlib.Path:
    """Ensure that *cfg.local_name* exists locally and return its path."""
    ckpt_path = _MODEL_ROOT / cfg.local_name
    if ckpt_path.exists():
        return ckpt_path
    print(f"Checkpoint {cfg.local_name} missing – attempting download …", flush=True)
    try:
        _download(cfg.url, ckpt_path)
    except Exception as e:
        raise RuntimeError(
            f"Unable to fetch the required model checkpoint {cfg.local_name}.\n{e}\n"
        )
    return ckpt_path


# -----------------------------------------------------------------------------
#  Adaptive Probability-Flow Sampler (placeholder)
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

__all__ = [
    "DatasetCfg",
    "ModelCfg",
    "SamplerCfg",
    "GlobalConfig",
    "load_config",
    "dump_yaml",
    "get_checkpoint",
    "APFSampler",
]


class _NotAvailableError(RuntimeError):
    """Raised when the full numerical solver is not present."""


class APFSampler(nn.Module):
    """Adaptive Probability-Flow sampler – *interface only*.

    A complete implementation requires a sophisticated numerical solver
    that is outside the scope of this refactor.  The class therefore
    validates that the requested networks are present and then raises a
    descriptive *_NotAvailableError* when `forward()` is called.
    """

    def __init__(
        self,
        teacher_network: nn.Module,
        order: str = "voei",
        distilled: bool = False,
        student_network: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher_network
        self.order = order
        self.distilled = distilled
        self.student = student_network if distilled else None
        if self.distilled and self.student is None:
            raise _NotAvailableError(
                "Distilled sampling requested but student network could not be loaded."
            )

    # ---------------------------------------------------------------------
    def forward(self, z_T: torch.Tensor, steps: int) -> torch.Tensor:  # noqa: N802
        raise _NotAvailableError(
            "Full APF sampler not yet implemented – please integrate the solver "
            "from the proprietary research repository."
        )
