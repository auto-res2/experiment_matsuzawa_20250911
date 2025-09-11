"""
src/evaluate.py
================
Evaluation utilities, plotting helpers and GPU-power instrumentation that were
previously embedded in the single-file experiment script.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from huggingface_hub import list_repo_files

# Force head-less backend – plots are saved directly to PDF.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# 1. Strict resource checker – ensure a model repository actually contains
#    weight files before we attempt to download / instantiate it.
# ---------------------------------------------------------------------------

def ensure_hf_model_exists(repo_id: str, revision: str | None = None) -> None:
    """Abort execution if *any* of the requested model files are missing on the Hub."""
    try:
        files = list_repo_files(repo_id, revision=revision)
    except Exception as exc:  # pragma: no cover – network / Hub errors are fatal.
        raise RuntimeError(
            f"Model repository '{repo_id}' not accessible on the HuggingFace Hub.\n"
            f"Original error: {exc}"
        ) from exc

    if not any(f.endswith((".bin", ".safetensors")) for f in files):
        raise RuntimeError(
            f"Model repository '{repo_id}' does not contain weight files (*.bin | *.safetensors)."
        )

# ---------------------------------------------------------------------------
# 2. GPU-power sampler – used to measure instantaneous energy draw during
#    inference / training.  Falls back gracefully when NVML is unavailable.
# ---------------------------------------------------------------------------

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlShutdown,
    )

    _NVML_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – CPU-only environments.
    _NVML_AVAILABLE = False


def sample_gpu_power(interval_s: float, stop_event, device_idx: int = 0):
    """Continuously sample GPU power usage (Watts) until *stop_event* is set.

    Parameters
    ----------
    interval_s : float
        Sampling interval in seconds.
    stop_event : threading.Event | torch.cuda.Event | Any
        Event-like object whose ``is_set()``/``query()`` method signals measurement
        termination.
    device_idx : int, default = 0
        GPU index.
    """
    if not _NVML_AVAILABLE:
        raise RuntimeError("pynvml is not installed – GPU power sampling is unavailable.")

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_idx)

    readings: list[tuple[float, float]] = []  # (seconds, Watts)
    start_time = time.time()

    # Accept both threading.Event (is_set) and torch.cuda.Event (query)
    is_finished = stop_event.is_set if hasattr(stop_event, "is_set") else stop_event.query

    try:
        while not is_finished():
            power_mw = nvmlDeviceGetPowerUsage(handle)  # returns mW
            readings.append((time.time() - start_time, power_mw / 1000.0))
            time.sleep(interval_s)
    finally:
        nvmlShutdown()

    return readings

# ---------------------------------------------------------------------------
# 3. Simple line-plot helper.
# ---------------------------------------------------------------------------

def save_line_plot(
    xs: List[float],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    """Save a labelled line plot as PDF and return the file path."""
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)

    for x_val, y_val in zip(xs, ys):
        plt.annotate(f"{y_val:.2f}", (x_val, y_val))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    Path("figures").mkdir(exist_ok=True)
    pdf_path = Path("figures") / filename
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.close()

    return str(pdf_path)

__all__ = [
    "ensure_hf_model_exists",
    "sample_gpu_power",
    "save_line_plot",
]
