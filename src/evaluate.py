"""src/evaluate.py
====================
Utility helpers used during the experiments:
• EnergyMeter         – context manager measuring wall-clock latency and
                         estimating energy (very rough!).
• MetricsAggregator   – running mean/std per metric  
• dp_epsilon          – closed-form RDP → (ε,δ) conversion for Gaussian noise
• plot_and_save_all   – quick-and-dirty Matplotlib visualisation
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

__all__ = [
    "EnergyMeter",
    "MetricsAggregator",
    "dp_epsilon",
    "plot_and_save_all",
]


# -----------------------------------------------------------------------------
# 1. Energy / latency measurement
# -----------------------------------------------------------------------------
class EnergyMeter:
    """Fail-safe *approximate* energy meter (CPU-only fallback)."""

    def __init__(self, device) -> None:  # device is str/torch.device
        self.device = str(device)
        self.t0: float = 0.0
        self.latency_ms: float = 0.0
        self.energy_J: float = 0.0

    # ------------------------------------------------------------------
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        dt = time.perf_counter() - self.t0
        self.latency_ms = dt * 1e3
        # Toy "energy" model:  2 W * dt   (typical laptop CPU-only)
        self.energy_J = 2.0 * dt
        return False  # re-raise exceptions if any


# -----------------------------------------------------------------------------
# 2. Online aggregator – mean & list per metric
# -----------------------------------------------------------------------------
class MetricsAggregator:
    def __init__(self):
        self._values: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    def update(self, key: str, value: float) -> None:
        self._values.setdefault(key, []).append(float(value))

    # ------------------------------------------------------------------
    def mean(self, key: str) -> float:
        v = self._values.get(key, [0.0])
        return sum(v) / max(1, len(v))

    # ------------------------------------------------------------------
    def std(self, key: str) -> float:
        v = self._values.get(key, [0.0])
        m = self.mean(key)
        return math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v)))


# -----------------------------------------------------------------------------
# 3. Differential-privacy accountant (Gaussian) – extremely small utility
# -----------------------------------------------------------------------------
#   ε(q,σ,steps,δ) ≈ steps * q^2 / (2σ^2)   (worst-case analytical bound)
# -----------------------------------------------------------------------------

def dp_epsilon(*, sigma: float, q: float, steps: int, delta: float = 1e-6) -> float:  # noqa: D401
    eps_rdp = steps * (q**2) / (2 * sigma**2)
    # Convert RDP → (ε,δ) : for Gaussian, ε ≈ RDP + 2 ⋅ √(RDP ⋅ ln(1/δ))
    eps = eps_rdp + 2 * math.sqrt(eps_rdp * math.log(1 / delta))
    return float(eps)


# -----------------------------------------------------------------------------
# 4. Simple plotting util – saves one PDF per metric
# -----------------------------------------------------------------------------

IMG_DIR_NAME = "images"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_and_save_all(all_results: List[Dict], results_dir: Path) -> None:
    img_dir = results_dir / IMG_DIR_NAME
    _ensure_dir(img_dir)

    # ---------------------------- Accuracy curve ----------------------------
    exp1 = next(r for r in all_results if r["experiment"] == "exp1")
    for method, curve in exp1["accuracy_curve"].items():
        plt.plot(curve["ts"], curve["values"], label=method)
    plt.title("Experiment-1 Accuracy vs Time (min)")
    plt.xlabel("Minutes")
    plt.ylabel("Accuracy")
    plt.legend()
    out_path = img_dir / "exp1_accuracy_curve.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------- Energy vs Budget ----------------------------
    exp3 = next(r for r in all_results if r["experiment"] == "exp3")
    plt.plot(exp3["energy_vs_B"]["B"], exp3["energy_vs_B"]["E"], marker="o")
    plt.title("Experiment-3 Energy vs Memory Budget")
    plt.xlabel("Budget (bytes)")
    plt.ylabel("Energy (J)")
    out_path = img_dir / "exp3_energy_vs_budget.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------ dump a tiny JSON manifest so that CI can verify plot presence ----
    manifest = {
        "figures": sorted([p.name for p in img_dir.glob("*.pdf")]),
        "dir": str(img_dir.resolve()),
    }
    (img_dir / "fig_manifest.json").write_text(json.dumps(manifest, indent=2))
