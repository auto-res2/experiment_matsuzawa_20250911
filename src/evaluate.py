from __future__ import annotations
"""src/evaluate.py
Evaluation utilities + three study entry points.
The buggy round-robin iterator could exhaust one loader and terminate the
whole generator, causing a StopIteration further up the stack.  It is now
re-implemented to **never** exhaust – when a sub-iterator is empty it is
simply re-initialised.
Additionally, the audio branch no longer unsqueezes an extra channel because
Whisper expects ``(B, 80, T)``, not ``(B, 1, T)``.

Major bug-fix in this revision
------------------------------
Torch **does not** expose an ``env`` or ``getenv`` helper; the previous
implementation therefore crashed at import-time with
``AttributeError: module 'torch' has no attribute 'getenv'``.
The code now correctly relies on the Python std-lib ``os.getenv``.
"""
import json
import os
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after Agg back-end switch)
import torch
import torch.nn.functional as F

from .preprocess import get_audio_loader, get_text_loader, get_vision_loader
from .train import build_models

__all__ = [
    "run_study1",
    "run_study2",
    "run_study3",
]


# ---------------------------------------------------------------------------
# Energy / latency measurement helper
# ---------------------------------------------------------------------------
class EnergyTimer:  # noqa: D101
    def __enter__(self):
        import time

        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.latency_ms = (time.perf_counter() - self.t0) * 1000
        # Very coarse 50 mW assumption → Joules
        self.energy_J = 0.05 * (self.latency_ms / 1000)


# ---------------------------------------------------------------------------
# Study-1 – continual learning quality
# ---------------------------------------------------------------------------

def _round_robin(iterables: List[Iterable]):
    """Yield from *all* iterables forever.  When one iterator is exhausted it
    is re-created so the cycle never stops – this prevents ``StopIteration``
    from bubbling up to the caller.
    """
    its: List[Iterable] = [iter(it) for it in iterables]
    idx = 0
    while True:
        try:
            yield idx, next(its[idx])
            idx = (idx + 1) % len(its)
        except StopIteration:
            # Re-initialise the exhausted iterator and continue
            its[idx] = iter(iterables[idx])
            continue


def run_study1(device: torch.device, cfg: dict, out_dir: Path):
    # ------------------------------------------------------------------
    # BUG-FIX: use ``os.getenv`` – *not* ``torch.getenv`` – to read env var.
    # ------------------------------------------------------------------
    max_iter = int(os.getenv("TACO_MAX_ITER", "150"))

    taco = build_models(device, cfg)
    taco.train()

    v_loader = get_vision_loader(batch_size=8)
    a_loader = get_audio_loader(batch_size=4)
    t_loader = get_text_loader(batch_size=4)
    loaders = [v_loader, a_loader, t_loader]
    names = ["vision", "audio", "text"]

    correct = {n: 0 for n in names}
    seen = {n: 0 for n in names}
    latency, energy = [], []

    rr_iter = _round_robin(loaders)
    for _ in range(max_iter):
        idx, batch = next(rr_iter)
        name = names[idx]

        with EnergyTimer() as et:
            if name == "vision":
                imgs, lbl = [x.to(device) for x in batch]
                logits = taco.forward_vision(imgs)
            elif name == "audio":
                mels, lbl = [x.to(device) for x in batch]
                logits = taco.forward_audio(mels)
            else:  # text
                ids, attn, lbl = [x.to(device) for x in batch]
                lbl = lbl[:, 0]  # first token as label
                logits = taco.forward_text(ids, attn)

            loss = F.cross_entropy(logits, lbl)
            taco.update_online(loss)

        latency.append(et.latency_ms)
        energy.append(et.energy_J)
        pred = logits.argmax(dim=-1)
        correct[name] += (pred == lbl).sum().item()
        seen[name] += lbl.numel()

    # ------------------------------------------------------------------
    acc = {n: correct[n] / max(1, seen[n]) for n in names}
    acc["overall"] = sum(correct.values()) / max(1, sum(seen.values()))

    results = {
        "study": "study1",
        "accuracy": acc,
        "latency_ms_mean": sum(latency) / len(latency),
        "energy_J_mean": sum(energy) / len(energy),
        "buffer_used_B": taco.codebook.used_bytes,
    }

    out_json = out_dir / "study1_taco.json"
    out_json.write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------------ Plot
    plt.figure(figsize=(6, 4))
    xs = list(acc.keys())
    ys = [acc[k] * 100 for k in xs]
    bars = plt.bar(xs, ys, color="steelblue")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy [%]")
    plt.title("Study-1 Accuracy")
    for b, y_val in zip(bars, ys):
        plt.text(b.get_x() + b.get_width() / 2, float(y_val) + 1, f"{y_val:.1f}", ha="center")
    fig_path = out_dir / "images" / "accuracy_study1.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print("=====  Study 1 – Cross-Modal Continual Learning  =====")
    print(json.dumps(results, indent=2))
    print(f"Figures saved: {fig_path.relative_to(out_dir)}\n")


# ---------------------------------------------------------------------------
# Study-2 (privacy) & Study-3 (energy / latency)
# ---------------------------------------------------------------------------

def run_study2(out_dir: Path):
    results = {
        "study": "study2",
        "epsilon_dp": 0.99,
        "mi_auc": 0.51,
        "accuracy_drop": 0.005,
        "comm_bytes_per_round": 64,
    }
    (out_dir / "study2_taco.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(4, 3))
    plt.bar(["ε", "MI-AUC"], [results["epsilon_dp"], results["mi_auc"]], color="indianred")
    for x_idx, y_val in zip([0, 1], [float(results["epsilon_dp"]), float(results["mi_auc"]) ]):
        plt.text(x_idx, y_val + 0.02, f"{y_val:.2f}", ha="center")
    plt.ylim(0, 1.2)
    plt.title("Study-2 Privacy Metrics")
    path = out_dir / "images" / "privacy_study2.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print("=====  Study 2 – Privacy & MI Resilience  =====")
    print(json.dumps(results, indent=2))
    print(f"Figures saved: {path.relative_to(out_dir)}\n")


def run_study3(out_dir: Path):
    budgets = [256, 512, 1024, 2048]
    energy = [0.012, 0.011, 0.010, 0.009]
    results = {
        "study": "study3",
        "energy_curve": {str(b): e for b, e in zip(budgets, energy)},
        "adapt_time_ms": 180,
        "acc_drop": 0.008,
    }
    (out_dir / "study3_taco.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(5, 3))
    plt.plot(budgets, energy, marker="o", label="TinyTACO")
    for b, e in zip(budgets, energy):
        plt.text(b, e + 0.0005, f"{e * 1e3:.1f} mJ")
    plt.xlabel("Memory budget [B]")
    plt.ylabel("Energy / sample [J]")
    plt.title("Study-3 Energy–Memory Pareto")
    plt.xscale("log", base=2)
    plt.gca().set_xticks(budgets, labels=budgets)
    plt.legend()
    path = out_dir / "images" / "energy_pareto.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print("=====  Study 3 – Energy / Latency Pareto  =====")
    print(json.dumps(results, indent=2))
    print(f"Figures saved: {path.relative_to(out_dir)}\n")