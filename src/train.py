"""src/train.py
All experiment-specific logic (training / compilation / evaluation helpers) lives here.
This file is a direct refactor of the original single-script experiment code – **no
behavioural changes have been made except for:
  •   Adhering to the repository-wide path convention requested in the
      remediation guidelines (JSON → .research/iteration6/, images →
      .research/iteration6/images).
"""
from __future__ import annotations

import json, os, sys, time, random, pathlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Callable, cast

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from sacrebleu import corpus_bleu

# ---------------------------------------------------------------------------
# Optional (vendor) dependency – provide graceful fallback
# ---------------------------------------------------------------------------
try:
    from conductor import AutoComposer, SyBayes, DOSD, LiCCA, CoFaD, HiRRB
    from conductor.metrics import tm_fid, pass_at_k, leqpc_auc
except ImportError:  # pragma: no cover – keep import-time overhead minimal

    class _MissingConductorPackage(Exception):
        """Raised when a required CONDUCTOR component is accessed without the
        optional `conductor-ai` package being installed. We *fail fast* instead
        of silently degrading functionality so that the user sees an immediate
        and clear error message.
        """

    def _raise_missing(*_args: Any, **_kwargs: Any):
        raise _MissingConductorPackage(
            "The optional `conductor-ai` package is not installed. Install it via\n"
            "    pip install conductor-ai>=0.5.1\n"
            "or remove CONDUCTOR-specific functionality from the experiment run."
        )

    # Stubs that raise at *instantiation* time – keeps static analysis happy but
    # guarantees a hard failure if the code path is executed at runtime.
    class _ConductorStub:  # pylint: disable=too-few-public-methods
        def __init__(self, *args: Any, **kwargs: Any):
            _raise_missing()

        # catch any attribute access on an already (not) constructed instance
        def __getattr__(self, _name: str):
            _raise_missing()

        def __call__(self, *args: Any, **kwargs: Any):
            _raise_missing()

    # Cast to Any so that type checkers accept the assignment without ignores
    AutoComposer = SyBayes = DOSD = LiCCA = CoFaD = HiRRB = cast(Any, _ConductorStub)
    tm_fid = pass_at_k = leqpc_auc = cast(Any, _raise_missing)

# ---------------------------------------------------------------------------
from pynvml import (  # noqa: E402 – external dep that may not exist on all hosts
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage,
)

# -----------------------------------------------------------------------------
# CONSTANTS & PATHS (loaded/overridden by src.main)
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
# All JSON artefacts must live directly under “.research/iteration6/”
ART_DIR = ROOT / ".research" / "iteration6"
IMG_DIR = ART_DIR / "images"
ART_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# BASIC UTILITIES
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def abort(msg: str):
    print(f"[FATAL] {msg}")
    sys.exit(1)


class PowerLogger:
    """GPU power sampler (@250 ms) using NVML. Handles missing NVML gracefully."""

    def __init__(self):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.active = True
        except Exception as e:  # pragma: no cover
            print("[WARN] NVML unavailable – energy metrics disabled:", e)
            self.active = False
        self.ts, self.p = [], []

    def sample(self):
        if not self.active:
            return
        try:
            self.ts.append(time.time())
            self.p.append(nvmlDeviceGetPowerUsage(self.handle) / 1000.0)  # mW → W
        except Exception:  # pragma: no cover
            pass  # ignore transient NVML failures

    def energy_j(self) -> float:
        if not self.active or len(self.ts) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(self.ts)):
            dt = self.ts[i] - self.ts[i - 1]
            e += self.p[i] * dt
        return e

    def close(self):
        if self.active:
            try:
                nvmlShutdown()
            except Exception:  # pragma: no cover
                pass

# -----------------------------------------------------------------------------
# FIGURE / PLOTTING HELPERS
# -----------------------------------------------------------------------------

def line_plot(
    x: List[float],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: pathlib.Path,
):
    """Thin wrapper around seaborn line-plot that guarantees the file is written
    underneath the mandated .research/iteration6/images directory."""
    filename = IMG_DIR / filename.with_suffix("").name  # enforce directory
    filename = filename.with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x, y=y, marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([title])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------
# SUMMARY CONTAINER
# -----------------------------------------------------------------------------

@dataclass
class Summary:
    exp_id: int
    seed: int
    condition: str
    primary: Dict[str, float]
    secondary: Dict[str, float]
    timestamp: str

    def save(self, path: pathlib.Path):
        """All summaries must be stored directly inside .research/iteration6."""
        path = ART_DIR / path.name  # enforce location
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

# -----------------------------------------------------------------------------
# PREPROCESS / DATALOADERS  – re-export from src.preprocess for convenience
# -----------------------------------------------------------------------------
from src.preprocess import (
    load_wmt22_en_de,
    load_protein_cath,
    load_humaneval,
)

# -----------------------------------------------------------------------------
# EXPERIMENT 1  – End-to-End Efficiency & Carbon Pay-Back
# -----------------------------------------------------------------------------

def _task_table(default_yaml) -> List[tuple]:
    """Returns the task configuration table used by Exp-1."""
    return [
        (
            "mt_en_de",
            default_yaml["models"]["bitdiff_t5_l"],
            load_wmt22_en_de,
            lambda p, r: {"BLEU": corpus_bleu(p, [r]).score},
        ),
        (
            "protein_cath",
            default_yaml["models"]["rfdiffusion_xl"],
            load_protein_cath,
            lambda p, r: {"TM_FID": tm_fid(p, r)},
        ),
        (
            "code_humaneval",
            default_yaml["models"]["absorbing_gpt3_1b3"],
            load_humaneval,
            lambda p, r: pass_at_k(p, r, k=[1, 10]),
        ),
    ]


def run_exp1(seed: int, default_yaml: Dict[str, Any]):
    set_seed(seed)
    exp_dir = ART_DIR / "exp1" / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_objects: List[Summary] = []

    conditions = {
        "baseline": {"comp": "single_axis"},
        "baseline_dndm": {"comp": "single_axis_dndm"},
        "conductor": {"comp": "full"},
    }

    for task_name, model_ckpt, ds_fn, metric_fn in _task_table(default_yaml):
        # strict model load
        try:
            model = torch.hub.load(model_ckpt, trust_repo=True)
        except Exception as e:  # pragma: no cover
            abort(f"Could not load model {model_ckpt}: {e}")

        dataset = ds_fn()
        n_eval = 10000 if len(dataset) > 10000 else len(dataset)
        dataset = dataset.select(range(n_eval))

        for cond_name, cond_cfg in conditions.items():
            cond_dir = exp_dir / cond_name
            cond_dir.mkdir(parents=True, exist_ok=True)
            power = PowerLogger(); power.sample()
            preds, refs, latencies = [], [], []

            # ---------------------------------------------------------------
            # compile / wrap
            # ---------------------------------------------------------------
            if cond_cfg["comp"] == "single_axis":
                compiled = model
            elif cond_cfg["comp"] == "single_axis_dndm":
                compiled = DOSD.wrap(model, offline_distill=True, steps=8)
            else:
                syb = SyBayes(cache_path="sybayes_a100.json")
                planner = AutoComposer(model, sybayes=syb)
                plan = planner.solve({"latency_ms": 50, "energy_j": 30})
                compiled = planner.compile(plan)
                compiled = DOSD.wrap(compiled)
                compiled = CoFaD.wrap(compiled, target_dp=0.3)
                compiled = LiCCA.wrap(compiled)

            compiled.eval()
            power.sample()  # t0

            for sample in tqdm(dataset, desc=f"{task_name}-{cond_name}"):
                if task_name == "mt_en_de":
                    src = sample["translation"]["en"]
                    tgt = sample["translation"]["de"]
                elif task_name == "protein_cath":
                    src = sample["seq"]
                    tgt = sample["seq"]
                else:
                    src = sample["prompt"]
                    tgt = sample["canonical_solution"]

                st = time.perf_counter()
                with torch.no_grad():
                    out = compiled.generate(src)
                latencies.append((time.perf_counter() - st) * 1000)
                preds.append(out if isinstance(out, str) else out[0])
                refs.append(tgt)
                if len(latencies) % 25 == 0:
                    power.sample()

            power.sample(); energy_j = power.energy_j(); power.close()

            primary = metric_fn(preds, refs)
            secondary = {
                "latency_median_ms": float(np.median(latencies)),
                "latency_p99_ms": float(np.percentile(latencies, 99)),
                "energy_j": energy_j,
            }
            summ = Summary(1, seed, f"{task_name}-{cond_name}", primary, secondary, datetime.utcnow().isoformat())
            summ_path = exp_dir / f"summary_{task_name}_{cond_name}.json"; summ.save(summ_path)

            # figure (always in IMG_DIR)
            fig_name = pathlib.Path(f"exp1_seed{seed}_{task_name}_{cond_name}_latency")
            line_plot(list(range(len(latencies))), latencies, "sample_idx", "latency (ms)", f"Latency – {task_name} – {cond_name}", fig_name)
            print(f"\n===== EXP-1 {task_name}/{cond_name} =====")
            print(json.dumps(asdict(summ), indent=2))
            print("Figure saved:", fig_name.with_suffix('.pdf').name)
            summary_objects.append(summ)

    # LEQPC-AUC
    all_energy = sum(s.secondary["energy_j"] for s in summary_objects)
    all_latency = np.array([s.secondary["latency_median_ms"] for s in summary_objects])
    leqpc = leqpc_auc(all_latency, all_energy)
    print("\nOverall LEQPC-AUC:", leqpc)

# -----------------------------------------------------------------------------
# EXPERIMENT 2  – SyBayes Accuracy & Planner Scalability
# -----------------------------------------------------------------------------

def run_exp2(seed: int, default_yaml: Dict[str, Any]):
    set_seed(seed)
    exp_dir = ART_DIR / "exp2" / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    chips = ["A100", "JetsonOrin", "Inferentia2", "Pixel8Pro"]
    models = [
        default_yaml["models"]["bitdiff_t5_l"],
        default_yaml["models"]["vit_diffuser_900m"],
    ]

    for chip in chips:
        for model_ckpt in models:
            tag = f"{chip}_{pathlib.Path(model_ckpt).name}"
            try:
                model = torch.hub.load(model_ckpt, trust_repo=True)
            except Exception as e:
                abort(f"Model unavailable {model_ckpt}: {e}")

            probe_file = exp_dir / f"probes_{tag}.pkl"
            if not probe_file.exists():
                abort("Probe collection must be run on-device; file missing: " + str(probe_file))
            probes = torch.load(probe_file)

            t0 = time.time()
            syb = SyBayes.fit(probes, max_time_min=30)
            train_wall = time.time() - t0
            holdout = probes.sample_holdout(1000)
            metrics = SyBayes.evaluate(syb, holdout)
            metrics.update({"train_wall_s": train_wall})
            out_path = exp_dir / f"sybayes_metrics_{tag}.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # plot error curve
            fig_name = pathlib.Path(f"exp2_{tag}_surrogate_error")
            line_plot(list(range(len(holdout))), metrics["abs_error"], "trace", "|ΔFLOPs|", f"SyBayes Error {tag}", fig_name)
            print(f"\n===== EXP-2  {tag} =====")
            print(json.dumps(metrics, indent=2))
            print("Figure saved:", fig_name.with_suffix('.pdf').name)

            # planner scalability
            planner = AutoComposer(model, sybayes=syb)
            sla_runtimes = []
            for _ in range(100):
                sla = {"latency_ms": random.randint(20, 120), "energy_j": random.randint(5, 50)}
                st = time.perf_counter(); planner.solve(sla); sla_runtimes.append((time.perf_counter() - st) * 1e3)
            rt_metrics = {
                "solve_time_p99_ms": float(np.percentile(sla_runtimes, 99)),
                "solve_time_mean_ms": float(np.mean(sla_runtimes)),
            }
            with open(exp_dir / f"planner_time_{tag}.json", "w") as f:
                json.dump(rt_metrics, f, indent=2)
            print("Planner run-time (ms) stats:", rt_metrics)

# -----------------------------------------------------------------------------
# EXPERIMENT 3  – Safety, Privacy & Fairness
# -----------------------------------------------------------------------------

def run_exp3(seed: int, default_yaml: Dict[str, Any]):
    set_seed(seed)
    exp_dir = ART_DIR / "exp3" / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = default_yaml["models"]["bitdiff_t5_l"]
    try:
        model = torch.hub.load(model_ckpt, trust_repo=True)
    except Exception as e:
        abort(f"Cannot load MT model for Exp-3: {e}")

    # strict data existence
    data_csv = ROOT / "data" / "mt50k_v1.csv"
    if not data_csv.exists():
        abort("Required Exp-3 dataset missing: " + str(data_csv))
    import pandas as pd

    df = pd.read_csv(data_csv)
    test_df = df.sample(frac=0.1, random_state=seed)

    conditions = {
        "baseline": {"hi_rrb": False, "cofad": False},
        "conductor": {"hi_rrb": True, "cofad": True},
        "ablate_hirrb": {"hi_rrb": False, "cofad": True},
        "ablate_cofad": {"hi_rrb": True, "cofad": False},
    }

    for cond, flags in conditions.items():
        cond_dir = exp_dir / cond; cond_dir.mkdir(parents=True, exist_ok=True)
        compiled = model.eval()
        hirrb = HiRRB(model) if flags["hi_rrb"] else None
        cofad = CoFaD(model, target_epsilon=0.3) if flags["cofad"] else None

        catastrophes, latency, preds, refs, groups = 0, [], [], [], []
        power = PowerLogger(); power.sample()

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=cond):
            src, tgt, gid = row.src, row.tgt, row.grp
            start = time.perf_counter()
            if hirrb is not None:
                risk, bound = hirrb.estimate(src)
                _ = risk < bound  # bookkeeping
            pred = compiled.generate(src)
            latency.append((time.perf_counter() - start) * 1000)
            preds.append(pred); refs.append(tgt); groups.append(gid)
        power.sample(); energy = power.energy_j(); power.close()

        human_labels = test_df["catastrophe"].values
        catastrophes = int(np.sum(human_labels))
        leakage_auc = cofad.leakage_auc(test_df) if cofad else 0.5
        fairness_gap = float(
            df.groupby("grp")["BLEU"].mean().max() - df.groupby("grp")["BLEU"].mean().min()
        )

        primary = {
            "catastrophe_rate": catastrophes / len(test_df),
            "leakage_auc": leakage_auc,
            "fairness_gap_bleu": fairness_gap,
        }
        secondary = {
            "latency_median_ms": float(np.median(latency)),
            "energy_j": energy,
        }
        summ = Summary(3, seed, cond, primary, secondary, datetime.utcnow().isoformat())
        summ.save(cond_dir / f"summary_exp3_{cond}.json")

        fig_name = pathlib.Path(f"exp3_{cond}_safety_privacy")
        line_plot([0, 1], [primary["catastrophe_rate"], primary["leakage_auc"]], "metric", "value", f"Safety/Privacy {cond}", fig_name)
        print(f"\n===== EXP-3 {cond} =====")
        print(json.dumps(asdict(summ), indent=2))
        print("Figure saved:", fig_name.with_suffix('.pdf').name)
