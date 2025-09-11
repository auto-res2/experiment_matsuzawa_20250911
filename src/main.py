# src/main.py
"""Main orchestration entry point – executed via `python -m src.main`.

This revision updates the research-artifact paths to comply with **iteration17**
(as mandated by the autopruner spec).  All helpers that write images/JSON must
therefore use the new paths.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, cast

import flwr as fl
import torch
import yaml

from .preprocess import FedPartitionDataset, abort
from .train import CarbonController, Client
from .evaluate import current_power_draw_watts, plot_accuracy, save_json

# ---------------------------------------------------------------------------
#  Resolve project root and mandatory research folders (iteration-17 layout)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration17"  # UPDATED → iteration17
DATA_DIR = RESEARCH_DIR / "data"
FIG_DIR = RESEARCH_DIR / "images"          # .research/iteration17/images
RES_DIR = RESEARCH_DIR                      # JSON files saved directly here
CONFIG_DIR = ROOT / "config"

for _d in (DATA_DIR, FIG_DIR, RES_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Load YAML configuration (mandatory – fail fast if missing)
# ---------------------------------------------------------------------------
CFG_FILE = CONFIG_DIR / "config.yaml"
if not CFG_FILE.exists():
    abort("Configuration file config/config.yaml missing – cannot continue.")

with open(CFG_FILE, "r", encoding="utf-8") as _fp:
    CONFIG: Dict = yaml.safe_load(_fp)

# ---------------------------------------------------------------------------
#  Hardware sanity checks – CI may run on CPU-only runners.
# ---------------------------------------------------------------------------

def ensure_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available – running on CPU.", file=os.sys.stderr)
        return
    n_gpu = torch.cuda.device_count()
    if n_gpu < CONFIG["hardware"]["gpus_required"]:
        print(
            f"WARNING: {CONFIG['hardware']['gpus_required']} GPUs required for full experiment, "
            f"but only {n_gpu} detected. Proceeding with the available GPUs.",
            file=os.sys.stderr,
        )
    gpu_name = torch.cuda.get_device_name(0)
    if CONFIG["hardware"]["expected_gpu_name"] not in gpu_name:
        print(
            f"WARNING: Expected GPU ‘{CONFIG['hardware']['expected_gpu_name']}’, got ‘{gpu_name}’.",
            file=os.sys.stderr,
        )


# ---------------------------------------------------------------------------
#  Custom FedAvg strategy that stores the final global parameters
# ---------------------------------------------------------------------------

class FedAvgSave(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters: fl.common.parameters.Parameters | None = None

    def aggregate_fit(self, rnd, results, failures):  # noqa: D401
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated[0]
        return aggregated


# ---------------------------------------------------------------------------
#  Experiment 1 – Federated Continuous-Time Training & Carbon Audit
# ---------------------------------------------------------------------------

def run_experiment_1() -> Dict:
    print(
        "Experiment 1 – Federated Continuous-Time Training & Carbon Audit\n"
        "Goal: Evaluate MAESTRO’s end-to-end benefits (accuracy, comms, latency, carbon)"
    )

    cfg_exp1 = CONFIG["experiments"]["exp1"]

    # 1 / Dataset partitions -------------------------------------------------
    partitions: List[FedPartitionDataset] = []
    part_cfg = CONFIG["datasets"]["ogbn_products_partitions"]
    for pid in range(part_cfg["num_partitions"]):
        repo = part_cfg["base"].format(pid)
        local_path = DATA_DIR / f"ogbn_products_p{pid}"
        try:
            ds = FedPartitionDataset(repo, local_path)
            partitions.append(ds)
        except Exception as exc:  # noqa: BLE001
            abort(f"Failed to load dataset partition {repo}: {exc}")

    # 2 / Carbon controller --------------------------------------------------
    carbon_ctl: CarbonController | None = None
    try:
        carbon_ctl = CarbonController(threshold=cfg_exp1["carbon_intensity_threshold"])
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: CarbonController disabled – reason: {exc}", file=os.sys.stderr)

    # 3 / Flower strategy ----------------------------------------------------
    strategy = FedAvgSave()

    # 4 / Simulation ---------------------------------------------------------
    clients = [lambda d=ds: Client(d, cfg_exp1) for ds in partitions]
    client_resources = {"num_gpus": 1} if torch.cuda.is_available() else {"num_cpus": 1}

    start_time = time.time()
    hist = fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)](),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=cfg_exp1["num_rounds"]),
        strategy=strategy,
        client_resources=client_resources,
    )
    duration_secs = time.time() - start_time

    # 5 / Metrics ------------------------------------------------------------
    acc_tuples: List[Tuple[int, float]] = []
    if hist.metrics_centralized.get("accuracy"):
        acc_tuples = cast(List[Tuple[int, float]], hist.metrics_centralized.get("accuracy"))
    elif hist.metrics_distributed.get("accuracy"):
        acc_tuples = cast(List[Tuple[int, float]], hist.metrics_distributed.get("accuracy"))

    if acc_tuples:
        rounds, accs = zip(*acc_tuples)
        rounds, accs = list(rounds), list(accs)
    else:
        # ---------------- Manual accuracy computation ----------------
        if strategy.final_parameters is None:
            print("WARNING: No aggregated parameters – cannot compute accuracy.", file=os.sys.stderr)
            rounds, accs = [], []
        else:
            reference_client = Client(partitions[0], cfg_exp1)
            reference_client.set_parameters(strategy.final_parameters.tensors)
            _, _, metrics = reference_client.evaluate(strategy.final_parameters.tensors)
            rounds, accs = [cfg_exp1["num_rounds"]], [metrics["accuracy"]]
            print(f"Computed final accuracy manually: {accs[0]:.4f}")

    # 6 / Communication volume ----------------------------------------------
    reference_client = Client(partitions[0], cfg_exp1)
    model_size = sum(p.numel() * p.element_size() for p in reference_client.model.parameters())
    total_uploads = cfg_exp1["num_rounds"] * len(clients)
    network_bytes = model_size * total_uploads

    # 7 / Energy usage -------------------------------------------------------
    watts = current_power_draw_watts()
    hours = duration_secs / 3600.0
    wh_compute = watts * hours
    wh_network = (network_bytes * 0.06e-6) / 3600.0  # µJ → Wh

    results = {
        "final_accuracy": float(accs[-1]) if accs else None,
        "best_accuracy": max(accs) if accs else None,
        "rounds": len(rounds),
        "network_bytes": int(network_bytes),
        "wh_compute": wh_compute,
        "wh_network": wh_network,
    }

    # 8 / Plot ---------------------------------------------------------------
    if rounds:
        fig_name = plot_accuracy(rounds, accs, FIG_DIR / "accuracy_maestro.pdf")
        print("Figure saved:", fig_name)

    # 9 / Persist ------------------------------------------------------------
    res_file = RES_DIR / "exp1_results.json"
    save_json(res_file, results)

    if carbon_ctl is not None:
        carbon_ctl.shutdown()

    print("Results written to", res_file)
    print(json.dumps(results, indent=2))  # mandatory stdout dump
    return results


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------

def main():  # noqa: D401
    ensure_gpu()
    run_experiment_1()


if __name__ == "__main__":
    main()
