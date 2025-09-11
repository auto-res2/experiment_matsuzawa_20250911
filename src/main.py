from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

from .preprocess import (
    DATA_DIR,
    FIG_DIR,
    RESULTS_DIR,
    DatasetDownloadError,
    download_and_verify,
)
from .train import CurvatureGatedGCN, train_epoch
from .evaluate import evaluate, plot_seed_auc, save_metrics_json

# ---------------------------- CONFIG ----------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    print("Configuration file missing – aborting.")
    sys.exit(1)
with open(CONFIG_PATH, "r") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)


# ----------------------- EXPERIMENT 1 ---------------------------

def run_experiment_1() -> List[Dict[str, Any]]:
    cfg = CONFIG["experiment_1"]
    datasets_cfg = cfg["datasets"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_all: List[Dict[str, Any]] = []

    for ds_name, meta in datasets_cfg.items():
        # 1) download -----------------------------------------------------
        _ = download_and_verify(ds_name, meta["url"], meta["sha256"])

        # 2) load dataset -------------------------------------------------
        if ds_name == "ogbn_arxiv_time":
            dataset = DglNodePropPredDataset(name="ogbn-arxiv")
            g, labels = dataset[0]
            labels = labels.squeeze()
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"].to(torch.long)
            val_idx = split_idx["valid"].to(torch.long)
            test_idx = split_idx["test"].to(torch.long)
        elif ds_name == "reddit_threads_2020":
            raise DatasetDownloadError(
                "Reddit-Threads-2020 loader not implemented – dataset is private."
            )
        else:
            raise ValueError(ds_name)

        # 3) graph pre-processing ---------------------------------------
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g.ndata["feat"] = g.ndata["feat"].float()

        in_dim = g.ndata["feat"].shape[1]
        out_dim = int(labels.max().item()) + 1

        # 4) training across seeds -------------------------------------
        auc_seeds = []
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = CurvatureGatedGCN(
                in_dim,
                cfg["model"]["hidden_dim"],
                out_dim,
                num_layers=cfg["model"]["layers"],
                use_bmrf=True,
                process_noise=1.0,
            ).to(device)

            opt = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4
            )

            best_val_auc = 0.0
            best_state = None
            losses: List[float] = []
            for epoch in range(1, 151):
                loss = train_epoch(
                    model,
                    g.to(device),
                    train_idx.to(device),
                    labels.to(device),
                    opt,
                )
                losses.append(loss)
                val_auc = evaluate(
                    model,
                    g.to(device),
                    val_idx.to(device),
                    labels.to(device),
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {
                        k: v.clone().detach().cpu() for k, v in model.state_dict().items()
                    }
                # early stop if no loss improvement for 20 epochs
                if epoch - int(np.argmin(losses)) > 20:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            test_auc = evaluate(
                model,
                g.to(device),
                test_idx.to(device),
                labels.to(device),
            )
            auc_seeds.append(test_auc)

        # 5) aggregate metrics -----------------------------------------
        mean_auc = float(np.mean(auc_seeds))
        ci95 = float(1.96 * np.std(auc_seeds, ddof=1) / np.sqrt(len(auc_seeds)))
        metrics = {
            "dataset": ds_name,
            "mean_test_roc_auc": mean_auc,
            "ci95": ci95,
            "seeds": [float(a) for a in auc_seeds],
        }
        metrics_all.append(metrics)

        # 6) save JSON ----------------------------------------------------
        save_metrics_json("exp1", metrics)

        # 7) figure -------------------------------------------------------
        fig_name = plot_seed_auc(ds_name, auc_seeds)

        # 8) stdout -------------------------------------------------------
        print("\n=== EXPERIMENT 1 –", ds_name, "===")
        print("Bayesian Multi-Resolution Curvature Filter vs raw SRS.")
        print(json.dumps(metrics, indent=2))
        print("Figures:", fig_name)

    return metrics_all


# ----------------------------- MAIN ----------------------------------------

def main() -> None:
    try:
        run_experiment_1()
        # run_experiment_2()  # dataset is private
        # run_experiment_3()  # physical testbed unavailable
    except DatasetDownloadError as e:
        print("\nERROR:", e)
        print("Terminating as per STRICT NO-FALLBACK RULE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
