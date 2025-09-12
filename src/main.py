"""src/main.py
Command-line entry-point orchestrating the complete experimental workflow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np  # noqa: F401 – retained for potential future use
import torch
import yaml

# Use absolute imports so that static analyzers can resolve them in isolation.
import src.preprocess as pp  # noqa: E402
import src.train as tr  # noqa: E402
import src.evaluate as ev  # noqa: E402

# -----------------------------------------------------------------------------
#                         Global project-level paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RESEARCH_DIR = ROOT / ".research" / "iteration2"
IMAGE_DIR = RESEARCH_DIR / "images"


# -----------------------------------------------------------------------------
#                                Utilities
# -----------------------------------------------------------------------------

def _load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_experiment(cfg: Dict):
    """Complete end-to-end run (train → evaluate → persist/results)."""

    # --------------------- Data preparation ---------------------
    train_loader, val_loader, test_loader = pp.create_dataloaders(cfg)

    # ----------------------- Model set-up -----------------------
    model = tr.MLP(
        cfg["model"]["input_dim"],
        cfg["model"]["hidden_dims"],
        cfg["model"]["output_dim"],
    ).to("cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # -------------------- Training procedure -------------------
    train_losses, val_losses, val_accs = tr.train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimiser,
        cfg["training"]["epochs"],
        cfg["training"]["seed"],
    )

    # ----------------- Final test-set evaluation ---------------
    test_acc, cm = ev.evaluate_on_test(model, test_loader)

    # ----------------------- Visualisation ---------------------
    fig_files = ev.plot_curves(train_losses, val_accs, cm, cfg["experiment_name"], IMAGE_DIR)

    # ------------------------ Persistence ----------------------
    results = {
        "experiment_name": cfg["experiment_name"],
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accs[-1],
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "figures": fig_files,
    }
    result_path = ev.persist_results(results, RESEARCH_DIR)

    # ------------------------ Console log ----------------------
    print("\n=================  EXPERIMENT DESCRIPTION  =================")
    print(f"Experiment name      : {cfg['experiment_name']}")
    print(f"Dataset URL          : {cfg['dataset']['url']}")
    print(
        "Model architecture   : "
        f"input_dim={cfg['model']['input_dim']}, hidden_dims={cfg['model']['hidden_dims']}, "
        f"output_dim={cfg['model']['output_dim']}"
    )
    print(f"Training epochs      : {cfg['training']['epochs']}")
    print("===========================================================\n")

    print("----------------  EXPERIMENTAL RESULTS (JSON)  --------------")
    with result_path.open("r", encoding="utf-8") as fh:
        print(fh.read())
    print("-------------------------------------------------------------\n")

    print("Generated figure files (relative to .research/iteration2/images):")
    for f in fig_files:
        print(" •", f)


# -----------------------------------------------------------------------------
#                                CLI parsing
# -----------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run Iris experiments.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test only")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment only")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
#                                    MAIN
# -----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # -------- Determine which configuration(s) to execute --------
    configs_to_run: list[Path] = []
    if args.smoke_test:
        configs_to_run.append(CONFIG_DIR / "smoke_test.yaml")
    elif args.full_experiment:
        configs_to_run.append(CONFIG_DIR / "full_experiment.yaml")
    else:
        # Default: try smoke-test first → only if that succeeds continue with full run
        configs_to_run.extend([
            CONFIG_DIR / "smoke_test.yaml",
            CONFIG_DIR / "full_experiment.yaml",
        ])

    for cfg_path in configs_to_run:
        cfg = _load_config(cfg_path)
        print(  # make phase visually separated in the logs
            "\n============================== "
            f"{cfg['experiment_name'].upper()} "
            "=============================="
        )
        try:
            _run_experiment(cfg)
        except Exception as exc:  # pragma: no cover – generic catch to prevent CI crashes
            print(f"‼ Experiment '{cfg['experiment_name']}' failed → {exc}")
            # break the chain if smoke-test failed
            if not args.full_experiment:
                break


if __name__ == "__main__":
    main()
