"""
src/main.py – unchanged logic, only description update for LM.
"""
from __future__ import annotations

import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml, torch

from src.preprocess import build_datamodule
from src.train import train_and_validate
from src.evaluate import evaluate_model


def _load_config(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _describe(cfg: Dict):
    lines = ["Experiment:"]
    lines += [f"  Name   : {cfg.get('experiment',{}).get('name')}",
              f"  Dataset: {cfg['dataset']['name']}",
              f"  Model  : {cfg['model']['type']}",
              f"  Epochs : {cfg['training']['epochs']}",
              f"  BS     : {cfg['dataset']['batch_size']}"]
    return "\n".join(lines)


def run(cfg_path: Path):
    cfg = _load_config(cfg_path)
    out = Path("outputs")/f"{cfg.get('experiment',{}).get('name')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(_describe(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = build_datamodule(cfg)
    cfg.setdefault("dataset", {})["input_dim"] = getattr(dm, "input_dim", 0)
    cfg["dataset"]["num_classes"] = getattr(dm, "num_classes", 0)

    train_out = train_and_validate(cfg, dm.train_loader, dm.val_loader, cfg["dataset"].get("input_dim",0), cfg["dataset"].get("num_classes",0), device, out)
    with open(out/"history.json", "w") as fp:
        json.dump(train_out["history"], fp, indent=2)

    eval_out = evaluate_model(train_out["best_ckpt_path"], dm.test_loader, device, out, cfg.get('experiment',{}).get('name'))
    print("\nRESULTS:\n", json.dumps(eval_out["metrics"], indent=2))

    figs = list((out/"figures").glob("*.pdf"))
    if figs:
        print("Generated figures:")
        for p in figs:
            print("  ", p.name)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true")
    g.add_argument("--full-experiment", action="store_true")
    g.add_argument("--config")
    args = p.parse_args()
    if args.smoke_test:
        cfg = Path("config/smoke_test.yaml")
    elif args.full_experiment:
        cfg = Path("config/full_experiment.yaml")
    else:
        cfg = Path(args.config)
    run(cfg)

if __name__ == "__main__":
    main()
