# src/main.py
"""Entry-point orchestrating the full continual-learning experiment.  Execute
via ``python -m src.main``.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .evaluate import save_line_plot
from .preprocess import DataUnavailableError, prepare_imagenette
from .train import build_resnet50_hira, run_epoch

# -----------------------------------------------------------------------------
#  DIRECTORIES & CONFIG
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / ".research" / "iteration1"
IMAGE_DIR = RESULT_DIR / "images"  # created by evaluate.py but ensure parent exists
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

for _d in (DATA_DIR, RESULT_DIR, IMAGE_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  DEFAULT CONFIGURATION (over-write on first run)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Dict] = {
    "datasets": {
        "imagenette": {
            "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "sha256": "a5090ff472b795ac2f7c5c5af1c9c80de13b9866f53c9c8867fd9678e9efb4a6",
            "filename": "imagenette2.tgz",
        }
    },
    "hyperparameters": {
        "lr": 1e-4,
        "rank": 32,
        "batch_size": 32,
    },
}


# -----------------------------------------------------------------------------
#  CONFIG LOADER / SAVER
# -----------------------------------------------------------------------------

def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
        print(f"[INFO] Default config written to {CONFIG_PATH}. Edit as needed.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
#  EXPERIMENT 1 – Imagenette Continual-Learning DEMO
# -----------------------------------------------------------------------------

def experiment1_imagenette(cfg: Dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) DATA -----------------------------------------------------------------
    prepare_imagenette(DATA_DIR / "imagenette2", cfg["datasets"]["imagenette"])

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_ds = datasets.ImageFolder(DATA_DIR / "imagenette2" / "train", transform=transform)

    # create 10 artificial tasks – one per class
    class_indices: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_ds):
        class_indices[label].append(idx)

    tasks: List[List[int]] = [indices for _, indices in sorted(class_indices.items())]

    # 2) MODEL & OPTIMISER ----------------------------------------------------
    hp = cfg["hyperparameters"]
    model = build_resnet50_hira(num_classes=10, rank=hp["rank"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.hira.parameters(), lr=hp["lr"], weight_decay=0.01)

    # 3) CONTINUAL LOOP -------------------------------------------------------
    acc_per_task: List[float] = []
    seen_indices: List[int] = []

    for t, idx_list in enumerate(tasks, start=1):
        seen_indices.extend(idx_list)
        train_subset = torch.utils.data.Subset(full_ds, seen_indices)
        len_val = int(0.1 * len(train_subset))
        len_train = len(train_subset) - len_val
        train_ds, val_ds = random_split(train_subset, [len_train, len_val])

        train_loader = DataLoader(
            train_ds,
            batch_size=hp["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=hp["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # single epoch per task (CI-friendly)
        train_acc = run_epoch(model, train_loader, criterion, optimizer, device=device)
        val_acc = run_epoch(model, val_loader, criterion, None, device=device)
        acc_per_task.append(val_acc)
        print(f"[Task {t:02d}] TrainAcc={train_acc:.2f}  ValAcc={val_acc:.2f}")

    # 4) PLOT & SERIALISE RESULTS -------------------------------------------
    fig_name = save_line_plot(
        list(range(1, 11)),
        acc_per_task,
        "Task",
        "Accuracy (%)",
        "Online Val Accuracy – Imagenette",
        "accuracy_harp_raft_imagenette",
    )

    result = {
        "experiment": "Experiment 1 – Vision-Discriminative (Imagenette demo)",
        "datetime": datetime.utcnow().isoformat(),
        "tasks": len(tasks),
        "accuracy_per_task": acc_per_task,
        "final_accuracy": acc_per_task[-1],
        "figure": fig_name,
    }

    res_file = RESULT_DIR / "experiment1_imagenette.json"
    with open(res_file, "w") as f:
        json.dump(result, f, indent=2)

    # mandatory STDOUT for CI validation
    print("\n=== Experiment 1 – Imagenette Continual-Learning ===")
    print(json.dumps(result, indent=2))
    print("Figure saved:", fig_name)


# -----------------------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    cfg = load_config()

    # Reproducibility ---------------------------------------------
    torch.manual_seed(13)
    random.seed(13)

    # Run experiments --------------------------------------------
    try:
        experiment1_imagenette(cfg)
    except DataUnavailableError as exc:
        print(f"[FATAL] {exc}. Aborting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
