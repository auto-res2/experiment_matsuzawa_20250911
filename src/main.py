"""src/main.py
---------------------------------------------------------------------
Entry point that mirrors – as closely as possible – the first 120 lines
of the original *monolithic* script while adapting imports so they point
at our new tightened package layout (`src.preprocess`, `src.train`, …).
---------------------------------------------------------------------"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import torch  # still required for device sanity checks further below
import yaml

# ------------------------------------------------------------------
# helper – abort immediately (STRICT NO-FALLBACK)
# ------------------------------------------------------------------

def _abort(msg: str) -> None:
    print("[PHOENIX-MEM] FATAL:", msg, file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------
# load config (no CLI args allowed by assignment → read env EXP_ID)
# ------------------------------------------------------------------
exp_id = os.getenv("PHOENIX_EXP", "experiment_1")
conf_path = Path(__file__).resolve().parent.parent / "config" / f"{exp_id}.yaml"
if not conf_path.exists():
    _abort(f"Configuration file {conf_path} not found. Set PHOENIX_EXP env var.")

with open(conf_path, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

# ------------------------------------------------------------------
# sanity-check that huge datasets are really reachable BEFORE doing anything
# ------------------------------------------------------------------
from .preprocess import ensure_all_datasets_exist  # noqa: E402  – local import is fine

try:
    ensure_all_datasets_exist(conf)  # will raise → _abort() propagated
except Exception as e:  # noqa: BLE001
    _abort(str(e))

# ------------------------------------------------------------------
# pretty experiment description to stdout
# ------------------------------------------------------------------
print(textwrap.dedent(conf.get("description", "<no description in YAML>")))

# ------------------------------------------------------------------
# run the selected experiment
# ------------------------------------------------------------------
from .train import run_exp1, run_exp2, run_exp3  # noqa: E402

if exp_id == "experiment_1":
    results_dict, figure_files = run_exp1(conf)
elif exp_id == "experiment_2":
    results_dict, figure_files = run_exp2(conf)
elif exp_id == "experiment_3":
    results_dict, figure_files = run_exp3(conf)
else:
    _abort(f"Unknown experiment id {exp_id}")

# ------------------------------------------------------------------
# dump & print JSON ----------------------------------------------------------
# ------------------------------------------------------------------
json_name = f"results_{exp_id}.json"
with open(json_name, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2)
print("\n=== Experimental Results (JSON) ==========================================")
print(json.dumps(results_dict, indent=2))
print("\nFigures written:")
for fig in figure_files:
    print("  •", fig)
