"""
main.py – orchestration entry-point
-----------------------------------
Loads the YAML config and sequentially runs the three experiments defined
in evaluate.py.  Execute via
    python -m src.main
"""
from pathlib import Path
import textwrap
import warnings

import yaml

from src import evaluate  # absolute import keeps import-time side effects tidy

# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------

DEFAULT_YAML = """
exp1:
  dataset_url: https://huggingface.co/datasets/iwslt2017
  dataset_cfg: de-en
  gpt2_model: gpt2-medium
  ppl_bins: [0, 20, 40, 60, 80]
  batch_size: 32
  max_steps: 50
  repetitions: 2  # keep runtime low for CI
  sprt_confidence: 99.5
  kappa_scale: 1.0
exp2:
  sd_checkpoint: runwayml/stable-diffusion-v1-5
  prompt_dataset_url: https://huggingface.co/datasets/Yuanzhi/aesthetics_prompts_laion
  num_prompts: 128  # keep runtime low
  rl_train_hours: 6
  lambda: 0.3
  mu: 0.1
  batch_size: 4
exp3:
  mobile_mt_sentences: 5000
  video_minutes: 10
  grid_regions:
    A100: US-CA
    RTX4060: EU-DE
    M2Pro: EU-DK
    RISCV: OFFGRID
"""

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_DIR.mkdir(exist_ok=True, parents=True)
CFG_FILE = CONFIG_DIR / "config.yaml"

if not CFG_FILE.exists():
    CFG_FILE.write_text(textwrap.dedent(DEFAULT_YAML))

with open(CFG_FILE) as f:
    CONFIG = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Run experiments – each prints its own JSON to stdout
# -----------------------------------------------------------------------------

try:
    evaluate.run_exp1(CONFIG["exp1"])
    evaluate.run_exp2(CONFIG["exp2"])
    evaluate.run_exp3(CONFIG["exp3"])
except Exception as e:  # pragma: no cover – ensures fail-fast with context
    warnings.warn(f"[FATAL] Experiment runner crashed: {e}")
    raise
