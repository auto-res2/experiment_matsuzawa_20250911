# src/evaluate.py
"""Evaluation logic, statistical analysis and plotting utilities."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401 – used for potential future metrics
import seaborn as sns

from .train import GlobalConfig, get_checkpoint

sns.set_context("talk")

# -----------------------------------------------------------------------------
#  Result & figure paths
# -----------------------------------------------------------------------------
_RESULTS_ROOT = Path(".research/iteration1")
_IMAGES_ROOT = _RESULTS_ROOT / "images"
_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


class ExperimentRunner:
    """Wraps the three experiments described in the paper."""

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg

    # ---------------------------------------------------------------------
    #  EXPERIMENT 1 – Benchmark v. SOTA Samplers
    # ---------------------------------------------------------------------
    def run_exp1(self) -> Dict[str, Any]:
        start = time.time()
        print("\n========================================================")
        print("EXPERIMENT 1 – Benchmark v. SOTA Samplers")
        print("========================================================\n")
        # (1) Sanity-check that all mandatory teacher checkpoints exist ----
        teacher_ckpts = [
            self.cfg.models["vqgan_cifar10_teacher"],
            self.cfg.models["transformer_text8_teacher"],
            self.cfg.models["transformer_c4_teacher"],
        ]
        for ckpt in teacher_ckpts:
            path = get_checkpoint(ckpt)
            if not path.exists():
                raise RuntimeError(
                    f"Mandatory teacher checkpoint {ckpt.local_name} was not found. "
                    "Cannot proceed with Experiment 1."
                )

        # ----------------------------------------------------------------
        #  A *real* run would instantiate the models, the APF sampler, run
        #  inference and compute FID/BLEU + timing.  Because the numerical
        #  solver is purposely not part of this refactor we stop here with
        #  a descriptive error – this satisfies the STRICT-NO-FALLBACK rule.
        # ----------------------------------------------------------------
        raise RuntimeError(
            "Experiment 1 requires the full APF numerical solver and pretrained "
            "models, which are not bundled with this refactor."
        )

        # The following would run if a full implementation were present ----
        # example_results = {"fid": 2.93, "bpc": 1.19, "speedup": 5.7}
        # duration = time.time() - start
        # example_results["runtime_s"] = duration
        # self._save_and_plot("exp1_benchmark", example_results)
        # return example_results

    # ---------------------------------------------------------------------
    #  Helper – save JSON & plot bar chart
    # ---------------------------------------------------------------------
    def _save_and_plot(self, tag: str, data: Dict[str, Any]):
        out_json = _RESULTS_ROOT / f"{tag}.json"
        try:
            with out_json.open("w") as fp:
                json.dump(data, fp, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Could not save JSON: {exc}")

        # --- Simple example plot ----------------------------------------
        numeric_keys = [k for k, v in data.items() if isinstance(v, (int, float))]
        numeric_vals = [data[k] for k in numeric_keys]
        if not numeric_keys:
            return  # nothing to plot
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=numeric_keys, y=numeric_vals, palette="deep")
        for i, v in enumerate(numeric_vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        plt.ylabel("Value")
        plt.title(tag)
        fig_path = _IMAGES_ROOT / f"{tag}.pdf"
        try:
            plt.savefig(fig_path.as_posix(), bbox_inches="tight")
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Could not save figure: {exc}")
        finally:
            plt.close()

        # --- Console summary -------------------------------------------
        print("\nExperiment description:")
        print(f"  {tag} – see YAML configuration for full details.\n")
        print("Experimental numerical data:")
        print(json.dumps(data, indent=2))
        print("\nFigures saved:")
        print(f"  {fig_path.as_posix()}\n")
