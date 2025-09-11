"""CLI entry-point so that `python -m src.main` executes Experiment 1 with the
minimal synthetic configuration bundled in *config/config.yaml*.

This is *not* part of the evaluation harness but offers a convenient manual
smoke-test.
"""
from __future__ import annotations

import pathlib
import yaml

from .config_loader import load_config
from .train import Experiment1Runner


def _main() -> None:  # noqa: D401
    cfg_path = pathlib.Path("config/config.yaml")
    exp_cfg, ds_cfgs, model_cfg = load_config(yaml.safe_load(cfg_path.read_text()))
    runner = Experiment1Runner(exp_cfg, ds_cfgs, model_cfg, pathlib.Path("outputs"))
    runner.run()


if __name__ == "__main__":  # pragma: no cover
    _main()
