"""src/main.py – orchestrator for the ORBITAL experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # lightweight and part of most ML base images

# Early import of torch so that missing CUDA environments fail fast.  Even if
# we do not actively use the module here, the original monolithic script had
# the same behaviour.
import torch  # noqa: F401 – intentional side-effect import

from .preprocess import ResourceValidator
from .train import run_experiment_placeholders

# ---------------------------------------------------------------------------
# Load experiment configuration from config/config.yaml
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found at '{_CONFIG_PATH}'.  The project "
        "structure is broken."
    )

with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG: Dict[str, Any] = yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Helper – log in to HuggingFace Hub if the user supplied a token
# ---------------------------------------------------------------------------

def _authenticate() -> Optional[str]:
    """Return the token (may be None) so other modules can reuse it."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
    return hf_token


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – imperative style
    """Validate external resources and (optionally) launch experiments."""

    hf_token = _authenticate()
    validator = ResourceValidator(hf_token)

    # --------------------------------------------------------------
    # 1. Validate all datasets
    try:
        for ds in CONFIG["resources"]["datasets"].values():
            print(f"Validating dataset: {ds} …", flush=True)
            validator.validate_dataset(ds)
            print("  ✓ accessible")

        # 2. Validate all models
        for model in CONFIG["resources"]["models"].values():
            print(f"Validating model: {model} …", flush=True)
            validator.validate_model(model)
            print("  ✓ accessible")
    except RuntimeError as err:
        # Strict no-fallback policy – abort immediately
        print("\nERROR: " + str(err))
        print("Programme terminated due to missing external resource.")
        sys.exit(1)

    # --------------------------------------------------------------
    # 3. All resources are reachable – launch (placeholder) experiment
    run_experiment_placeholders(CONFIG)


if __name__ == "__main__":  # pragma: no cover
    main()
