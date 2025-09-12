"""Model-training utilities.

The original experiment did not contain any concrete training logic –
only an informational message embedded in the former monolithic script.
That behaviour is preserved here so that the refactor introduces **no
new model code** while still exposing a public `train` function that can
be called from the orchestrating script.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["train"]


def train(cfg: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
    """Dummy training stub extracted from the original script.

    A real training loop is intentionally **not** implemented because the
    user must supply a valid configuration and integrate their own model
    code.  The function returns a minimal dictionary so that downstream
    evaluation logic has a well-defined input.
    """
    # Inform the user that no training is executed – exactly as in the
    # original single-file script.
    print(
        "[INFO] All mandatory configuration fields found.  However, "
        "actual model training is disabled in this auto-generated stub.  "
        "Please integrate your training pipeline here."
    )
    print(
        "[INFO] Exiting without running experiments to comply with resource "
        "constraints and awaiting valid, user-supplied configuration."
    )

    # Return a minimal structure so that the caller can still serialise a
    # result JSON.
    return {
        "status": "training_skipped",
        "dataset": cfg.get("dataset", {}).get("name"),
        "model": cfg.get("model", {}).get("name"),
    }
