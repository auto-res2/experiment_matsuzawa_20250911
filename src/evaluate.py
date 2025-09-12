"""Evaluation utilities.

The original code base only provided a `save_json` helper.  It is moved
here unchanged so that other modules can persist results without adding
new dependencies or files.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

__all__ = ["save_json", "evaluate"]


def save_json(obj: Dict[str, Any] | Any, path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def evaluate(cfg: Dict[str, Any], train_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """A placeholder evaluation step so that the overall pipeline has a
    consistent signature.  No additional logic is introduced beyond the
    original repository’s functionality.
    """
    # Nothing to evaluate – faithfully propagate training information.
    result = {"status": "evaluation_skipped"}
    result.update(train_outputs)
    return result
