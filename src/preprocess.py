"""src/preprocess.py – data-loading and resource-validation utilities."""
from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


class ResourceDescriptor:
    """Simple container holding information about one external resource."""

    def __init__(self, hf_id: str, is_gated: bool = False):
        self.hf_id = hf_id
        self.is_gated = is_gated

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401 – keep simple
        return f"ResourceDescriptor(hf_id={self.hf_id!r}, is_gated={self.is_gated})"


class ResourceValidator:
    """Validates availability of datasets and models on the HuggingFace Hub."""

    def __init__(self, hf_token: Optional[str]):
        self.api = HfApi(token=hf_token)

    # ------------------------------------------------------------------
    def validate_dataset(self, ds_name: str) -> None:  # noqa: D401 – imperative style
        """Raise RuntimeError if the dataset cannot be accessed or streamed."""
        try:
            # Check that the dataset repository exists
            _ = self.api.dataset_info(ds_name)
            # Stream a single example – cheap probe to verify permissions
            _ = next(iter(load_dataset(ds_name, split="train", streaming=True)))
        except Exception as exc:  # pragma: no cover – network faults
            raise RuntimeError(
                f"Dataset '{ds_name}' is not accessible.  Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def validate_model(self, model_name: str) -> None:  # noqa: D401 – imperative style
        """Raise RuntimeError if the model cannot be accessed (including gated)."""
        try:
            info = self.api.model_info(model_name)
            gated = bool(getattr(info, "gated", False))
            if gated:
                if self.api.token is None:
                    raise RuntimeError(
                        f"Model '{model_name}' is gated but no HF_TOKEN was provided."
                    )
                # Attempt to download a tiny file to confirm permission
                hf_hub_download(
                    repo_id=model_name, filename="config.json", token=self.api.token
                )
        except Exception as exc:  # pragma: no cover – network faults
            raise RuntimeError(
                f"Model '{model_name}' is not accessible.  Original error: {exc}"
            ) from exc
