"""src/train.py
Model and training-related components extracted from the original monolithic
script.  The original implementation made two critical mistakes that caused
runtime failures:
  1.  LoRA parameters were **not registered** inside the parent ``nn.Module``,
      therefore they never moved to the right device and were never optimised.
  2.  The encoder branch for audio expected an extra channel-dimension that is
      *not* required by Whisper.  Passing a tensor of shape ``(B, 1, T)``
      raises a shape error inside 🤗 Transformers.

Both issues are fixed below while keeping the public API unchanged.
"""
from __future__ import annotations
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import WhisperModel, DistilBertModel
import timm

__all__ = [
    "TinyCodeBuffer",
    "Rank1LoRA",
    "TinyTACO",
    "build_models",
]

# ---------------------------------------------------------------------------
# Helper – 512-byte code buffer (32-bit hashes)
# ---------------------------------------------------------------------------
class TinyCodeBuffer:
    """Fixed 512-byte circular buffer storing 4-byte int hashes → max 128 codes."""

    CAPACITY = 128  # 128 × 4 B = 512 B

    def __init__(self):
        self._codes: List[int] = []
        self._idx = 0

    # -------------------------------------------------------------------
    def push(self, code: int):
        if len(self._codes) < self.CAPACITY:
            self._codes.append(code & 0xFFFFFFFF)
        else:
            self._codes[self._idx] = code & 0xFFFFFFFF
            self._idx = (self._idx + 1) % self.CAPACITY

    # -------------------------------------------------------------------
    @property
    def used_bytes(self) -> int:  # always ≤512
        return 4 * len(self._codes)


# ---------------------------------------------------------------------------
# LoRA rank-1 patch utility
# ---------------------------------------------------------------------------
class Rank1LoRA(nn.Module):
    """A *minimal* rank-1 LoRA adapter for a Linear layer."""

    def __init__(self, layer: nn.Linear, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.weight_a = nn.Parameter(torch.zeros((1, layer.in_features)))
        self.weight_b = nn.Parameter(torch.zeros((layer.out_features, 1)))
        nn.init.normal_(self.weight_a, std=1e-4)
        nn.init.normal_(self.weight_b, std=1e-4)

    def forward(self, x):  # pylint: disable=arguments-differ
        delta_w = (self.weight_b @ self.weight_a) * self.alpha
        return F.linear(x, self.layer.weight + delta_w, self.layer.bias)


# ---------------------------------------------------------------------------
# TinyTACO – minimal continual learner (incl. classification heads)
# ---------------------------------------------------------------------------
class TinyTACO(nn.Module):
    """A *minimal* continual-learning module with LoRA patches and tiny heads."""

    def __init__(
        self,
        vision_trunk: nn.Module,
        audio_trunk: WhisperModel,
        text_trunk: DistilBertModel,
        device: torch.device,
        cfg: dict,
    ):
        super().__init__()
        self.device = device
        self.codebook = TinyCodeBuffer()

        # Attach LoRA rank-1 patches to *last* Linear layer of each trunk
        self.vision = vision_trunk
        self.audio = audio_trunk
        self.text = text_trunk
        self._patch_last_linear(self.vision)
        self._patch_last_linear(self.audio)
        self._patch_last_linear(self.text)

        # ------------------------------------------------------------------
        # Task-specific classification heads
        # ------------------------------------------------------------------
        self.audio_head = nn.Linear(self.audio.config.d_model, 50)  # ESC-50
        self.text_head = nn.Linear(self.text.config.hidden_size, self.text.config.vocab_size)

        # Simple probe network (used for drift estimation)
        self.probe = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, cfg["hyper_params"]["probe_dim"]),
        )

        self.to(device)
        self.opt = optim.Adam(self.parameters(), lr=cfg["hyper_params"]["lr"])

    # ------------------------------------------------------------------
    @staticmethod
    def _patch_last_linear(model: nn.Module):
        """Monkey-patch the *last* nn.Linear inside ``model`` with a LoRA layer.
        The LoRA module is attached to the parent so its parameters are
        properly registered – this was missing in the original code.
        """
        last_lin = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_lin = m  # keep overwriting → last one wins
        if last_lin is None:
            raise RuntimeError("No Linear layer found for LoRA patching.")

        # Create & **register** the LoRA adapter
        rank1 = Rank1LoRA(last_lin)
        setattr(model, f"_rank1_lora_{id(last_lin)}", rank1)

        # Replace the forward of ``last_lin`` so that it calls the LoRA module
        def _patched(x, rank1_layer=rank1):
            return rank1_layer(x)

        last_lin.forward = _patched  # override

    # ------------------------------------------------------------------
    # Forward helpers return *logits*
    # ------------------------------------------------------------------
    def forward_vision(self, imgs):
        return self.vision(imgs)  # already (B,1000)

    def forward_audio(self, mels):
        feats = self.audio(mels).last_hidden_state.mean(dim=1)
        return self.audio_head(feats)

    def forward_text(self, input_ids, attention_mask):
        feats = self.text(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.text_head(feats)

    # ------------------------------------------------------------------
    def update_online(self, loss: torch.Tensor):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        # Push a random 32-bit hash into the 512-B buffer
        self.codebook.push(random.getrandbits(32))


# ---------------------------------------------------------------------------
# Convenience factory -------------------------------------------------------
# ---------------------------------------------------------------------------

def build_models(device: torch.device, cfg: dict) -> TinyTACO:
    vision_trunk = timm.create_model(cfg["models"]["vision_trunk"], pretrained=True)
    audio_trunk = WhisperModel.from_pretrained(cfg["models"]["audio_trunk"]).encoder
    text_trunk = DistilBertModel.from_pretrained(cfg["models"]["text_trunk"])
    return TinyTACO(vision_trunk, audio_trunk, text_trunk, device, cfg)
