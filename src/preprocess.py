"""src/preprocess.py
Light-weight data-loader utilities that keep the runtime *tiny* while still
using *real* datasets (no synthetic placeholders):
  • Vision : CIFAR-10 (220 KiB download) resized to 224² so it matches
             MobileViT-S' default input size.
  • Audio  : A tiny slice of LibriSpeech dummy set hosted on the Hub
             (``hf-internal-testing/librispeech_asr_dummy`` – 1.3 MB).
  • Text   : WikiText-2 which is <5 MB compressed; we grab the first 1 000
             lines.
All loaders are endless so that the main round-robin loop never stalls.
"""
from __future__ import annotations
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, DistilBertTokenizerFast
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Vision – CIFAR-10 resized to 224² (MobileViT default)
# ---------------------------------------------------------------------------
class _VisionWrapper(Dataset):
    def __init__(self):
        tfm = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.ds = tv_datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True, transform=tfm)

    def __len__(self):  # arbitrary large value so DataLoader never thinks it's exhausted
        return 10_000_000

    def __getitem__(self, idx):
        real_idx = idx % len(self.ds)
        img, lbl = self.ds[real_idx]
        return img, torch.tensor(lbl, dtype=torch.long)


def get_vision_loader(batch_size: int = 8):
    return DataLoader(_VisionWrapper(), batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Audio – LibriSpeech dummy slice → Mel-spectrogram (80 × T)
# ---------------------------------------------------------------------------
class _AudioWrapper(Dataset):
    def __init__(self):
        self.ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        self.feat_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def __len__(self):
        return 10_000_000

    def __getitem__(self, idx):
        ex = self.ds[idx % len(self.ds)]
        wav = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        # WhisperFeatureExtractor returns list[FloatTensor] size (80, T)
        mels = self.feat_extractor(wav, sampling_rate=sr).input_features[0]
        mels = torch.tensor(mels, dtype=torch.float32)
        lbl = torch.tensor(idx % 50, dtype=torch.long)  # pseudo-label (0-49)
        return mels, lbl


def _collate_audio(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    # Pad to longest sequence in *this* batch so Whisper can batch-process
    max_len = max(b[0].shape[-1] for b in batch)
    feats = []
    labels = []
    for m, l in batch:
        pad_len = max_len - m.shape[-1]
        if pad_len > 0:
            m = torch.nn.functional.pad(m, (0, pad_len))
        feats.append(m)
        labels.append(l)
    feats = torch.stack(feats)  # (B, 80, T)
    labels = torch.stack(labels)
    return feats, labels


def get_audio_loader(batch_size: int = 4):
    return DataLoader(
        _AudioWrapper(),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_audio,
    )


# ---------------------------------------------------------------------------
# Text – WikiText-2 first 1 000 lines
# ---------------------------------------------------------------------------
class _TextWrapper(Dataset):
    def __init__(self):
        self.ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
        self.tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return 10_000_000

    def __getitem__(self, idx):
        txt = self.ds[idx % len(self.ds)]["text"]
        enc = self.tok(
            txt,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # as tensors with shape (seq_len,)
        ids = enc.input_ids.squeeze(0)
        attn = enc.attention_mask.squeeze(0)
        return ids, attn, ids  # label = ids (as used in evaluate.py)


def _collate_text(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    ids = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])
    lbl = torch.stack([b[2] for b in batch])
    return ids, attn, lbl


def get_text_loader(batch_size: int = 4):
    return DataLoader(
        _TextWrapper(),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_text,
    )
