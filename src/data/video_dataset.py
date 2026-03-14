"""
Video dataset loading for 1.58-bit video model training.

Supports two data sources:
1. WebDataset shards (from video2dataset)
2. Local video files with captions

The dataset provides pre-encoded VAE latents + text embeddings
to avoid running VAE/text encoder during training (saves VRAM).
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import random


class PrecomputedLatentDataset(Dataset):
    """Dataset of precomputed VAE latents + text embeddings.

    During dataset preparation (scripts/prepare_latents.py):
    1. Videos are encoded through LTX-Video's VAE -> latents
    2. Captions are encoded through text encoder -> embeddings
    3. Both are saved as .pt files

    This avoids loading VAE + text encoder during training,
    saving ~4GB VRAM on the 2x 3090 setup.

    Directory structure:
        data_dir/
            00000.pt  # {"latents": tensor, "text_emb": tensor, "caption": str}
            00001.pt
            ...
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], weights_only=True)
        return {
            "latents": data["latents"],      # (C, T, H, W)
            "text_emb": data["text_emb"],    # (seq_len, dim)
        }


class PromptOnlyDataset(Dataset):
    """Dataset of text prompts only (for self-supervised training).

    No video data needed — just prompts that will be used with
    random noise to train via teacher distillation.

    Based on 1.58-bit FLUX approach which used 7,232 prompts
    from Parti-1k + T2I CompBench.
    """

    def __init__(self, prompts_file: str, text_encoder=None, cache_dir: Optional[str] = None):
        """
        Args:
            prompts_file: path to .txt file with one prompt per line, or .json list
            text_encoder: if provided, encode prompts on init and cache
            cache_dir: directory to cache encoded prompts
        """
        # Load prompts
        if prompts_file.endswith(".json"):
            with open(prompts_file) as f:
                self.prompts = json.load(f)
        else:
            with open(prompts_file) as f:
                self.prompts = [line.strip() for line in f if line.strip()]

        self.embeddings = None
        self.cache_path = None

        if cache_dir:
            self.cache_path = Path(cache_dir) / "prompt_embeddings.pt"
            if self.cache_path.exists():
                self.embeddings = torch.load(self.cache_path, weights_only=True)
                print(f"Loaded cached embeddings: {self.embeddings.shape}")

    def encode_prompts(self, text_encoder, tokenizer, device: str = "cuda", batch_size: int = 32):
        """Precompute text embeddings for all prompts."""
        all_embeddings = []
        text_encoder = text_encoder.to(device).eval()

        with torch.no_grad():
            for i in range(0, len(self.prompts), batch_size):
                batch = self.prompts[i : i + batch_size]
                tokens = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                emb = text_encoder(**tokens).last_hidden_state
                all_embeddings.append(emb.cpu())

        self.embeddings = torch.cat(all_embeddings, dim=0)

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.embeddings, self.cache_path)
            print(f"Cached {len(self.prompts)} prompt embeddings to {self.cache_path}")

        return self.embeddings

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        result = {"prompt": self.prompts[idx]}
        if self.embeddings is not None:
            result["text_emb"] = self.embeddings[idx]
        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for video training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
