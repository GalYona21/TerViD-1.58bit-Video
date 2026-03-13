"""
Step 2: Pre-encode videos into VAE latents + text embeddings.

This is run ONCE before training to avoid loading VAE + text encoder
during the QAT training loop (saves ~4GB VRAM on the 3090s).

Input: WebDataset shards from video2dataset (data/raw_videos/)
Output: .pt files with {latents, text_emb, caption} (data/latents/)

Usage:
    python scripts/02_prepare_latents.py \
        --input_dir data/raw_videos \
        --output_dir data/latents \
        --model_id Lightricks/LTX-Video \
        --resolution 256 \
        --num_frames 16
"""

import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-encode videos into VAE latents")
    parser.add_argument("--input_dir", type=str, default="data/raw_videos")
    parser.add_argument("--output_dir", type=str, default="data/latents")
    parser.add_argument("--model_id", type=str, default="Lightricks/LTX-Video")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print(f"Loading VAE and text encoder from {args.model_id}...")

    # Import here to avoid loading everything at module level
    from diffusers import AutoencoderKLLTXVideo
    from transformers import T5EncoderModel, T5Tokenizer

    # Load VAE
    vae = AutoencoderKLLTXVideo.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device).eval()

    # Load text encoder
    text_encoder = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")

    print("Models loaded. Processing videos...")

    # Load videos from WebDataset shards
    import webdataset as wds
    import torchvision.transforms as T
    import decord
    from io import BytesIO

    decord.bridge.set_bridge("torch")

    transform = T.Compose([
        T.Resize((args.resolution, args.resolution)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
    ])

    # Find all tar shards
    input_dir = Path(args.input_dir)
    shard_paths = sorted(input_dir.glob("*.tar"))
    if not shard_paths:
        # Try flat directory of mp4 + txt files
        shard_paths = sorted(input_dir.glob("*.mp4"))
        print(f"Found {len(shard_paths)} video files")
    else:
        print(f"Found {len(shard_paths)} WebDataset shards")

    dataset = wds.WebDataset([str(p) for p in shard_paths]).decode("torchrgb")

    sample_idx = 0
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Encoding"):
            try:
                # Get video tensor
                if "mp4" in sample:
                    video = sample["mp4"]  # (T, H, W, C) or similar
                elif "video.mp4" in sample:
                    video = sample["video.mp4"]
                else:
                    continue

                # Get caption
                caption = ""
                if "txt" in sample:
                    caption = sample["txt"]
                elif "caption" in sample:
                    caption = sample["caption"]

                # Ensure correct shape: (B, C, T, H, W)
                if video.dim() == 4:  # (T, H, W, C)
                    video = video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
                elif video.dim() == 3:  # (H, W, C) single frame
                    video = video.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

                # Subsample frames
                if video.shape[2] > args.num_frames:
                    indices = torch.linspace(0, video.shape[2] - 1, args.num_frames).long()
                    video = video[:, :, indices]

                # Resize and normalize
                B, C, T, H, W = video.shape
                video = video.float()
                frames = []
                for t in range(T):
                    frame = transform(video[0, :, t])
                    frames.append(frame)
                video = torch.stack(frames, dim=1).unsqueeze(0)  # (1, C, T, H, W)

                # Encode through VAE
                video = video.to(device=device, dtype=torch.float16)
                latents = vae.encode(video).latent_dist.sample()

                # Encode caption
                tokens = tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                text_emb = text_encoder(**tokens).last_hidden_state

                # Save
                torch.save(
                    {
                        "latents": latents.cpu(),
                        "text_emb": text_emb.cpu(),
                        "caption": caption,
                    },
                    output_dir / f"{sample_idx:06d}.pt",
                )
                sample_idx += 1

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

    print(f"Done! Encoded {sample_idx} samples to {output_dir}")
    print(f"Next step: python scripts/03_train.py")


if __name__ == "__main__":
    main()
