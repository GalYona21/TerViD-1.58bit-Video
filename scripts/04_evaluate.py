"""
Step 4: Evaluate the 1.58-bit video model.

Generates videos and computes metrics for the paper:
- VBench scores (video quality benchmark)
- FVD (Frechet Video Distance)
- CLIP score (text-video alignment)
- Model size comparison (checkpoint bytes)
- Inference VRAM usage
- Inference speed (tokens/sec)

Usage:
    python scripts/04_evaluate.py \
        --ternary_checkpoint checkpoints/final/ternary_model.pt \
        --model_id Lightricks/LTX-Video \
        --prompts_file data/eval_prompts.txt \
        --output_dir results/
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 1.58-bit video model")
    parser.add_argument("--ternary_checkpoint", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Lightricks/LTX-Video")
    parser.add_argument("--prompts_file", type=str, default="data/eval_prompts.txt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compare_fp16", action="store_true", default=True,
                        help="Also run FP16 baseline for comparison")
    return parser.parse_args()


def measure_model_size(path: str) -> float:
    """Get checkpoint size in MB."""
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def measure_peak_vram(fn, *args, **kwargs):
    """Measure peak VRAM during a function call."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    result = fn(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return result, peak_mb


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load eval prompts
    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Evaluating on {len(prompts)} prompts")

    results = {
        "config": vars(args),
        "metrics": {},
    }

    # --- Model size comparison ---
    fp16_size = measure_model_size(
        os.path.join(args.model_id, "transformer", "diffusion_pytorch_model.safetensors")
    ) if os.path.exists(os.path.join(args.model_id, "transformer")) else None

    ternary_size = measure_model_size(args.ternary_checkpoint)
    results["metrics"]["ternary_checkpoint_mb"] = ternary_size
    if fp16_size:
        results["metrics"]["fp16_checkpoint_mb"] = fp16_size
        results["metrics"]["compression_ratio"] = fp16_size / ternary_size

    print(f"Ternary checkpoint: {ternary_size:.1f} MB")
    if fp16_size:
        print(f"FP16 checkpoint: {fp16_size:.1f} MB")
        print(f"Compression: {fp16_size / ternary_size:.1f}x")

    # --- Generate videos ---
    # TODO: implement full generation pipeline with:
    # 1. Load ternary model
    # 2. Generate videos for each prompt
    # 3. Save videos
    # 4. Compute VBench, FVD, CLIP scores

    print("\nVideo generation and metric computation to be implemented")
    print("See docs/evaluation.md for full evaluation protocol")

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
