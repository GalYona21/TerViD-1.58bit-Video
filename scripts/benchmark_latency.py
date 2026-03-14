"""
Benchmark: compare inference latency of FP16 vs 1.58-bit (ternary) LTX-Video transformer.

Uses a tiny config by default for quick testing, or the full 2B model with --full.
Measures wall-clock time, memory usage, and throughput for both models.

Usage:
    python scripts/benchmark_latency.py              # tiny model, quick test
    python scripts/benchmark_latency.py --full        # full 2B model (needs ~20GB VRAM)
    python scripts/benchmark_latency.py --device cpu  # force CPU
"""

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.ternary_dit import TernaryDiTWrapper
from src.training.distillation import _pack_latents


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS has issues with some ops
    return "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """Estimate model memory footprint in MB."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def build_tiny_models(device, dtype):
    from diffusers import LTXVideoTransformer3DModel

    config = {
        "in_channels": 8,
        "out_channels": 8,
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "cross_attention_dim": 64,
        "caption_channels": 64,
        "num_layers": 2,
    }
    latent_channels = config["in_channels"]
    cross_dim = config["cross_attention_dim"]
    latent_t, latent_h, latent_w = 2, 4, 4
    text_seq_len = 8

    fp16_model = LTXVideoTransformer3DModel(**config).to(device=device, dtype=dtype)
    ternary_base = LTXVideoTransformer3DModel(**config).to(device=device, dtype=dtype)
    ternary_base.load_state_dict(fp16_model.state_dict())
    ternary_model = TernaryDiTWrapper(ternary_base, strategy="full").to(device)

    return fp16_model, ternary_model, latent_channels, cross_dim, latent_t, latent_h, latent_w, text_seq_len


def build_full_models(device, dtype):
    from diffusers import LTXVideoTransformer3DModel

    print("  Loading full LTX-Video 2B model (this may take a minute)...")
    fp16_model = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video", subfolder="transformer", torch_dtype=dtype
    ).to(device)

    print("  Creating ternary copy...")
    ternary_base = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video", subfolder="transformer", torch_dtype=dtype
    ).to(device)
    ternary_model = TernaryDiTWrapper(ternary_base, strategy="full").to(device)

    latent_channels = 128
    cross_dim = 2048
    latent_t, latent_h, latent_w = 2, 8, 8
    text_seq_len = 16

    return fp16_model, ternary_model, latent_channels, cross_dim, latent_t, latent_h, latent_w, text_seq_len


def benchmark_model(model, inputs, num_warmup=5, num_runs=50, device="cuda"):
    """Benchmark a single model's forward pass latency."""
    model.eval()

    hidden_states, encoder_hidden_states, timesteps, encoder_attention_mask, num_frames, height, width = inputs

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                return_dict=True,
            )
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure peak memory after warmup
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                return_dict=True,
            )

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    peak_mem_mb = 0
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    times_ms = [t * 1000 for t in times]
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP16 vs 1.58-bit latency")
    parser.add_argument("--full", action="store_true", help="Use full 2B model (needs ~20GB VRAM)")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num_runs", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    device = args.device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    mode = "FULL 2B" if args.full else "TINY (sanity check)"
    print(f"\n{'='*70}")
    print(f"  Latency Benchmark: FP16 vs 1.58-bit LTX-Video Transformer")
    print(f"  Mode: {mode}")
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Warmup: {args.num_warmup}, Runs: {args.num_runs}, Batch: {args.batch_size}")
    print(f"{'='*70}\n")

    # Build models
    print("Building models...")
    if args.full:
        (fp16_model, ternary_model, latent_channels, cross_dim,
         latent_t, latent_h, latent_w, text_seq_len) = build_full_models(device, dtype)
    else:
        (fp16_model, ternary_model, latent_channels, cross_dim,
         latent_t, latent_h, latent_w, text_seq_len) = build_tiny_models(device, dtype)

    # Print model stats
    fp16_params = count_parameters(fp16_model)
    ternary_params = count_parameters(ternary_model)
    fp16_size = get_model_size_mb(fp16_model)
    ternary_size = get_model_size_mb(ternary_model)

    print(f"\n  FP16 model:    {fp16_params:>12,} params, {fp16_size:>8.1f} MB")
    print(f"  Ternary model: {ternary_params:>12,} params, {ternary_size:>8.1f} MB")

    # Build inputs (packed latents as transformer expects)
    B = args.batch_size
    noisy_latents_5d = torch.randn(B, latent_channels, latent_t, latent_h, latent_w,
                                    device=device, dtype=dtype)
    hidden_states = _pack_latents(noisy_latents_5d)  # (B, T*H*W, C)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    encoder_hidden_states = torch.randn(B, text_seq_len, cross_dim, device=device, dtype=dtype)
    encoder_attention_mask = torch.ones(B, text_seq_len, device=device, dtype=dtype)

    inputs = (hidden_states, encoder_hidden_states, timesteps,
              encoder_attention_mask, latent_t, latent_h, latent_w)

    # Benchmark FP16
    print(f"\nBenchmarking FP16 model ({args.num_runs} runs)...")
    if device == "cuda":
        torch.cuda.empty_cache()
    fp16_results = benchmark_model(fp16_model, inputs, args.num_warmup, args.num_runs, device)

    # Free FP16 model to save memory
    del fp16_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Benchmark Ternary
    print(f"Benchmarking Ternary model ({args.num_runs} runs)...")
    ternary_results = benchmark_model(ternary_model, inputs, args.num_warmup, args.num_runs, device)

    del ternary_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Report
    speedup = fp16_results["mean_ms"] / ternary_results["mean_ms"] if ternary_results["mean_ms"] > 0 else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Metric':<25} {'FP16':>12} {'Ternary':>12} {'Ratio':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Mean latency (ms)':<25} {fp16_results['mean_ms']:>12.2f} {ternary_results['mean_ms']:>12.2f} {speedup:>9.2f}x")
    print(f"  {'Median latency (ms)':<25} {fp16_results['median_ms']:>12.2f} {ternary_results['median_ms']:>12.2f}")
    print(f"  {'Min latency (ms)':<25} {fp16_results['min_ms']:>12.2f} {ternary_results['min_ms']:>12.2f}")
    print(f"  {'Std dev (ms)':<25} {fp16_results['std_ms']:>12.2f} {ternary_results['std_ms']:>12.2f}")

    if device == "cuda":
        mem_ratio = ternary_results["peak_mem_mb"] / fp16_results["peak_mem_mb"] if fp16_results["peak_mem_mb"] > 0 else 0
        print(f"  {'Peak VRAM (MB)':<25} {fp16_results['peak_mem_mb']:>12.1f} {ternary_results['peak_mem_mb']:>12.1f} {mem_ratio:>9.2f}x")

    print(f"\n  Model size (params):    FP16={fp16_params:,}  Ternary={ternary_params:,}")
    print(f"  Model size (memory):    FP16={fp16_size:.1f} MB  Ternary={ternary_size:.1f} MB")

    if speedup > 1:
        print(f"\n  >> Ternary is {speedup:.2f}x FASTER than FP16")
    elif speedup < 1 and speedup > 0:
        print(f"\n  >> Ternary is {1/speedup:.2f}x SLOWER than FP16")
        print(f"     (Expected: ternary weights are still FP32 during training.")
        print(f"      Real speedup comes from custom ternary kernels at inference.)")
    else:
        print(f"\n  >> Similar performance")

    print(f"\n  NOTE: This benchmark uses PyTorch F.linear with float weights.")
    print(f"  The ternary model stores {-1,0,+1} weights but still uses float matmul.")
    print(f"  True speedup requires custom CUDA kernels that replace multiply-accumulate")
    print(f"  with addition/subtraction — expected 2-5x speedup at deployment.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
