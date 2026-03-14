"""
Smoke test: verify the full pipeline works end-to-end on CPU/MPS.

Uses a tiny LTXVideoTransformer3DModel config (not the full 2B model)
to validate BitLinear replacement, distillation, and gradient flow.

Usage:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Determine device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "cpu"  # MPS has issues with some ops, use CPU for smoke test
else:
    DEVICE = "cpu"

DTYPE = torch.float32  # CPU/MPS needs float32


def test_bitlinear():
    """Test BitLinear layer standalone."""
    print("=" * 60)
    print("TEST 1: BitLinear layer")
    print("=" * 60)

    from src.models.bitlinear import BitLinear, absmean_quantize

    # Test absmean quantization
    w = torch.randn(64, 128)
    w_ternary, gamma = absmean_quantize(w)
    unique = set(w_ternary.unique().tolist())
    assert unique.issubset({-1.0, 0.0, 1.0}), f"Expected ternary, got {unique}"
    print(f"  absmean_quantize: OK (unique values: {unique})")

    # Test BitLinear forward
    layer = BitLinear(128, 64, bias=True)
    x = torch.randn(2, 10, 128)
    out = layer(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
    print(f"  BitLinear forward: OK (output shape: {out.shape})")

    # Test gradient flow through STE
    x = torch.randn(2, 10, 128, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None, "No gradient on weight!"
    assert x.grad is not None, "No gradient on input!"
    print(f"  STE gradient flow: OK")

    # Test from_pretrained_linear
    linear = nn.Linear(128, 64)
    bit_linear = BitLinear.from_pretrained_linear(linear)
    assert torch.allclose(bit_linear.weight.data, linear.weight.data)
    print(f"  from_pretrained_linear: OK")

    # Test ternary weight extraction
    w_t, g = bit_linear.get_ternary_weights()
    assert w_t.dtype == torch.int8
    print(f"  get_ternary_weights: OK (dtype={w_t.dtype}, shape={w_t.shape})")

    print("  PASSED\n")


def test_replace_linear():
    """Test replacing nn.Linear with BitLinear in a model."""
    print("=" * 60)
    print("TEST 2: replace_linear_with_bitlinear")
    print("=" * 60)

    from src.models.bitlinear import BitLinear, replace_linear_with_bitlinear

    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )

    # Count original linear layers
    replace_linear_with_bitlinear(model)

    assert isinstance(model[0], BitLinear), "Layer 0 should be BitLinear"
    assert isinstance(model[2], BitLinear), "Layer 2 should be BitLinear"
    print(f"  All linear layers replaced: OK")

    # Test forward still works
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == (2, 32)
    print(f"  Forward after replacement: OK (shape: {out.shape})")

    # Test with skip_modules
    model2 = nn.ModuleDict({
        "spatial": nn.Linear(64, 32),
        "temporal": nn.Linear(64, 32),
    })
    replace_linear_with_bitlinear(model2, skip_modules=["temporal"])
    assert isinstance(model2["spatial"], BitLinear)
    assert isinstance(model2["temporal"], nn.Linear)
    print(f"  skip_modules: OK (temporal kept as nn.Linear)")

    print("  PASSED\n")


def test_tiny_dit_distillation():
    """Test full distillation pipeline with a tiny transformer."""
    print("=" * 60)
    print("TEST 3: Distillation with tiny LTX-Video transformer")
    print("=" * 60)

    from diffusers import LTXVideoTransformer3DModel
    from src.models.ternary_dit import TernaryDiTWrapper
    from src.training.distillation import DistillationTrainer

    # Create a TINY transformer config (not the full 2B)
    tiny_config = {
        "in_channels": 8,          # reduced from 128
        "out_channels": 8,
        "num_attention_heads": 4,   # reduced from 32
        "attention_head_dim": 16,   # reduced from 64
        "cross_attention_dim": 64,  # reduced from 2048
        "caption_channels": 64,    # must match text embedding dim
        "num_layers": 2,            # reduced from 28
    }

    print(f"  Creating tiny transformer (config: {tiny_config})...")
    teacher = LTXVideoTransformer3DModel(**tiny_config).to(dtype=DTYPE, device=DEVICE)
    student_base = LTXVideoTransformer3DModel(**tiny_config).to(dtype=DTYPE, device=DEVICE)

    # Copy teacher weights to student before ternarization
    student_base.load_state_dict(teacher.state_dict())

    # Wrap student for ternary training
    print(f"  Wrapping student with TernaryDiTWrapper...")
    student = TernaryDiTWrapper(student_base, strategy="full")

    stats = student.count_params()
    print(f"  Student params: {stats['total_params']:,} total, "
          f"{stats['ternary_ratio']:.1%} ternarized")

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": student.get_ternarized_params(), "lr": 5e-4},
            {"params": student.get_fullprecision_params(), "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )

    # Create trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        device=DEVICE,
        gradient_accumulation_steps=1,
    )

    # Create a fake batch
    B = 2
    latent_channels = tiny_config["in_channels"]
    latent_t, latent_h, latent_w = 2, 4, 4
    text_seq_len = 8
    cross_dim = tiny_config["cross_attention_dim"]

    noisy_latents = torch.randn(B, latent_channels, latent_t, latent_h, latent_w,
                                device=DEVICE, dtype=DTYPE)
    timesteps = torch.randint(0, 1000, (B,), device=DEVICE)
    text_emb = torch.randn(B, text_seq_len, cross_dim, device=DEVICE, dtype=DTYPE)
    encoder_attention_mask = torch.ones(B, text_seq_len, device=DEVICE, dtype=DTYPE)

    batch = {
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
        "text_embeddings": text_emb,
        "encoder_attention_mask": encoder_attention_mask,
        "num_frames": latent_t,
        "height": latent_h,
        "width": latent_w,
    }

    # Run a few training steps
    print(f"  Running 5 distillation steps...")
    losses = []
    for step in range(5):
        metrics = trainer.train_step_self_supervised(batch, step)
        losses.append(metrics["loss"])
        print(f"    Step {step}: loss={metrics['loss']:.6f}, lr={metrics['lr']:.2e}")

    # Verify loss is finite and decreasing (or at least not exploding)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "Loss went to inf/nan!"
    print(f"  Loss trend: {losses[0]:.6f} -> {losses[-1]:.6f}")

    # Verify ternary weights are actually ternary
    from src.models.bitlinear import BitLinear
    for name, module in student.model.named_modules():
        if isinstance(module, BitLinear):
            w_t, gamma = module.get_ternary_weights()
            unique = set(w_t.unique().tolist())
            assert unique.issubset({-1, 0, 1}), f"{name}: expected ternary, got {unique}"
    print(f"  Ternary weights verified: all {{-1, 0, +1}}")

    # Test checkpoint saving
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ternary.pt")
        student.save_ternary_checkpoint(ckpt_path)
        size_kb = os.path.getsize(ckpt_path) / 1024
        print(f"  Checkpoint saved: {size_kb:.1f} KB")

    print("  PASSED\n")


def main():
    print(f"\nTerViD Smoke Test")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"PyTorch: {torch.__version__}\n")

    test_bitlinear()
    test_replace_linear()
    test_tiny_dit_distillation()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe code is ready for GPU training. On your 3090s, run:")
    print("  accelerate launch --num_processes 2 scripts/03_train.py \\")
    print("    --mode self_supervised --model_id Lightricks/LTX-Video ...")


if __name__ == "__main__":
    main()
