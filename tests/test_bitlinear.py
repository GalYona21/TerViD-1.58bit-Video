"""
Tests for the BitLinear layer — the core building block.

Run: python -m pytest tests/test_bitlinear.py -v
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bitlinear import BitLinear, absmean_quantize, replace_linear_with_bitlinear


class TestAbsmeanQuantize:
    def test_output_is_ternary(self):
        """Weights should only be {-1, 0, +1}."""
        w = torch.randn(64, 128)
        w_ternary, gamma = absmean_quantize(w)
        unique = torch.unique(w_ternary)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())

    def test_gamma_is_positive(self):
        """Scaling factor should be positive."""
        w = torch.randn(64, 128)
        _, gamma = absmean_quantize(w)
        assert (gamma > 0).all()

    def test_gamma_shape(self):
        """Gamma should be per output channel."""
        w = torch.randn(64, 128)
        _, gamma = absmean_quantize(w)
        assert gamma.shape == (64, 1)


class TestBitLinear:
    def test_forward_shape(self):
        """Output shape should match nn.Linear."""
        layer = BitLinear(128, 64)
        x = torch.randn(2, 10, 128)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_from_pretrained(self):
        """Should initialize from pretrained nn.Linear."""
        linear = nn.Linear(128, 64)
        bit_linear = BitLinear.from_pretrained_linear(linear)
        assert bit_linear.weight.shape == linear.weight.shape
        assert torch.allclose(bit_linear.weight.data, linear.weight.data)

    def test_gradients_flow(self):
        """STE should allow gradients to flow through quantization."""
        layer = BitLinear(128, 64)
        x = torch.randn(2, 10, 128, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape

    def test_ternary_weights_are_int8(self):
        """get_ternary_weights should return int8 tensors."""
        layer = BitLinear(128, 64)
        w_ternary, gamma = layer.get_ternary_weights()
        assert w_ternary.dtype == torch.int8
        assert set(w_ternary.unique().tolist()).issubset({-1, 0, 1})

    def test_no_bias(self):
        """Should work without bias."""
        layer = BitLinear(128, 64, bias=False)
        x = torch.randn(2, 10, 128)
        out = layer(x)
        assert out.shape == (2, 10, 64)
        assert layer.bias is None


class TestReplaceLinear:
    def test_replace_all(self):
        """Should replace all nn.Linear with BitLinear."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        replace_linear_with_bitlinear(model)
        assert isinstance(model[0], BitLinear)
        assert isinstance(model[2], BitLinear)

    def test_skip_modules(self):
        """Should skip specified modules."""
        model = nn.ModuleDict({
            "spatial": nn.Linear(128, 64),
            "temporal": nn.Linear(128, 64),
        })
        replace_linear_with_bitlinear(model, skip_modules=["temporal"])
        assert isinstance(model["spatial"], BitLinear)
        assert isinstance(model["temporal"], nn.Linear)  # skipped

    def test_target_modules(self):
        """Should only replace targeted modules."""
        model = nn.ModuleDict({
            "attn": nn.Linear(128, 64),
            "ffn": nn.Linear(128, 64),
        })
        replace_linear_with_bitlinear(model, target_modules=["ffn"])
        assert isinstance(model["attn"], nn.Linear)  # not targeted
        assert isinstance(model["ffn"], BitLinear)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
