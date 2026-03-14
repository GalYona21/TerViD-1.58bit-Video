"""
BitLinear: 1.58-bit (ternary) linear layer for Diffusion Transformers.

Based on:
- BitNet b1.58 (Microsoft): absmean quantization, STE training
- TerDiT: ternary DiT with RMS norm stabilization
- 1.58-bit FLUX: self-supervised distillation for T2I DiTs

Weights are constrained to {-1, 0, +1} with per-channel scaling factors.
During training, latent FP16 weights are maintained and quantized on-the-fly.
Gradients flow through the Round() via Straight-Through Estimator (STE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StraightThroughEstimator(torch.autograd.Function):
    """Round with straight-through gradient estimator.

    Forward: quantize weights to {-1, 0, +1}
    Backward: pass gradient through as-is (bypass the non-differentiable round)
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # Scale by absmean, round to nearest int, clamp to [-1, 1]
        w_scaled = w / (gamma + 1e-8)
        w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradient through unchanged
        return grad_output, None


def absmean_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to ternary {-1, 0, +1} using absmean scaling.

    For each output channel, compute gamma = mean(|w|), then:
        w_ternary = Clip(Round(w / gamma), -1, 1)
    Effective weight = w_ternary * gamma

    Args:
        w: weight tensor of shape (out_features, in_features)

    Returns:
        w_ternary: ternary weights {-1, 0, +1}
        gamma: per-channel scaling factors
    """
    # Per-channel (per output row) scaling factor
    gamma = w.abs().mean(dim=-1, keepdim=True)
    w_ternary = StraightThroughEstimator.apply(w, gamma)
    return w_ternary, gamma


def activation_quant_8bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to 8-bit using absmax per-token scaling.

    Args:
        x: activation tensor of shape (..., features)

    Returns:
        x_quant: quantized activations
        scale: per-token scaling factors
    """
    Qb = 127.0  # 8-bit range
    scale = x.abs().max(dim=-1, keepdim=True).values / Qb
    x_quant = torch.clamp(torch.round(x / (scale + 1e-8)), -Qb, Qb)
    return x_quant, scale


class _RMSNorm(nn.Module):
    """RMS normalization (compatible with PyTorch < 2.4 which lacks nn.RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms


class BitLinear(nn.Module):
    """1.58-bit linear layer with ternary weights and 8-bit activations.

    During training:
        - Maintains latent FP16/BF16 weights (self.weight)
        - Quantizes to ternary on-the-fly in forward pass
        - Gradients pass through via STE to update latent weights

    During inference:
        - Only ternary weights + scaling factors needed
        - Matrix multiply becomes integer addition/subtraction

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to include bias (kept in full precision)
        quantize_activations: whether to also quantize activations to 8-bit
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_activations: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_activations = quantize_activations

        # Latent full-precision weights (updated by optimizer)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMS norm for activation stabilization (from TerDiT)
        self.rms_norm = _RMSNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_pretrained_linear(cls, linear: nn.Linear, quantize_activations: bool = False) -> "BitLinear":
        """Initialize BitLinear from a pretrained nn.Linear layer.

        This is the "smart initialization" approach:
        - Copy the pretrained FP16 weights as latent weights
        - The absmean quantization will map them to {-1, 0, +1}
        - This preserves the pretrained knowledge as starting point for QAT
        """
        has_bias = linear.bias is not None
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            quantize_activations=quantize_activations,
        )
        # Copy pretrained weights as latent weights
        bit_linear.weight.data.copy_(linear.weight.data)
        if has_bias and linear.bias is not None:
            bit_linear.bias.data.copy_(linear.bias.data)
        return bit_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMS norm for activation stabilization (TerDiT)
        x = self.rms_norm(x)

        # Quantize weights to ternary: w_ternary in {-1,0,+1}, gamma shape (out, 1)
        w_ternary, gamma = absmean_quantize(self.weight)

        # Effective weight = w_ternary * gamma, shape (out, in)
        w_effective = w_ternary * gamma

        # Optionally quantize activations to 8-bit
        if self.quantize_activations and self.training:
            x_q, x_scale = activation_quant_8bit(x)
            out = F.linear(x_q, w_ternary)
            # Rescale: multiply by (x_scale * gamma) to get correct magnitudes
            out = out * gamma.squeeze(-1) * x_scale
        else:
            out = F.linear(x, w_effective)

        if self.bias is not None:
            out = out + self.bias

        return out

    def get_ternary_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the current ternary weights and scaling factors (for inference/saving)."""
        with torch.no_grad():
            gamma = self.weight.abs().mean(dim=-1, keepdim=True)
            w_ternary = torch.clamp(torch.round(self.weight / (gamma + 1e-8)), -1.0, 1.0)
        return w_ternary.to(torch.int8), gamma

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantize_activations={self.quantize_activations}"
        )


def replace_linear_with_bitlinear(
    model: nn.Module,
    target_modules: Optional[list[str]] = None,
    skip_modules: Optional[list[str]] = None,
    quantize_activations: bool = False,
) -> nn.Module:
    """Recursively replace nn.Linear layers with BitLinear in a model.

    Args:
        model: the model to modify (in-place)
        target_modules: if provided, only replace linears whose name contains one of these strings.
                        Example: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        skip_modules: if provided, skip linears whose name contains one of these strings.
                      Example: ["temporal_attn"] to keep temporal attention in full precision.
        quantize_activations: whether BitLinear layers should also quantize activations

    Returns:
        The modified model (same object, modified in-place)
    """
    replacements = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check skip list
        if skip_modules and any(skip in name for skip in skip_modules):
            continue

        # Check target list
        if target_modules and not any(target in name for target in target_modules):
            continue

        replacements[name] = BitLinear.from_pretrained_linear(
            module, quantize_activations=quantize_activations
        )

    # Apply replacements
    for name, new_module in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    n = len(replacements)
    total_linear = sum(1 for _, m in model.named_modules() if isinstance(m, (nn.Linear, BitLinear)))
    print(f"Replaced {n}/{n + (total_linear - n)} linear layers with BitLinear")

    return model
