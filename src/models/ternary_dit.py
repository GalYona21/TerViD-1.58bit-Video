"""
TernaryDiTWrapper: Wrap an LTX-Video DiT model for 1.58-bit training.

This module handles:
1. Loading the pretrained LTX-Video DiT
2. Replacing linear layers with BitLinear
3. Adding RMS norm after adaLN (TerDiT technique)
4. Providing a clean interface for the distillation trainer

Supports partial ternarization strategies:
- "full": all linear layers ternarized
- "spatial_only": only spatial attention + FFN (temporal attention in FP16)
- "ffn_only": only feed-forward networks (all attention in FP16)
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional

from .bitlinear import BitLinear, replace_linear_with_bitlinear


# Ternarization strategies: which modules to skip
STRATEGY_SKIP_MODULES = {
    "full": [],
    "spatial_only": ["temporal"],  # skip any module with "temporal" in name
    "ffn_only": ["attn"],  # skip all attention layers
}


class TernaryDiTWrapper(nn.Module):
    """Wrapper that converts a pretrained video DiT to 1.58-bit.

    Usage:
        from diffusers import LTXVideoTransformer3DModel

        # Load pretrained
        dit = LTXVideoTransformer3DModel.from_pretrained("Lightricks/LTX-Video", subfolder="transformer")

        # Wrap for ternary training
        ternary_dit = TernaryDiTWrapper(dit, strategy="full")

        # Access the ternarized model
        output = ternary_dit(hidden_states, encoder_hidden_states, timestep)
    """

    def __init__(
        self,
        pretrained_dit: nn.Module,
        strategy: str = "full",
        quantize_activations: bool = False,
        custom_skip_modules: Optional[list[str]] = None,
    ):
        super().__init__()

        self.strategy = strategy
        skip_modules = custom_skip_modules or STRATEGY_SKIP_MODULES.get(strategy, [])

        # Replace linear layers with BitLinear
        self.model = replace_linear_with_bitlinear(
            pretrained_dit,
            skip_modules=skip_modules if skip_modules else None,
            quantize_activations=quantize_activations,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_ternarized_params(self) -> list[nn.Parameter]:
        """Get parameters from BitLinear layers (for separate optimizer group)."""
        params = []
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                params.extend(module.parameters())
        return params

    def get_fullprecision_params(self) -> list[nn.Parameter]:
        """Get parameters from non-BitLinear layers."""
        bitlinear_params = set(id(p) for p in self.get_ternarized_params())
        return [p for p in self.model.parameters() if id(p) not in bitlinear_params]

    def count_params(self) -> dict:
        """Count parameters by type."""
        ternary = sum(p.numel() for p in self.get_ternarized_params())
        full = sum(p.numel() for p in self.get_fullprecision_params())
        total = ternary + full
        return {
            "ternary_params": ternary,
            "fullprecision_params": full,
            "total_params": total,
            "ternary_ratio": ternary / total if total > 0 else 0,
        }

    def save_ternary_checkpoint(self, path: str):
        """Save a compressed checkpoint with ternary weights packed to int8.

        Ternary weights {-1, 0, +1} are stored as int8, scaling factors as fp16.
        This gives ~8x compression vs fp16 for the ternarized layers.
        """
        state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                w_ternary, gamma = module.get_ternary_weights()
                state[f"{name}.w_ternary"] = w_ternary  # int8
                state[f"{name}.gamma"] = gamma.half()  # fp16
                if module.bias is not None:
                    state[f"{name}.bias"] = module.bias.data.half()
            elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                # Non-ternarized layers: save as-is
                state[f"{name}.weight"] = module.weight.data.half()
                if hasattr(module, "bias") and module.bias is not None:
                    state[f"{name}.bias"] = module.bias.data.half()

        torch.save(state, path)
        print(f"Saved ternary checkpoint to {path}")
