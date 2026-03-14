"""
Knowledge Distillation trainer for 1.58-bit video DiT.

Based on the approach from 1.58-bit FLUX and QVGen:
- Frozen FP16 teacher model
- 1.58-bit student model (initialized from teacher)
- Self-supervised: teacher generates targets, student learns to match
- Feature-level MSE loss between teacher and student outputs

Two training modes:
1. Self-supervised (no video data needed): run diffusion on random noise + prompts,
   match teacher outputs. This is how 1.58-bit FLUX works.
2. Data-driven distillation: use actual video data for training signal.
   Stronger but requires dataset preparation.

LTX-Video transformer forward signature:
    forward(
        hidden_states,           # (B, seq_len, inner_dim) — patchified latents
        encoder_hidden_states,   # (B, text_seq_len, cross_attn_dim)
        timestep,                # (B,) LongTensor
        encoder_attention_mask,  # (B, text_seq_len)
        num_frames, height, width,  # ints for rotary embeddings
        ...
    ) -> Transformer2DModelOutput(sample=...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import math


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Convert (B, C, T, H, W) latents to (B, seq_len, patch_channels) for the transformer.

    This replicates LTXPipeline._pack_latents from diffusers.
    With patch_size=1 and patch_size_t=1, this is simply a reshape:
        (B, C, T, H, W) -> (B, T*H*W, C)
    """
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size, -1,
        post_patch_num_frames, patch_size_t,
        post_patch_height, patch_size,
        post_patch_width, patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int,
    patch_size: int = 1, patch_size_t: int = 1,
) -> torch.Tensor:
    """Convert (B, seq_len, patch_channels) back to (B, C, T, H, W).

    This replicates LTXPipeline._unpack_latents from diffusers.
    """
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _call_transformer(model, hidden_states_5d, encoder_hidden_states, timestep,
                      encoder_attention_mask, num_frames, height, width):
    """Call LTX-Video transformer with correct input format.

    The transformer expects PACKED latents: (B, seq_len, C), not (B, C, T, H, W).
    The pipeline handles this packing; we replicate it here.
    """
    # Pack: (B, C, T, H, W) -> (B, T*H*W, C)
    hidden_states = _pack_latents(hidden_states_5d)

    out = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        return_dict=True,
    )
    output = out.sample if hasattr(out, "sample") else out

    # Unpack back: (B, T*H*W, C) -> (B, C, T, H, W)
    output = _unpack_latents(output, num_frames, height, width)
    return output


class DistillationTrainer:
    """Trains a 1.58-bit student DiT to match a frozen FP16 teacher.

    Args:
        teacher: frozen FP16 video DiT model (LTXVideoTransformer3DModel)
        student: 1.58-bit (BitLinear) video DiT model (TernaryDiTWrapper)
        optimizer: optimizer for student parameters
        scheduler: optional LR scheduler
        device: training device
        gradient_accumulation_steps: accumulate gradients over N steps
        max_grad_norm: gradient clipping threshold
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",  # "mse" or "huber"
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.loss_type = loss_type
        self.dtype = next(self.teacher.parameters()).dtype

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def compute_distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-level distillation loss."""
        if self.loss_type == "mse":
            return F.mse_loss(student_output, teacher_output)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(student_output, teacher_output)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    @torch.no_grad()
    def generate_self_supervised_batch(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        text_embeddings: torch.Tensor,
        vae_latent_channels: int = 128,
        spatial_compression: int = 32,
        temporal_compression: int = 8,
    ) -> dict:
        """Generate a self-supervised training batch (no video data needed).

        Follows 1.58-bit FLUX approach:
        1. Sample random noise in latent space
        2. Sample random diffusion timestep
        3. Teacher predicts denoised output → student must match

        LTX-Video VAE compression: 32x32 spatial, 8x temporal.
        Latent channels: 128.
        """
        # Compute latent dimensions after VAE compression
        latent_h = height // spatial_compression
        latent_w = width // spatial_compression
        latent_t = max(1, num_frames // temporal_compression)

        # Sample random latents — shape matches what LTX-Video transformer expects
        # The transformer's proj_in handles (B, C, T, H, W) -> (B, seq, dim)
        noisy_latents = torch.randn(
            batch_size, vae_latent_channels, latent_t, latent_h, latent_w,
            device=self.device, dtype=self.dtype,
        )

        # Sample timesteps with bias toward high-noise (from BitsFusion)
        beta_dist = torch.distributions.Beta(3.0, 1.0)
        t_normalized = beta_dist.sample((batch_size,)).to(self.device)
        timesteps = (t_normalized * 1000).long()

        # Select random text embeddings from the pool
        idx = torch.randint(0, text_embeddings.shape[0], (batch_size,))
        batch_text = text_embeddings[idx].to(device=self.device, dtype=self.dtype)

        # Attention mask: all ones (no masking) for the text sequence
        encoder_attention_mask = torch.ones(
            batch_size, batch_text.shape[1],
            device=self.device, dtype=self.dtype,
        )

        return {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "text_embeddings": batch_text,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": latent_t,
            "height": latent_h,
            "width": latent_w,
        }

    def train_step_self_supervised(
        self,
        batch: dict,
        step: int,
    ) -> dict:
        """Single self-supervised training step.

        Teacher and student both process the same noisy latents + text.
        Student learns to match teacher's output.
        """
        noisy_latents = batch["noisy_latents"]
        timesteps = batch["timesteps"]
        text_embeddings = batch["text_embeddings"]
        encoder_attention_mask = batch["encoder_attention_mask"]
        num_frames = batch["num_frames"]
        height = batch["height"]
        width = batch["width"]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_pred = _call_transformer(
                self.teacher, noisy_latents, text_embeddings, timesteps,
                encoder_attention_mask, num_frames, height, width,
            )

        # Student forward
        student_pred = _call_transformer(
            self.student, noisy_latents, text_embeddings, timesteps,
            encoder_attention_mask, num_frames, height, width,
        )

        # Distillation loss
        loss = self.compute_distillation_loss(student_pred, teacher_pred)
        loss = loss / self.grad_accum_steps

        # Backward
        loss.backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % self.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        return {
            "loss": loss.item() * self.grad_accum_steps,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train_step_data_driven(
        self,
        noisy_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        step: int,
    ) -> dict:
        """Training step using actual video data.

        Uses both:
        1. Noise prediction loss (standard diffusion training)
        2. Feature distillation from teacher
        Combined with weighting factor alpha.
        """
        # Teacher forward
        with torch.no_grad():
            teacher_pred = _call_transformer(
                self.teacher, noisy_latents, text_embeddings, timesteps,
                encoder_attention_mask, num_frames, height, width,
            )

        # Student forward
        student_pred = _call_transformer(
            self.student, noisy_latents, text_embeddings, timesteps,
            encoder_attention_mask, num_frames, height, width,
        )

        # Combined loss: noise prediction + distillation
        noise_loss = F.mse_loss(student_pred, noise)
        distill_loss = self.compute_distillation_loss(student_pred, teacher_pred)

        alpha = 0.5  # balance between noise prediction and distillation
        loss = (alpha * noise_loss + (1 - alpha) * distill_loss) / self.grad_accum_steps

        loss.backward()

        if (step + 1) % self.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        return {
            "loss": loss.item() * self.grad_accum_steps,
            "noise_loss": noise_loss.item(),
            "distill_loss": distill_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
