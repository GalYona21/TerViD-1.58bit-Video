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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import math


class DistillationTrainer:
    """Trains a 1.58-bit student DiT to match a frozen FP16 teacher.

    Args:
        teacher: frozen FP16 video DiT model
        student: 1.58-bit (BitLinear) video DiT model
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
        3. Teacher predicts denoised output
        4. Student must match teacher's prediction

        Args:
            batch_size: number of samples
            num_frames: video frames (will be compressed by VAE)
            height: pixel height (will be compressed by VAE)
            width: pixel width (will be compressed by VAE)
            text_embeddings: precomputed text encoder outputs
            vae_latent_channels: latent channels from VAE
            spatial_compression: VAE spatial compression ratio
            temporal_compression: VAE temporal compression ratio

        Returns:
            dict with noisy_latents, timesteps, text_embeddings
        """
        # Compute latent dimensions
        latent_h = height // spatial_compression
        latent_w = width // spatial_compression
        latent_t = max(1, num_frames // temporal_compression)

        # Sample random latents (as if from VAE encoding)
        noisy_latents = torch.randn(
            batch_size, vae_latent_channels, latent_t, latent_h, latent_w,
            device=self.device, dtype=torch.float16,
        )

        # Sample timesteps with bias toward high-noise (where quantization error is worst)
        # Using Beta distribution to focus on high timesteps (from BitsFusion insight)
        beta_dist = torch.distributions.Beta(3.0, 1.0)
        t_normalized = beta_dist.sample((batch_size,)).to(self.device)
        timesteps = (t_normalized * 1000).long()

        # Select random text embeddings from the pool
        idx = torch.randint(0, text_embeddings.shape[0], (batch_size,))
        batch_text = text_embeddings[idx].to(self.device)

        return {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "text_embeddings": batch_text,
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

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_embeddings,
                timestep=timesteps,
            )
            if hasattr(teacher_out, "sample"):
                teacher_pred = teacher_out.sample
            else:
                teacher_pred = teacher_out

        # Student forward
        student_out = self.student(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_embeddings,
            timestep=timesteps,
        )
        if hasattr(student_out, "sample"):
            student_pred = student_out.sample
        else:
            student_pred = student_out

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
            teacher_out = self.teacher(
                hidden_states=noisy_latents,
                encoder_hidden_states=text_embeddings,
                timestep=timesteps,
            )
            teacher_pred = teacher_out.sample if hasattr(teacher_out, "sample") else teacher_out

        # Student forward
        student_out = self.student(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_embeddings,
            timestep=timesteps,
        )
        student_pred = student_out.sample if hasattr(student_out, "sample") else student_out

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
