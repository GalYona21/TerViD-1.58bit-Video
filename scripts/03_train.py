"""
Step 3: Train the 1.58-bit video DiT via knowledge distillation.

Supports two modes:
  --mode self_supervised : no video data, teacher generates targets from noise+prompts
  --mode data_driven    : uses precomputed latents from step 2

Usage:
    # Self-supervised (easiest start, no dataset needed):
    accelerate launch --num_processes 2 scripts/03_train.py \
        --mode self_supervised \
        --prompts_file data/prompts.txt \
        --model_id Lightricks/LTX-Video \
        --strategy full \
        --num_steps 50000 \
        --batch_size 1 \
        --lr 5e-4

    # Data-driven (better quality, needs prepared latents):
    accelerate launch --num_processes 2 scripts/03_train.py \
        --mode data_driven \
        --data_dir data/latents \
        --model_id Lightricks/LTX-Video \
        --strategy full \
        --num_steps 50000 \
        --batch_size 1 \
        --lr 5e-4
"""

import argparse
import os
import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ternary_dit import TernaryDiTWrapper
from src.training.distillation import DistillationTrainer
from src.data.video_dataset import PrecomputedLatentDataset, PromptOnlyDataset, create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1.58-bit video DiT")
    # Mode
    parser.add_argument("--mode", type=str, choices=["self_supervised", "data_driven"], default="self_supervised")

    # Model
    parser.add_argument("--model_id", type=str, default="Lightricks/LTX-Video")
    parser.add_argument("--strategy", type=str, default="full",
                        choices=["full", "spatial_only", "ffn_only"])

    # Data
    parser.add_argument("--data_dir", type=str, default="data/latents")
    parser.add_argument("--prompts_file", type=str, default="data/prompts.txt")

    # Training
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # Video dimensions (for self-supervised mode)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--resume_from", type=str, default=None)

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"])

    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_min=1e-4):
    """Cosine annealing with warmup (from TerDiT)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        base_lr = optimizer.defaults["lr"]
        return max(lr_min / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    # Use accelerate for multi-GPU
    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training 1.58-bit video DiT")
        print(f"  Mode: {args.mode}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Model: {args.model_id}")
        print(f"  Steps: {args.num_steps}")
        print(f"  LR: {args.lr} -> {args.lr_min}")
        print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation_steps} accum")

    # --- Load models ---
    print("Loading pretrained DiT...")
    from diffusers import LTXVideoTransformer3DModel

    # Teacher: frozen FP16
    teacher = LTXVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    )

    # Student: initialized from teacher, then ternarized
    student_base = LTXVideoTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.float16
    )
    student = TernaryDiTWrapper(student_base, strategy=args.strategy)

    if accelerator.is_main_process:
        stats = student.count_params()
        print(f"  Ternary params: {stats['ternary_params']:,} ({stats['ternary_ratio']:.1%})")
        print(f"  Full precision params: {stats['fullprecision_params']:,}")

    # Gradient checkpointing
    if args.gradient_checkpointing and hasattr(student.model, "enable_gradient_checkpointing"):
        student.model.enable_gradient_checkpointing()
        print("  Gradient checkpointing: enabled")

    # --- Optimizer ---
    # TerDiT uses higher LR (5e-4) for ternary params
    optimizer = torch.optim.AdamW(
        [
            {"params": student.get_ternarized_params(), "lr": args.lr},
            {"params": student.get_fullprecision_params(), "lr": args.lr * 0.1},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.num_steps, lr_min=args.lr_min
    )

    # --- Data ---
    if args.mode == "data_driven":
        dataset = PrecomputedLatentDataset(args.data_dir)
        dataloader = create_dataloader(dataset, batch_size=args.batch_size)
        if accelerator.is_main_process:
            print(f"  Dataset: {len(dataset)} precomputed samples")
    else:
        # Self-supervised: load prompts and precompute text embeddings
        prompt_dataset = PromptOnlyDataset(
            args.prompts_file,
            cache_dir=os.path.join(args.output_dir, "cache"),
        )
        if prompt_dataset.embeddings is None:
            from transformers import T5EncoderModel, T5Tokenizer
            text_encoder = T5EncoderModel.from_pretrained(
                args.model_id, subfolder="text_encoder", torch_dtype=torch.float16
            )
            tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
            prompt_dataset.encode_prompts(text_encoder, tokenizer, device=str(device))
            del text_encoder, tokenizer
            torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print(f"  Prompts: {len(prompt_dataset)}")

    # --- Prepare with accelerate ---
    teacher = teacher.to(device).eval()
    student, optimizer, scheduler = accelerator.prepare(student, optimizer, scheduler)
    if args.mode == "data_driven":
        dataloader = accelerator.prepare(dataloader)

    # --- Trainer ---
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )

    # --- Training loop ---
    if accelerator.is_main_process:
        print("\n=== Starting training ===\n")

    global_step = 0
    if args.mode == "data_driven":
        data_iter = iter(dataloader)

    while global_step < args.num_steps:
        if args.mode == "self_supervised":
            batch = trainer.generate_self_supervised_batch(
                batch_size=args.batch_size,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                text_embeddings=prompt_dataset.embeddings,
            )
            metrics = trainer.train_step_self_supervised(batch, global_step)
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            latents = batch["latents"].to(device)
            text_emb = batch["text_emb"].to(device)
            B = latents.shape[0]

            # Compute latent spatial/temporal dims for transformer
            _, _, lt, lh, lw = latents.shape

            metrics = trainer.train_step_self_supervised(
                {
                    "noisy_latents": latents,
                    "timesteps": torch.randint(0, 1000, (B,), device=device),
                    "text_embeddings": text_emb,
                    "encoder_attention_mask": torch.ones(
                        B, text_emb.shape[1], device=device, dtype=text_emb.dtype,
                    ),
                    "num_frames": lt,
                    "height": lh,
                    "width": lw,
                },
                global_step,
            )

        global_step += 1

        # Logging
        if accelerator.is_main_process and global_step % args.log_every == 0:
            print(f"Step {global_step}/{args.num_steps} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}")

        # Save checkpoint
        if accelerator.is_main_process and global_step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{global_step}")
            os.makedirs(ckpt_path, exist_ok=True)
            unwrapped = accelerator.unwrap_model(student)
            unwrapped.save_ternary_checkpoint(os.path.join(ckpt_path, "ternary_model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_ternary_checkpoint(os.path.join(final_path, "ternary_model.pt"))
        print(f"\n=== Training complete! Final model saved to {final_path} ===")


if __name__ == "__main__":
    main()
