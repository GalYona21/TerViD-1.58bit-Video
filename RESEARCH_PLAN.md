# 1.58-Bit Video Generation Model — Research Plan

## 1. The Opportunity (Why This Is a Paper)

There is a clear, **unexploited gap** in the literature:

| What exists | Bits | Modality | Key paper |
|---|---|---|---|
| BitNet b1.58 | 1.58-bit | LLMs | Microsoft, 2024 |
| TerDiT | 1.58-bit | Image DiT (ImageNet) | Cheng et al., 2024 |
| 1.58-bit FLUX | 1.58-bit | Image DiT (T2I) | Community, late 2024 |
| QVGen | 3-4 bit | Video DiT (QAT) | 2025 |
| BitsFusion | ~2 bit | Image UNet (SD 1.5) | 2024 |

**Nobody has done 1.58-bit (ternary) for video generation.**

QVGen only went down to 3-bit. TerDiT only did images. Extending ternary training to video DiTs is a natural, publishable next step with clear novelty.

### Publication Targets
- **ICME 2026** (Bangkok, July 6-10) — has an official "Low-Bit-Width Large Model Quantization" challenge track covering Text-to-Video. Perfect fit.
- **NeurIPS 2026** (deadline ~May 2026)
- **ICLR 2027** / **CVPR 2027**
- **ArXiv preprint** regardless

---

## 2. Base Model Selection

### Recommendation: **LTX-Video 2B** (primary) + AnimateDiff (fallback/comparison)

| Criterion | LTX-Video 2B | AnimateDiff |
|---|---|---|
| Architecture | DiT (same as TerDiT) | UNet + motion module |
| Parameters | ~2B | ~417M (motion module only) |
| License | Apache 2.0 | Apache 2.0 |
| Training code | Official trainer available | Community trainers |
| Quality | SOTA for size, real-time 30fps | Good but dated (SD 1.5 era) |
| TerDiT transfer | Direct (same arch family) | Requires adaptation |
| Novelty for paper | High (modern DiT video model) | Lower (older, UNet-based) |
| Ease of training | Medium (2B is manageable) | Easy (417M motion module) |

**Primary path: LTX-Video 2B** — DiT architecture means TerDiT's techniques (absmean quantization, RMS norm after adaLN, STE training) transfer almost directly. This gives the strongest paper.

**Fallback/warm-up: AnimateDiff motion module** — Only 417M params. Can serve as proof-of-concept on your 2x 3090 setup before scaling to LTX-Video. Could also be a comparison point in the paper.

### Why NOT LTX-2 / LTX-2.3
- 19B params (14B video + 5B audio) — too large for 2x 3090 training
- The original LTX-Video 2B is the sweet spot

---

## 3. Technical Approach

### 3.1 Core Method: QAT from Pretrained Weights (not from scratch)

As the Gemini chat correctly identified, training from scratch is infeasible on 2x 3090. Instead:

1. **Initialize from pretrained FP16/BF16 checkpoint** using absmean quantization:
   - For each weight matrix W: `gamma = mean(|W|)`
   - Ternary weights: `W_ternary = Clip(Round(W / gamma), -1, 1)`
   - Effective weights during forward: `W_ternary * gamma`

2. **BitLinear layer** with Straight-Through Estimator (STE):
   - Forward: use quantized ternary weights
   - Backward: gradients pass through to latent FP16 weights
   - Latent FP16 weights updated by optimizer, re-quantized each forward pass

3. **Knowledge Distillation** from frozen FP16 teacher:
   - Teacher: frozen pretrained LTX-Video 2B (FP16)
   - Student: 1.58-bit LTX-Video 2B (initialized from teacher)
   - Loss: MSE between teacher and student outputs (feature distillation)
   - This is how 1.58-bit FLUX was successfully created

### 3.2 Key Architectural Modifications (from TerDiT)

- **RMS Norm after adaLN**: TerDiT found this critical for stable ternary training in DiTs. The adaptive layer norm in DiTs produces large activation values that destabilize ternary networks. Adding RMS norm after adaLN's MLP fixes this.
- **Higher initial learning rate** (5e-4 vs standard 1e-4), then decay to 1e-4
- **Weight-only quantization**: weights to {-1, 0, +1}, activations stay 8-bit

### 3.3 Video-Specific Considerations

- **Temporal attention sensitivity**: As the Gemini chat noted, temporal attention layers may be more sensitive to extreme quantization than spatial layers. The paper should investigate:
  - (a) Full ternarization (all layers)
  - (b) Partial ternarization (spatial + FFN only, temporal in FP16)
  - (c) Mixed precision (temporal at 4-bit, rest at 1.58-bit)
  - Comparing these ablations would strengthen the paper significantly.

- **LTX-Video's VAE**: Keep the VAE in full precision. Only ternarize the DiT backbone. This is standard practice (BitsFusion, QVGen, TerDiT all do this).

### 3.4 Memory Budget on 2x 3090 (48GB total)

During QAT you need in memory:
- Student model FP16 latent weights: ~4GB
- Student model ternary weights (forward): ~0.25GB
- Teacher model FP16 (frozen, no grad): ~4GB
- Optimizer states (AdamW, 2x FP32 copies): ~16GB
- Activations/gradients for video: variable, large

**Required memory optimizations:**
- DeepSpeed ZeRO Stage 2 or 3 (shard optimizer across GPUs)
- Gradient checkpointing (trade compute for memory)
- Small batch size + gradient accumulation
- Short video clips initially (e.g., 16 frames at 256x256, then scale up)
- Mixed precision training (FP16 latent weights, FP32 master weights in optimizer)

**If 48GB is still too tight:**
- Use LoRA-style approach: freeze most of the student, only ternarize + train specific layers progressively
- Or rent cloud GPUs for the final training run (the paper just needs good results, not necessarily trained at home)

---

## 4. Experimental Plan

### Phase 1: Proof of Concept (Weeks 1-3)
**Goal**: Validate that ternary training works on a video DiT at all.

1. Implement BitLinear layer with STE in PyTorch
2. Apply to AnimateDiff motion module (417M params) as quick validation
3. Initialize from pretrained weights, distill from FP16 teacher
4. Train on a small video dataset (e.g., WebVid-2M subset)
5. Qualitative evaluation: does it generate coherent motion?

**Success criteria**: Generates recognizable video clips (doesn't need to be great)

### Phase 2: LTX-Video 2B Ternarization (Weeks 4-8)
**Goal**: Ternarize the LTX-Video 2B DiT backbone.

1. Fork LTX-Video-Trainer, integrate BitLinear layers
2. Add RMS norm after adaLN (TerDiT technique)
3. Set up knowledge distillation pipeline
4. Train on OpenVidHQ or similar open video dataset
5. Run ablations on partial vs full ternarization

### Phase 3: Evaluation & Benchmarking (Weeks 9-10)
**Goal**: Rigorous comparison for the paper.

Metrics:
- **VBench / VBench-2.0** (standard video generation benchmark)
- **FVD** (Frechet Video Distance)
- **CLIP score** (text-video alignment)
- **Inference speed** (tokens/sec, wall-clock time)
- **Memory usage** (peak VRAM during inference)
- **Model size** (checkpoint size in MB)

Comparisons:
- Full precision LTX-Video 2B (BF16)
- LTX-Video 2B at 4-bit (QVGen-style or PTQ)
- LTX-Video 2B at 1.58-bit (ours)
- Include partial ternarization ablations

### Phase 4: Paper Writing (Weeks 10-12)
**Goal**: Submit-ready manuscript.

Paper structure:
1. Introduction: efficiency crisis in video generation, success of 1.58-bit in LLMs/image DiTs, gap in video
2. Related work: BitNet, TerDiT, QVGen, BitsFusion
3. Method: BitLinear for video DiTs, architectural modifications, distillation
4. Experiments: benchmarks, ablations, efficiency measurements
5. Conclusion: first 1.58-bit video generation model

---

## 5. Expected Results & Contribution

Based on TerDiT (image) and BitNet (LLM) results, we can reasonably expect:

| Metric | FP16 baseline | 1.58-bit (expected) |
|---|---|---|
| Checkpoint size | ~4GB | ~0.5GB (8x reduction) |
| Inference VRAM | ~6GB | ~1-2GB |
| Quality (VBench) | 100% | 85-95% (based on TerDiT image results) |
| Inference speed* | 1x | Potentially 2-4x with kernel support |

*Speed gains require ternary kernel implementations (bitnet.cpp style). Even without custom kernels, the memory savings alone are a major contribution.

### Paper Title Ideas
- "TerViD: Ternary Video Diffusion Transformers for Efficient Video Generation"
- "1.58-Bit Video Generation: Extreme Quantization of Video Diffusion Transformers"
- "From Bits to Clips: Ternary Weight Training for Video Diffusion Models"

---

## 6. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Temporal coherence degrades at 1.58-bit | Medium | Partial ternarization (keep temporal attention in higher precision); this itself is a finding |
| Can't fit QAT on 2x 3090 | Medium | Start with AnimateDiff; use gradient checkpointing + DeepSpeed; rent cloud if needed |
| Results not competitive enough | Low-Medium | Even showing the gap and analyzing why is publishable; partial ternarization as fallback |
| Someone publishes this first | Low | TerDiT is from May 2024, still no video extension as of March 2026 — field is open |
| No custom kernels for speed benchmarks | High | Focus paper on memory/size reduction; note speed as future work with hardware support |

---

## 7. Key Resources & References

### Papers to Cite
- **BitNet b1.58**: "The Era of 1-bit LLMs" (Microsoft, 2024)
- **TerDiT**: "Ternary Diffusion Models with Transformers" (2024)
- **QVGen**: "Pushing the Limit of Quantized Video Generative Models" (2025)
- **BitsFusion**: "1.99 bits Weight Quantization of Diffusion Model" (2024)
- **LTX-Video**: "Realtime Video Latent Diffusion" (Lightricks, 2025)
- **1.58-bit FLUX**: Community work on ternary FLUX.1-dev (late 2024)

### Code Repositories
- Microsoft BitNet: https://github.com/microsoft/BitNet
- TerDiT: https://github.com/Lucky-Lance/TerDiT
- LTX-Video: https://github.com/Lightricks/LTX-Video
- LTX-Video Trainer: available via Lightricks
- AnimateDiff: https://github.com/guoyww/AnimateDiff

### Datasets
- OpenVidHQ-4M (used by QVGen)
- WebVid-2M / WebVid-10M
- Panda-70M (subset)

---

## 8. Repository Structure

```
VideoGenerationModel1_58bit/
├── src/
│   ├── models/
│   │   ├── bitlinear.py          # Core 1.58-bit linear layer (STE + absmean)
│   │   └── ternary_dit.py        # Wrapper to ternarize any video DiT
│   ├── training/
│   │   └── distillation.py       # Knowledge distillation trainer
│   ├── data/
│   │   └── video_dataset.py      # Dataset loaders (WebDataset + prompts-only)
│   ├── inference/                # TODO: inference pipeline
│   └── kernels/                  # TODO: custom ternary CUDA kernels
├── configs/
│   ├── train_selfsupervised.yaml # Self-supervised config (no data needed)
│   ├── train_datadriven.yaml     # Data-driven config
│   └── deepspeed_zero2.json      # DeepSpeed config for 2x 3090
├── scripts/
│   ├── 01_download_videos.sh     # Download videos via video2dataset
│   ├── 02_prepare_latents.py     # Pre-encode VAE latents + text embeddings
│   ├── 03_train.py               # Main training script (accelerate launch)
│   └── 04_evaluate.py            # Evaluation and benchmarking
├── data/
│   ├── prompts.txt               # Text prompts for self-supervised training
│   └── video_urls_example.csv    # Example CSV for video2dataset
├── tests/
│   └── test_bitlinear.py         # Unit tests for BitLinear
├── notebooks/                    # Jupyter notebooks for exploration
├── docs/                         # Documentation
├── RESEARCH_PLAN.md              # This file
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## 9. Step-by-Step Execution Plan

### Step 0: Environment Setup
```bash
# Create environment
conda create -n tervid python=3.11 -y
conda activate tervid

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify BitLinear works
python -m pytest tests/test_bitlinear.py -v
```

### Step 1: Self-Supervised PoC (No Dataset Needed)
**The fastest path to first results.** Based on 1.58-bit FLUX approach.

```bash
# Just need prompts (already provided in data/prompts.txt)
# Launch self-supervised training:
accelerate launch --num_processes 2 scripts/03_train.py \
    --mode self_supervised \
    --prompts_file data/prompts.txt \
    --model_id Lightricks/LTX-Video \
    --strategy full \
    --num_steps 10000 \
    --batch_size 1 \
    --lr 5e-4 \
    --num_frames 16 \
    --height 256 --width 256
```

### Step 2: Build Video Dataset (for stronger training)
```bash
# a) Collect video URLs into a CSV:
#    data/video_urls.csv with columns: url, caption

# b) Download + preprocess with video2dataset:
bash scripts/01_download_videos.sh

# c) Pre-encode into VAE latents (run once, saves VRAM during training):
python scripts/02_prepare_latents.py \
    --input_dir data/raw_videos \
    --output_dir data/latents \
    --model_id Lightricks/LTX-Video \
    --resolution 256 --num_frames 16
```

**Dataset sources for the CSV:**
- **Panda-70M** — 70M high-quality video-text pairs (use subset)
- **WebVid-2M** — 2.5M video-text pairs, widely used
- **OpenVidHQ-4M** — used by QVGen, high quality
- **InternVid** — 7M curated clips
- Or generate videos with the teacher model itself (true self-supervision)

### Step 3: Data-Driven Training
```bash
accelerate launch --num_processes 2 scripts/03_train.py \
    --mode data_driven \
    --data_dir data/latents \
    --model_id Lightricks/LTX-Video \
    --strategy full \
    --num_steps 50000 \
    --batch_size 1 \
    --lr 5e-4
```

### Step 4: Ablation Studies (for paper)
Run with different strategies:
```bash
# Full ternarization
accelerate launch scripts/03_train.py --strategy full ...

# Spatial-only (temporal attention in FP16)
accelerate launch scripts/03_train.py --strategy spatial_only ...

# FFN-only (all attention in FP16)
accelerate launch scripts/03_train.py --strategy ffn_only ...
```

### Step 5: Evaluate
```bash
python scripts/04_evaluate.py \
    --ternary_checkpoint checkpoints/final/ternary_model.pt \
    --compare_fp16
```

---

## 10. Key References

### Papers
- **BitNet b1.58**: "The Era of 1-bit LLMs" (Microsoft, 2024) — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **TerDiT**: "Ternary Diffusion Models with Transformers" (2024) — [arXiv:2405.14854](https://arxiv.org/abs/2405.14854)
- **1.58-bit FLUX**: ByteDance, 2024 — [arXiv:2412.18653](https://arxiv.org/abs/2412.18653)
- **QVGen**: "Pushing the Limit of Quantized Video Generative Models" (2025) — [arXiv:2505.11497](https://arxiv.org/abs/2505.11497)
- **BitsFusion**: "1.99 bits Weight Quantization of Diffusion Model" (2024) — [arXiv:2406.04333](https://arxiv.org/abs/2406.04333)
- **LTX-Video**: "Realtime Video Latent Diffusion" (Lightricks, 2025) — [arXiv:2501.00103](https://arxiv.org/abs/2501.00103)

### Code
- TerDiT: https://github.com/Lucky-Lance/TerDiT
- Microsoft BitNet: https://github.com/microsoft/BitNet
- LTX-Video: https://github.com/Lightricks/LTX-Video
- LTX-Video-Trainer: https://github.com/Lightricks/LTX-Video-Trainer
- video2dataset: https://github.com/iejMac/video2dataset
- 1.58-bit FLUX: https://github.com/Chenglin-Yang/1.58bit.flux (code "coming soon")
