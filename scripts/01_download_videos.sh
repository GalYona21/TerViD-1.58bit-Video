#!/bin/bash
# Step 1: Download and preprocess videos using video2dataset
#
# Prerequisite: pip install video2dataset
#
# Input: CSV file with columns: url, caption
# Output: WebDataset shards in data/raw_videos/
#
# Adjust --resize and --fps for your VRAM budget:
#   - 256x256 @ 8fps: minimal VRAM for proof-of-concept
#   - 512x512 @ 16fps: better quality, needs more VRAM
#   - 768x768 @ 24fps: high quality, requires gradient checkpointing

set -e

DATA_DIR="data/raw_videos"
INPUT_CSV="data/video_urls.csv"
RESOLUTION=256
FPS=8
NUM_FRAMES=16  # ~2 seconds at 8fps

echo "=== Downloading videos with video2dataset ==="
echo "Resolution: ${RESOLUTION}x${RESOLUTION}, FPS: ${FPS}, Frames: ${NUM_FRAMES}"

video2dataset \
    --url_list="${INPUT_CSV}" \
    --input_format="csv" \
    --url_col="url" \
    --caption_col="caption" \
    --output_folder="${DATA_DIR}" \
    --output_format="webdataset" \
    --encode_formats='{"video": "mp4"}' \
    --video_size="${RESOLUTION}" \
    --video_fps="${FPS}" \
    --number_sample_per_shard=100 \
    --processes_count=4 \
    --thread_count=16 \
    --resize_mode="scale" \
    --min_resolution="${RESOLUTION}" \
    --max_aspect_ratio=2.0

echo "=== Done. Videos saved to ${DATA_DIR} ==="
echo "Next step: python scripts/02_prepare_latents.py"
