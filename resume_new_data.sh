#!/usr/bin/env bash
# Resume (or start) the new-data training runs from their newest checkpoints.
# Idempotent: each dataset continues from its latest 10k checkpoint via --resume,
# or starts fresh if none exists. Safe to run after any stop. Same memory
# adaptations as train_new_data.sh (expandable_segments + 5.5Mpix cap).
set -u
cd /home/nwu/school/Fast-Image-Processing-FCN
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] RESUME div2k"
uv run train.py --dataset datasets/div2k --model CAN32+AN --iterations 500000 \
  --max-pixels 5500000 --resume >> logs/div2k_train.out 2>&1
echo "[$(date)] div2k exit=$?"

echo "[$(date)] RESUME flickr2k"
uv run train.py --dataset datasets/flickr2k --model CAN32+AN --iterations 500000 \
  --max-pixels 5500000 --resume >> logs/flickr2k_train.out 2>&1
echo "[$(date)] flickr2k exit=$?"

echo "[$(date)] ALL DONE"
