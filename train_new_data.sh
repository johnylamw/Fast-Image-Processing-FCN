#!/usr/bin/env bash
# Orchestrates the new-data variant training runs (DIV2K then Flickr2K),
# each with the same config as the Adobe5k CAN32+AN baseline (500k iters).
#
# Two memory adaptations vs the Adobe baseline, needed for native-2K images on
# a 32GB GPU. Neither affects Adobe/Flickr2K behaviour:
#   - expandable_segments: allocator setting only, no effect on training.
#   - --max-pixels 5500000: per-sample area cap; only binds on DIV2K's <1%
#     extreme-aspect panoramas (cap > Adobe/Flickr2K worst case, so a no-op there).
set -u
cd /home/nwu/school/Fast-Image-Processing-FCN
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] START div2k"
uv run train.py --dataset datasets/div2k --model CAN32+AN --iterations 500000 \
  --max-pixels 5500000 > logs/div2k_train.out 2>&1
echo "[$(date)] div2k exit=$?"

echo "[$(date)] START flickr2k"
uv run train.py --dataset datasets/flickr2k --model CAN32+AN --iterations 500000 \
  --max-pixels 5500000 > logs/flickr2k_train.out 2>&1
echo "[$(date)] flickr2k exit=$?"

echo "[$(date)] ALL DONE"
