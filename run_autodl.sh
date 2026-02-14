#!/usr/bin/env bash
set -euo pipefail

# AutoDL quick-start runner (edit paths if needed)

PROJ=/root/autodl-tmp/autorootsam
#PROJ="$(cd "$(dirname "$0")" && pwd)"
DATA=/root/autodl-tmp/PRMI
CFG=configs/sam2.1/sam2.1_hiera_l.yaml
CKPT=$PROJ/checkpoints/sam2.1_hiera_large.pt

export PYTHONPATH=$PROJ:$PYTHONPATH

echo "[1/5] Download SAM2.1 ckpt (if missing)"
cd "$PROJ"
python download_ckpt.py --out "$CKPT" || true

echo "[2/5] PRMI stats (Cotton example)"
python tools/prmi_stats.py \
  --json "$DATA/train/labels_image_gt/Cotton_736x552_DPI150_train.json" \
  --json "$DATA/val/labels_image_gt/Cotton_736x552_DPI150_val.json" \
  --json "$DATA/test/labels_image_gt/Cotton_736x552_DPI150_test.json" \
  --out "$PROJ/logs/stats_cotton_150.json"

echo "[3/5] Single-frame warmup (Cotton example)"
CUDA_VISIBLE_DEVICES=0 python train_single.py \
  --root_dir "$DATA" \
  --json "$DATA/train/labels_image_gt/Cotton_736x552_DPI150_train.json" \
  --backbone sam2 \
  --sam2_cfg "$CFG" \
  --sam2_ckpt "$CKPT" \
  --sam2_image_size 1024 \
  --tile 1024 --batch 4 --epochs 2 --num_workers 8 --amp \
  --out_dir "$PROJ/runs_single/cotton150"

echo "[4/5] TL training (Cotton example)"
CUDA_VISIBLE_DEVICES=0 python train_timelapse.py \
  --root_dir "$DATA" \
  --json "$DATA/train/labels_image_gt/Cotton_736x552_DPI150_train.json" \
  --backbone sam2 \
  --sam2_cfg "$CFG" \
  --sam2_ckpt "$CKPT" \
  --sam2_image_size 1024 \
  --tile 1024 --batch 2 --epochs 2 --num_workers 8 --amp \
  --tau_days 14.0 \
  --out_dir "$PROJ/runs_tl/cotton150"

echo "[5/5] Done. See RUNBOOK_AUTODL.md for full commands."
