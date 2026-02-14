# AutoDL Runbook (path-aligned)

This runbook assumes you want the project to live at:

`/root/autodl-tmp/autorootsam/`

and your PRMI dataset is located at:

`/root/autodl-tmp/PRMI/`  (i.e. it contains `train/`, `val/`, `test/`)

It also assumes you have installed the **latest `sam2`** (pip editable or pip install) and you will use the **built-in config path**:

`configs/sam2.1/sam2.1_hiera_l.yaml`

## 0) One-time setup

```bash
# (A) Create project directory
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp

# (B) Unzip the project here (replace with the zip file you downloaded)
unzip -o autorootsam_autodl.zip -d /root/autodl-tmp/

# (C) Create common folders (already included, but safe)
mkdir -p /root/autodl-tmp/autorootsam/{checkpoints,runs_single,runs_tl,outputs,logs,data/PRMI}

# (D) Optional: link your dataset (recommended)
# If your dataset is already at /root/autodl-tmp/PRMI, link it into the project:
ln -sfn /root/autodl-tmp/PRMI /root/autodl-tmp/autorootsam/data/PRMI

# (E) Create venv (optional) + install requirements
cd /root/autodl-tmp/autorootsam
python -m pip install -U pip
pip install -r requirements.txt

# (F) Install sam2 (only if not installed yet)
# Option 1: pip install from github (recommended)
pip install "git+https://github.com/facebookresearch/sam2.git"

# Option 2: if you already cloned sam2 repo:
# pip install -e /path/to/sam2
```

## 1) Download SAM2.1 checkpoint

```bash
cd /root/autodl-tmp/autorootsam
python download_ckpt.py --out checkpoints/sam2.1_hiera_large.pt
```

## 2) Verify PRMI JSON statistics (train/val/test)

```bash
cd /root/autodl-tmp/autorootsam

# Example: Cotton only (train+val+test)
python tools/prmi_stats.py \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Cotton_736x552_DPI150_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Cotton_736x552_DPI150_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Cotton_736x552_DPI150_test.json \
  --out logs/stats_cotton_150.json

# All 7 subsets, all splits (train+val+test)
python tools/prmi_stats.py \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Cotton_736x552_DPI150_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Cotton_736x552_DPI150_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Cotton_736x552_DPI150_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Papaya_736x552_DPI150_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Papaya_736x552_DPI150_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Papaya_736x552_DPI150_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Peanut_640x480_DPI120_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Peanut_640x480_DPI120_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Peanut_640x480_DPI120_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Peanut_736x552_DPI150_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Peanut_736x552_DPI150_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Peanut_736x552_DPI150_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Sesame_640x480_DPI120_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Sesame_640x480_DPI120_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Sesame_640x480_DPI120_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Sesame_736x552_DPI150_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Sesame_736x552_DPI150_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Sesame_736x552_DPI150_test.json \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Sunflower_640x480_DPI120_train.json \
  --json /root/autodl-tmp/PRMI/val/labels_image_gt/Sunflower_640x480_DPI120_val.json \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Sunflower_640x480_DPI120_test.json \
  --out logs/stats_all_splits.json
```

## 3) Training (single-frame warmup)

> This trains prompt maps + topology/boundary/rootness on single frames (no memory), which stabilizes subsequent TL training.

```bash
cd /root/autodl-tmp/autorootsam
export PYTHONPATH=/root/autodl-tmp/autorootsam:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python train_single.py \
  --root_dir /root/autodl-tmp/PRMI \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Cotton_736x552_DPI150_train.json \
  --backbone sam2 \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_image_size 1024 \
  --tile 1024 --batch 4 --epochs 10 --num_workers 8 --amp \
  --out_dir runs_single/cotton150
```

## 4) Training (time-lapse adjacent-pair, Δt-aware memory)

```bash
cd /root/autodl-tmp/autorootsam
export PYTHONPATH=/root/autodl-tmp/autorootsam:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python train_timelapse.py \
  --root_dir /root/autodl-tmp/PRMI \
  --json /root/autodl-tmp/PRMI/train/labels_image_gt/Cotton_736x552_DPI150_train.json \
  --backbone sam2 \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_image_size 1024 \
  --tile 1024 --batch 2 --epochs 30 --num_workers 8 --amp \
  --tau_days 14.0 \
  --out_dir runs_tl/cotton150
```

## 5) Inference (native-resolution tiled, single image)

```bash
cd /root/autodl-tmp/autorootsam
export PYTHONPATH=/root/autodl-tmp/autorootsam:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python infer_tiled.py \
  --ckpt runs_tl/cotton150/checkpoints/best.pt \
  --image /root/autodl-tmp/PRMI/test/images/Cotton_736x552_DPI150/Cotton_T004_L016_2012.06.22_091757_AMC_DPI150.jpg \
  --out outputs/cotton150_single_mask.png \
  --tile 1024 --overlap 128
```

## 6) Inference (sequence, Δt-aware rolling memory; reads test JSON)

```bash
cd /root/autodl-tmp/autorootsam
export PYTHONPATH=/root/autodl-tmp/autorootsam:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python infer_sequence.py \
  --ckpt runs_tl/cotton150/checkpoints/best.pt \
  --json /root/autodl-tmp/PRMI/test/labels_image_gt/Cotton_736x552_DPI150_test.json \
  --root_dir /root/autodl-tmp/PRMI \
  --out_dir outputs/cotton150_seq \
  --tile 1024 --overlap 128
```

## Notes / Common pitfalls
- If you see a `ModuleNotFoundError: mdrs`, ensure:
  `export PYTHONPATH=/root/autodl-tmp/autorootsam:$PYTHONPATH`
- `--tile` should match `--sam2_image_size` when using the official SAM2 adapter (1024 recommended).
- If your GPU is small, reduce `--batch`, or use `--tile 768` with `--sam2_image_size 1024` (the code pads tiles to the required size).
