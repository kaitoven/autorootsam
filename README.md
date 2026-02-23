# Root-STC-SAM 2.1++-TL (Verson 01 start)

**TL = Time-lapse aware**, designed for PRMI minirhizotron data where frames are sampled every **~10–30+ days**.

## What this repo implements
- Native-resolution tiling + **Spatial Anchor** (coordinate-aware across tiles)
- Frozen SAM2.1 Hiera backbone + LoRA(Q/V) (adapter hook)
- **LWA++**: Wavelet high-frequency + oriented-line enhancement + sparse gate (Stage1&2)
- **PromptMapGenerator** (Hybrid Route-3+): differentiable prompt maps
  - centerline heatmap
  - tip heatmap
  - bg-hard heatmap (pseudo on has_root=0)
- PromptMapEncoder: prompt maps -> dense prompt embedding (SAM-like)
- SAM2.1 mask decoder (external) takes dense prompt (+ optional sparse peaks)
- Topology head (skeleton + tangent/flux) + GFU bridge
- Boundary head refinement (robust to PRMI rectangle-derived masks)
- Rootness head uses PRMI `has_root` weak labels
- **Time-lapse Memory-Gating**: memory usage is **decayed by Δt(days)**
- MemoryWriter++ writes (mask, flux, anchor) back to a memory token

> A ToyBackbone is included to run the pipeline without SAM2.1.
> For real results, implement `Sam21BackboneAdapter` for `facebookresearch/sam2`.

## Install
```bash
pip install -r requirements.txt
```

## Download SAM2.1 checkpoint
```bash
python download_ckpt.py \
  --url https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  --out checkpoints/sam2.1_hiera_large.pt
```

## Train (single-frame)
```bash
python train_single.py --root_dir /path/PRMI --json /path/Cotton_736x552_DPI150_train.json --tile 768 --epochs 30
```

During training, the script runs validation each epoch (if `--val_json` is provided or can be auto-guessed)
and saves:
- `out_dir/checkpoints/best.pt` (best by **val mean Dice**)
- `out_dir/checkpoints/epoch_XXX.pt` (latest snapshots, optionally pruned)

## Train (time-lapse pairs; Δt-aware memory)
```bash
python train_timelapse.py --root_dir /path/PRMI --json /path/Cotton_736x552_DPI150_train.json --tile 768 --epochs 30
```

Likewise saves `checkpoints/best.pt` selected by **val mean Dice**.

### Using the official SAM2.1 backbone
1) Install SAM2 repo:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
2) Train with `--backbone sam2` (recommend `--tile 1024` with padding):
```bash
python train_timelapse.py \
  --backbone sam2 \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --tile 1024
```

## Native-resolution tiled inference (single image)
```bash
python infer_tiled.py --ckpt runs/epoch_029.pt --image img.jpg --out mask.png --tile 768 --overlap 128
```

## Time-lapse sequence inference (uses Δt + tile-level memory tokens)
```bash
python infer_sequence.py \
  --ckpt runs_tl/epoch_029.pt \
  --json /path/Cotton_736x552_DPI150_test.json \
  --root_dir /path/PRMI \
  --out_dir outputs \
  --tile 768 --overlap 128
```

## Integrate with SAM2.1
Implement `mdrs.models.autorootsam.Sam21BackboneAdapter`:
- `encode_image(x)` -> dict with `stage1, stage2, stage4`
- `decode_masks(feats, dense_prompt, sparse_coords=None, sparse_labels=None, memory_tokens=None)` -> mask_logits at input resolution

Template: `sam2_adapter_template.py`

We also provide a ready-to-use official adapter:
`mdrs/backbones/sam2_official_adapter.py` (expects tiles padded/resized to SAM2 `image_size`).

## Dataset statistics (train/val/test json)
Compute time-lapse sequence statistics (seq length, Δt distribution, root/non-root ratios):

```bash
python tools/prmi_stats.py --json_dir /path/PRMI/train/labels_image_gt
python tools/prmi_stats.py --json /path/Cotton_736x552_DPI150_train.json --json /path/Cotton_736x552_DPI150_val.json --json /path/Cotton_736x552_DPI150_test.json
```

## Export all source code into one text file

```bash
python tools/print_repo.py --root . --out full_source.txt
```
