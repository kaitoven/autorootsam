import argparse
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mdrs.data.prmi_dataset import load_prmi_json
from mdrs.models.autorootsam import RootSTCSAM21PPTL
from mdrs.utils.timelapse import parse_timestamp_from_image_name
from mdrs.utils.tiling import anchor_token_params, sliding_tiles
from mdrs.utils.seed import set_seed


def build_backbone(args, device: str):
    if args.backbone == "toy":
        from mdrs.backbones.toy_backbone import ToyBackbone

        return ToyBackbone(stage1_dim=args.stage1_dim, stage2_dim=args.stage2_dim, stage4_dim=args.stage4_dim)

    if args.backbone == "sam2":
        from mdrs.backbones.sam2_official_adapter import Sam2AdapterConfig, SAM2OfficialBackboneAdapter

        cfg = Sam2AdapterConfig(
            model_cfg=args.sam2_cfg,
            checkpoint=args.sam2_ckpt,
            image_size=args.sam2_image_size,
            do_normalize=not args.sam2_no_normalize,
        )
        return SAM2OfficialBackboneAdapter(cfg, device=device)

    raise ValueError(f"Unknown backbone: {args.backbone}")


def read_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_mask_bin(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_rgb).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def pad_to(x: torch.Tensor, th: int, tw: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    c, h, w = x.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, (h, w)


def cosine_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(w) if w > 1 else np.ones((1,), dtype=np.float32)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 0.05, 1.0)


def infer_tiled_with_memory(
    model: RootSTCSAM21PPTL,
    img_rgb: np.ndarray,
    *,
    tile: int,
    overlap: int,
    model_input_size: int,
    device: str,
    memory_bank: Dict[Tuple[int, int, int, int], torch.Tensor],
    delta_t_days: float,
    rootness_thresh: Optional[float],
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int, int], torch.Tensor]]:
    """Run tiled inference for a single frame with a per-tile memory bank."""
    H, W = img_rgb.shape[:2]
    tiles = sliding_tiles(H, W, tile=tile, overlap=overlap)

    logits_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    new_bank: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    for t in tiles:
        key = (t.y, t.x, t.h, t.w)
        mem = memory_bank.get(key, None)
        patch = img_rgb[t.y : t.y + t.h, t.x : t.x + t.w]
        x = to_tensor(patch)
        x, (oh, ow) = pad_to(x, model_input_size, model_input_size)

        anchor = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32, device=device)
        dt = torch.tensor([[float(delta_t_days)]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = model(
                x.unsqueeze(0).to(device),
                anchor=anchor,
                delta_t=dt,
                memory_tokens=None if mem is None else mem.to(device),
                flux_prev=None,
                rootness_thresh=rootness_thresh,
                return_aux=False,
            )
            logit = pred["mask_logits"][0, 0].detach().cpu().numpy()[:oh, :ow]
            new_bank[key] = pred["memory_token"].detach().cpu()

        wmap = cosine_window(oh, ow)
        logits_sum[t.y : t.y + oh, t.x : t.x + ow] += logit * wmap
        weight_sum[t.y : t.y + oh, t.x : t.x + ow] += wmap

    return logits_sum / (weight_sum + 1e-6), new_bank


def dice_iou(prob: np.ndarray, gt: np.ndarray, thr: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    pred = (prob >= thr).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float(np.sum(pred * gt))
    p = float(np.sum(pred))
    g = float(np.sum(gt))
    dice = (2 * inter + eps) / (p + g + eps)
    union = float(np.sum((pred + gt) > 0))
    iou = (inter + eps) / (union + eps)
    return dice, iou


def main():
    ap = argparse.ArgumentParser(description="Sequence inference on PRMI JSON (time-lapse, Δt-aware memory)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", required=True, help="PRMI split json (e.g., *_test.json)")
    ap.add_argument("--root_dir", required=True, help="PRMI root dir")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--rootness_thresh", type=float, default=0.25)
    ap.add_argument("--max_sequences", type=int, default=0)
    ap.add_argument("--max_frames_per_seq", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--backbone", choices=["toy", "sam2"], default="toy")
    ap.add_argument("--stage1_dim", type=int, default=64)
    ap.add_argument("--stage2_dim", type=int, default=96)
    ap.add_argument("--stage4_dim", type=int, default=128)
    ap.add_argument("--prompt_dim", type=int, default=256)

    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2_image_size", type=int, default=1024)
    ap.add_argument("--sam2_no_normalize", action="store_true")

    args = ap.parse_args()

    if args.backbone == "sam2" and args.tile != args.sam2_image_size:
        print(f"[warn] backbone=sam2 expects tile==sam2_image_size. Forcing tile={args.sam2_image_size}.")
        args.tile = args.sam2_image_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    backbone = build_backbone(args, device)
    model = RootSTCSAM21PPTL(
        backbone=backbone,
        stage1_dim=args.stage1_dim,
        stage2_dim=args.stage2_dim,
        stage4_dim=args.stage4_dim,
        prompt_dim=args.prompt_dim,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Parse json and group by sequence_id
    records = load_prmi_json(args.json)
    by_seq = defaultdict(list)
    for r in records:
        ts = parse_timestamp_from_image_name(r.image_name, r.date)
        if ts is None:
            continue
        by_seq[r.seq_id].append((ts, r))

    seq_ids = sorted(by_seq.keys())
    if args.max_sequences and args.max_sequences > 0:
        seq_ids = seq_ids[: args.max_sequences]

    os.makedirs(args.out_dir, exist_ok=True)

    all_dice = []
    all_iou = []

    for sid in tqdm(seq_ids, desc="sequences"):
        items = sorted(by_seq[sid], key=lambda x: x[0])
        if args.max_frames_per_seq and args.max_frames_per_seq > 0:
            items = items[: args.max_frames_per_seq]

        # Infer split/subset from json file name (best-effort)
        # Resolve image/mask paths based on PRMI standard structure
        # We try both: <root>/<split>/images/<subset>/<image_name> and a direct join
        # by leveraging the fact that image_name doesn't contain subdirs.
        base_json = os.path.basename(args.json)
        subset = base_json.replace("_train.json", "").replace("_val.json", "").replace("_test.json", "")
        split = "train" if "_train.json" in base_json else "val" if "_val.json" in base_json else "test"

        seq_out = os.path.join(args.out_dir, sid.replace("|", "__"))
        os.makedirs(seq_out, exist_ok=True)

        memory_bank: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
        prev_ts = None

        for ts, r in items:
            dt_days = 0.0
            if prev_ts is not None:
                dt_days = (ts - prev_ts).total_seconds() / 86400.0
            prev_ts = ts

            img_path = os.path.join(args.root_dir, split, "images", subset, r.image_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(args.root_dir, r.image_name)

            m_path = os.path.join(args.root_dir, split, "masks_pixel_gt", subset, r.binary_mask)
            if not os.path.exists(m_path):
                m_path = os.path.join(args.root_dir, r.binary_mask)

            img = read_image_rgb(img_path)
            H, W = img.shape[:2]
            logits, memory_bank = infer_tiled_with_memory(
                model,
                img,
                tile=args.tile,
                overlap=args.overlap,
                model_input_size=args.sam2_image_size if args.backbone == "sam2" else args.tile,
                device=device,
                memory_bank=memory_bank,
                delta_t_days=dt_days,
                rootness_thresh=args.rootness_thresh,
            )
            prob = 1.0 / (1.0 + np.exp(-logits))
            pred_mask = (prob >= args.thr).astype(np.uint8) * 255

            out_name = os.path.splitext(r.image_name)[0] + "_pred.png"
            cv2.imwrite(os.path.join(seq_out, out_name), pred_mask)

            # Metric (if GT exists)
            if os.path.exists(m_path):
                gt = read_mask_bin(m_path, (H, W))
                d, j = dice_iou(prob, gt, thr=args.thr)
                all_dice.append(d)
                all_iou.append(j)

    if all_dice:
        print(f"[summary] mean dice={float(np.mean(all_dice)):.4f} | mean iou={float(np.mean(all_iou)):.4f} | N={len(all_dice)}")
    else:
        print("[summary] no GT masks found for metric computation.")


if __name__ == "__main__":
    main()
