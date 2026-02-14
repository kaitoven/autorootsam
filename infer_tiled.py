import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from mdrs.models.autorootsam import RootSTCSAM21PPTL
from mdrs.utils.tiling import sliding_tiles, Tile, anchor_token_params
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
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()
    return x


def cosine_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h) if h > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(w) if w > 1 else np.ones((1,), dtype=np.float32)
    win = np.outer(wy, wx).astype(np.float32)
    # avoid zeros at borders (especially if overlap is small)
    win = np.clip(win, 0.05, 1.0)
    return win


def pad_to(x: torch.Tensor, th: int, tw: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad CHW tensor to (th,tw). Returns padded and original (h,w)."""
    c, h, w = x.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, (h, w)


def main():
    ap = argparse.ArgumentParser(description="Tiled inference for Root-STC-SAM2.1++-TL")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.5)
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

    if args.backbone == "sam2" and (args.stage1_dim, args.stage2_dim, args.stage4_dim) == (64, 96, 128):
        args.stage1_dim = 256
        args.stage2_dim = 256
        args.stage4_dim = 256
        print("[info] backbone=sam2 -> auto set stage dims to 256 (override defaults).")

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

    img = read_image_rgb(args.image)
    H, W = img.shape[:2]

    tiles = sliding_tiles(H, W, tile=args.tile, overlap=args.overlap)
    model_input_size = args.sam2_image_size if args.backbone == "sam2" else args.tile

    logits_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for t in tiles:
        patch = img[t.y : t.y + t.h, t.x : t.x + t.w]
        x = to_tensor(patch)
        x, (oh, ow) = pad_to(x, model_input_size, model_input_size)

        anchor = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32)

        with torch.no_grad():
            pred = model(
                x.unsqueeze(0).to(device),
                anchor=anchor.to(device),
                delta_t=torch.zeros((1, 1), device=device),
                memory_tokens=None,
                flux_prev=None,
            )
            logit = pred["mask_logits"][0, 0].detach().cpu().numpy()[:oh, :ow]

        wmap = cosine_window(oh, ow)
        logits_sum[t.y : t.y + oh, t.x : t.x + ow] += logit * wmap
        weight_sum[t.y : t.y + oh, t.x : t.x + ow] += wmap

    logits_avg = logits_sum / (weight_sum + 1e-6)
    prob = 1.0 / (1.0 + np.exp(-logits_avg))
    mask = (prob >= args.thr).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, mask)
    print(f"Saved mask: {args.out}")


if __name__ == "__main__":
    main()
