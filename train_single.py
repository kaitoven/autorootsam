import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mdrs.data.prmi_dataset import PRMIDatasetSingle
from mdrs.data.transforms import EnsureAnchor, RandomTileCrop
from mdrs.losses_tl import CompoundLossTL
from mdrs.models.autorootsam import RootSTCSAM21PPTL
from mdrs.utils.seed import set_seed
from mdrs.utils.tiling import Tile, anchor_token_params, sliding_tiles
from mdrs.utils.lora import apply_lora


def collate_keep_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "meta":
            out[k] = [b.get(k) for b in batch]
            continue
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b.get(k) for b in batch]
    return out


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
        bb = SAM2OfficialBackboneAdapter(cfg, device=device)

        # Optional LoRA on SAM2 attention projections (best-effort)
        if args.lora_r > 0:
            hit = apply_lora(bb.sam2, keywords=("q", "v"), r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            print(f"[LoRA] injected into {hit} Linear layers (keywords=q/v).")
        return bb

    raise ValueError(f"Unknown backbone: {args.backbone}")


def pad_to(x: torch.Tensor, th: int, tw: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    c, h, w = x.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, (h, w)


@torch.no_grad()
def infer_logits_full(
    model: RootSTCSAM21PPTL,
    img: torch.Tensor,
    *,
    tile: int,
    overlap: int,
    model_input_size: int,
    device: str,
) -> torch.Tensor:
    """Infer full-resolution mask logits for a single image tensor (3,H,W)."""
    img = img.to(device)
    H, W = img.shape[-2:]

    # Fast path: single pass when image fits in one crop
    if H <= tile and W <= tile:
        x, (oh, ow) = pad_to(img, model_input_size, model_input_size)
        anchor = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        pred = model(
            x.unsqueeze(0),
            anchor=anchor,
            delta_t=torch.zeros((1, 1), device=device),
            memory_tokens=None,
            flux_prev=None,
            return_aux=False,
        )
        logits = pred["mask_logits"][0, 0][:oh, :ow]
        return logits[:H, :W].detach().cpu()

    # General path: tiled inference
    tiles = sliding_tiles(H, W, tile=tile, overlap=overlap)
    logits_sum = torch.zeros((H, W), dtype=torch.float32)
    weight_sum = torch.zeros((H, W), dtype=torch.float32)

    def cosine_window(h: int, w: int) -> torch.Tensor:
        wy = torch.hann_window(h) if h > 1 else torch.ones((1,))
        wx = torch.hann_window(w) if w > 1 else torch.ones((1,))
        win = torch.outer(wy, wx)
        win = torch.clamp(win, 0.05, 1.0)
        return win

    for t in tiles:
        patch = img[:, t.y : t.y + t.h, t.x : t.x + t.w]
        x, (oh, ow) = pad_to(patch, model_input_size, model_input_size)
        anchor = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32, device=device)
        pred = model(
            x.unsqueeze(0),
            anchor=anchor,
            delta_t=torch.zeros((1, 1), device=device),
            memory_tokens=None,
            flux_prev=None,
            return_aux=False,
        )
        logit = pred["mask_logits"][0, 0][:oh, :ow].detach().cpu()
        wmap = cosine_window(oh, ow)
        logits_sum[t.y : t.y + oh, t.x : t.x + ow] += logit * wmap
        weight_sum[t.y : t.y + oh, t.x : t.x + ow] += wmap

    return logits_sum / (weight_sum + 1e-6)


def dice_iou_from_logits(logits: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> Tuple[float, float]:
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).to(torch.float32)
    gt = (gt > 0.5).to(torch.float32)
    inter = torch.sum(pred * gt).item()
    union = torch.sum((pred + gt) > 0.5).item()
    p = torch.sum(pred).item()
    g = torch.sum(gt).item()
    dice = (2.0 * inter + eps) / (p + g + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def guess_split_json(train_json: str, target_split: str) -> str | None:
    """Best-effort guess of val/test json path from train json path."""
    base = os.path.basename(train_json)
    cand = base.replace("_train.json", f"_{target_split}.json")
    # 1) same folder
    p1 = os.path.join(os.path.dirname(train_json), cand)
    if os.path.exists(p1):
        return p1
    # 2) replace /train/ with /val/ etc
    p2 = train_json.replace(os.sep + "train" + os.sep, os.sep + target_split + os.sep)
    p2 = os.path.join(os.path.dirname(p2), cand)
    if os.path.exists(p2):
        return p2
    p3 = train_json.replace("_train.json", f"_{target_split}.json")
    if os.path.exists(p3):
        return p3
    return None


def save_ckpt(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int, best_metric: float, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser(description="Single-frame training (Root-STC-SAM2.1++-TL)")
    ap.add_argument("--root_dir", required=True, help="PRMI root dir (contains train/val/test)")
    ap.add_argument("--json", required=True, help="train json path")
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--test_json", default=None)

    ap.add_argument("--out_dir", default="runs_single")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--pos_prob", type=float, default=0.7, help="probability of root-guided crop in training")

    ap.add_argument("--backbone", choices=["toy", "sam2"], default="toy")
    ap.add_argument("--stage1_dim", type=int, default=64)
    ap.add_argument("--stage2_dim", type=int, default=96)
    ap.add_argument("--stage4_dim", type=int, default=128)
    ap.add_argument("--prompt_dim", type=int, default=256)

    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2_image_size", type=int, default=1024)
    ap.add_argument("--sam2_no_normalize", action="store_true")

    ap.add_argument("--lora_r", type=int, default=0)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--keep_epochs", type=int, default=3)

    args = ap.parse_args()

    # For SAM2, the model expects fixed square input (1024 by default).
    if args.backbone == "sam2" and args.tile != args.sam2_image_size:
        print(f"[warn] backbone=sam2 expects tile==sam2_image_size. Forcing tile={args.sam2_image_size}.")
        args.tile = args.sam2_image_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    val_json = args.val_json or guess_split_json(args.json, "val")
    test_json = args.test_json or guess_split_json(args.json, "test")
    if val_json is None:
        print("[warn] val_json not found; will train without validation.")

    backbone = build_backbone(args, device)

    # If using sam2 backbone, stage dims must be 256
    if args.backbone == "sam2" and (args.stage1_dim, args.stage2_dim, args.stage4_dim) == (64, 96, 128):
        args.stage1_dim = 256
        args.stage2_dim = 256
        args.stage4_dim = 256
        print("[info] backbone=sam2 -> auto set stage dims to 256 (override defaults).")

    model = RootSTCSAM21PPTL(
        backbone=backbone,
        stage1_dim=args.stage1_dim,
        stage2_dim=args.stage2_dim,
        stage4_dim=args.stage4_dim,
        prompt_dim=args.prompt_dim,
    ).to(device)

    loss_fn = CompoundLossTL()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_tf = lambda s: EnsureAnchor()(RandomTileCrop(tile=args.tile, pos_prob=args.pos_prob)(s))
    train_ds = PRMIDatasetSingle(args.root_dir, args.json, transform=train_tf, return_flux=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_keep_meta,
    )

    val_ds = PRMIDatasetSingle(args.root_dir, val_json, transform=None, return_flux=True) if val_json else None

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["image"].to(device)
            m = batch["mask"].to(device)
            has_root = batch["has_root"].to(device)
            flux = batch.get("flux")
            if flux is not None:
                flux = flux.to(device)
            anchor = batch.get("anchor")
            if anchor is not None:
                anchor = anchor.to(device)

            dt = torch.zeros((x.shape[0], 1), device=device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x, anchor=anchor, delta_t=dt, memory_tokens=None, flux_prev=None)
                pred["input_image"] = x  # for pseudo bg_hard
                loss, logs = loss_fn(pred, m, has_root, gt_flux=flux, prev_pred=None, delta_t_days=None)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({"loss": float(logs["total"].item())})

        # Validation
        if val_ds is not None and epoch % max(1, args.save_every) == 0:
            model.eval()
            dices: List[float] = []
            ious: List[float] = []
            # full-image eval (PRMI images are small enough)
            for i in tqdm(range(len(val_ds)), desc="val", leave=False):
                s = val_ds[i]
                img = s["image"]
                gt = s["mask"][0]
                logits = infer_logits_full(
                    model,
                    img,
                    tile=args.tile,
                    overlap=args.overlap,
                    model_input_size=args.sam2_image_size if args.backbone == "sam2" else args.tile,
                    device=device,
                )
                d, j = dice_iou_from_logits(logits, gt, thr=0.5)
                dices.append(d)
                ious.append(j)

            mean_dice = float(np.mean(dices)) if dices else 0.0
            mean_iou = float(np.mean(ious)) if ious else 0.0
            print(f"[val] epoch={epoch} dice={mean_dice:.4f} iou={mean_iou:.4f}")

            # Save best and periodic
            epoch_dir = os.path.join(args.out_dir, "checkpoints")
            if epoch % args.save_every == 0:
                save_ckpt(os.path.join(epoch_dir, f"epoch_{epoch:03d}.pt"), model, opt, epoch, best_dice, args)

            if mean_dice > best_dice:
                best_dice = mean_dice
                save_ckpt(os.path.join(epoch_dir, "best.pt"), model, opt, epoch, best_dice, args)
                print(f"[best] updated best dice -> {best_dice:.4f}")

            # cleanup old epochs
            if args.keep_epochs > 0:
                kept = set(range(max(1, epoch - args.keep_epochs + 1), epoch + 1))
                for fn in os.listdir(epoch_dir):
                    if fn.startswith("epoch_") and fn.endswith(".pt"):
                        e = int(fn.split("_")[1].split(".")[0])
                        if e not in kept:
                            try:
                                os.remove(os.path.join(epoch_dir, fn))
                            except Exception:
                                pass

    # Optional test with best checkpoint
    if test_json and os.path.exists(os.path.join(args.out_dir, "checkpoints", "best.pt")):
        print(f"[test] evaluating best.pt on {test_json}")
        ckpt = torch.load(os.path.join(args.out_dir, "checkpoints", "best.pt"), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        test_ds = PRMIDatasetSingle(args.root_dir, test_json, transform=None, return_flux=False)
        dices = []
        for i in tqdm(range(len(test_ds)), desc="test", leave=False):
            s = test_ds[i]
            logits = infer_logits_full(
                model,
                s["image"],
                tile=args.tile,
                overlap=args.overlap,
                model_input_size=args.sam2_image_size if args.backbone == "sam2" else args.tile,
                device=device,
            )
            d, _ = dice_iou_from_logits(logits, s["mask"][0], thr=0.5)
            dices.append(d)
        print(f"[test] mean dice = {float(np.mean(dices)) if dices else 0.0:.4f}")


if __name__ == "__main__":
    main()
