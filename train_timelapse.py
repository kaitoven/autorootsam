import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mdrs.data.prmi_dataset import PRMIPairDataset, PRMIDatasetSingle
from mdrs.data.transforms import EnsureAnchor, PairRandomTileCropShared, RandomTileCrop
from mdrs.losses_tl import CompoundLossTL
from mdrs.models.autorootsam import RootSTCSAM21PPTL
from mdrs.utils.seed import set_seed
from mdrs.utils.tiling import anchor_token_params, sliding_tiles
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
    memory_tokens: torch.Tensor | None = None,
    flux_prev: torch.Tensor | None = None,
    delta_t: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer full-resolution logits for a single image tensor (3,H,W).

    Returns:
        logits (H,W), memory_token (1,N,D)
    """
    img = img.to(device)
    H, W = img.shape[-2:]

    if H <= tile and W <= tile:
        x, (oh, ow) = pad_to(img, model_input_size, model_input_size)
        anchor = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        pred = model(
            x.unsqueeze(0),
            anchor=anchor,
            delta_t=delta_t,
            memory_tokens=memory_tokens,
            flux_prev=flux_prev,
            return_aux=False,
        )
        logits = pred["mask_logits"][0, 0][:oh, :ow]
        return logits[:H, :W].detach().cpu(), pred["memory_token"].detach().cpu()

    tiles = sliding_tiles(H, W, tile=tile, overlap=overlap)
    logits_sum = torch.zeros((H, W), dtype=torch.float32)
    weight_sum = torch.zeros((H, W), dtype=torch.float32)

    def cosine_window(h: int, w: int) -> torch.Tensor:
        wy = torch.hann_window(h) if h > 1 else torch.ones((1,))
        wx = torch.hann_window(w) if w > 1 else torch.ones((1,))
        win = torch.outer(wy, wx)
        win = torch.clamp(win, 0.05, 1.0)
        return win

    # For tiled full-image eval, we do not propagate memory across tiles in this helper.
    last_mem = None
    for t in tiles:
        patch = img[:, t.y : t.y + t.h, t.x : t.x + t.w]
        x, (oh, ow) = pad_to(patch, model_input_size, model_input_size)
        anchor = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32, device=device)
        pred = model(
            x.unsqueeze(0),
            anchor=anchor,
            delta_t=delta_t,
            memory_tokens=memory_tokens,
            flux_prev=flux_prev,
            return_aux=False,
        )
        last_mem = pred["memory_token"].detach().cpu()
        logit = pred["mask_logits"][0, 0][:oh, :ow].detach().cpu()
        wmap = cosine_window(oh, ow)
        logits_sum[t.y : t.y + oh, t.x : t.x + ow] += logit * wmap
        weight_sum[t.y : t.y + oh, t.x : t.x + ow] += wmap

    return logits_sum / (weight_sum + 1e-6), (last_mem if last_mem is not None else torch.zeros((1, 1, model.prompt_dim)))


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
    base = os.path.basename(train_json)
    cand = base.replace("_train.json", f"_{target_split}.json")
    p1 = os.path.join(os.path.dirname(train_json), cand)
    if os.path.exists(p1):
        return p1
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
    ap = argparse.ArgumentParser(description="Train Root-STC-SAM2.1++-TL (time-lapse pairs)")
    ap.add_argument("--root_dir", required=True, help="PRMI root dir (contains train/val/test)")
    ap.add_argument("--json", required=True, help="train json path")
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--test_json", default=None)

    ap.add_argument("--out_dir", default="runs_tl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
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

    ap.add_argument("--val_mode", choices=["pair", "single"], default="pair", help="how to validate")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--keep_epochs", type=int, default=3)

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

    # Guess val/test json if not given
    val_json = args.val_json or guess_split_json(args.json, "val")
    test_json = args.test_json or guess_split_json(args.json, "test")
    if val_json:
        print(f"[val] using {val_json}")
    else:
        print("[val] no val_json found; validation will be skipped.")

    backbone = build_backbone(args, device)
    model = RootSTCSAM21PPTL(
        backbone=backbone,
        stage1_dim=args.stage1_dim,
        stage2_dim=args.stage2_dim,
        stage4_dim=args.stage4_dim,
        prompt_dim=args.prompt_dim,
    ).to(device)

    train_tf = lambda s: EnsureAnchor()(PairRandomTileCropShared(tile=args.tile, pos_prob=args.pos_prob)(s))
    train_ds = PRMIPairDataset(args.root_dir, args.json, pair_transform=train_tf, return_flux=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_keep_meta,
    )

    # Validation datasets
    if val_json and args.val_mode == "pair":
        # Validate on full images (no random cropping) while still using prev->cur memory.
        # We do not need anchors from dataset here because inference builds anchors per tile.
        val_ds_pair = PRMIPairDataset(args.root_dir, val_json, pair_transform=None, return_flux=False)
    else:
        val_ds_pair = None

    val_ds_single = None
    if val_json and args.val_mode == "single":
        val_ds_single = PRMIDatasetSingle(args.root_dir, val_json, transform=None, return_flux=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    loss_fn = CompoundLossTL()

    best_dice = -1.0
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
        losses = []

        for batch in pbar:
            x_prev = batch["image_prev"].to(device)
            m_prev = batch["mask_prev"].to(device)
            x = batch["image"].to(device)
            m = batch["mask"].to(device)
            dt = batch["delta_t"].to(device)
            anchor = batch.get("anchor")
            if anchor is not None:
                anchor = anchor.to(device)

            has_root = batch["has_root"].to(device)
            flux_prev = batch.get("flux_prev")
            if flux_prev is not None:
                flux_prev = flux_prev.to(device)
            flux = batch.get("flux")
            if flux is not None:
                flux = flux.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                prev_pred = model(
                    x_prev,
                    anchor=anchor,
                    memory_tokens=None,
                    flux_prev=None,
                    delta_t=torch.zeros_like(dt),
                    return_aux=True,
                )
                mem = prev_pred["memory_token"].detach()  # stop gradients through memory across time

                pred = model(
                    x,
                    anchor=anchor,
                    memory_tokens=mem,
                    flux_prev=prev_pred["flux"].detach(),
                    delta_t=dt,
                    return_aux=True,
                )

                loss, ld = loss_fn(
                    pred,
                    m,
                    has_root,
                    gt_flux=flux,
                    prev_pred=prev_pred,
                    delta_t_days=dt,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)), seg=float(ld.get("seg", 0.0)))

        # Validation
        val_dice = None
        if val_json:
            model.eval()
            dices = []

            if args.val_mode == "pair" and val_ds_pair is not None:
                # Pair-based validation (prev->cur)
                loader = DataLoader(val_ds_pair, batch_size=1, shuffle=False, num_workers=max(1, args.num_workers // 2), collate_fn=collate_keep_meta)
                for batch in tqdm(loader, desc="val(pair)", leave=False):
                    x_prev = batch["image_prev"].to(device)
                    x = batch["image"].to(device)
                    gt = batch["mask"].to(device)
                    dt = batch["delta_t"].to(device)
                    anchor = batch.get("anchor")
                    if anchor is not None:
                        anchor = anchor.to(device)

                    prev_pred = model(
                        x_prev,
                        anchor=anchor,
                        memory_tokens=None,
                        flux_prev=None,
                        delta_t=torch.zeros_like(dt),
                        return_aux=False,
                    )
                    mem = prev_pred["memory_token"].detach()
                    pred = model(
                        x,
                        anchor=anchor,
                        memory_tokens=mem,
                        flux_prev=prev_pred["flux"].detach(),
                        delta_t=dt,
                        return_aux=False,
                    )
                    d, _ = dice_iou_from_logits(pred["mask_logits"][0, 0].detach().cpu(), gt[0, 0].detach().cpu(), thr=0.5)
                    dices.append(d)

            else:
                # Single-frame full-image validation
                assert val_ds_single is not None
                model_input_size = args.sam2_image_size if args.backbone == "sam2" else args.tile
                for i in tqdm(range(len(val_ds_single)), desc="val(single)", leave=False):
                    s = val_ds_single[i]
                    logits, _ = infer_logits_full(
                        model,
                        s["image"],
                        tile=args.tile,
                        overlap=args.overlap,
                        model_input_size=model_input_size,
                        device=device,
                        memory_tokens=None,
                        flux_prev=None,
                        delta_t=torch.zeros((1, 1), device=device),
                    )
                    d, _ = dice_iou_from_logits(logits, s["mask"][0], thr=0.5)
                    dices.append(d)

            val_dice = float(np.mean(dices)) if dices else 0.0
            print(f"[val] epoch={epoch} mean dice={val_dice:.4f} (mode={args.val_mode})")

        # Save checkpoints
        if epoch % args.save_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"), model, opt, epoch, best_dice, args)

        if val_dice is not None and val_dice > best_dice:
            best_dice = val_dice
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, opt, epoch, best_dice, args)
            print(f"[best] updated: dice={best_dice:.4f} @ epoch={epoch}")

        save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, opt, epoch, best_dice, args)

        if args.keep_epochs > 0:
            # keep only last N epoch_*.pt files
            files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")])
            if len(files) > args.keep_epochs:
                for fn in files[: len(files) - args.keep_epochs]:
                    try:
                        os.remove(os.path.join(ckpt_dir, fn))
                    except Exception:
                        pass

    # Optional test with best checkpoint
    if test_json and os.path.exists(os.path.join(ckpt_dir, "best.pt")):
        print(f"[test] evaluating best.pt on {test_json}")
        ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        # For test we use single-frame full-image evaluation by default
        test_ds = PRMIDatasetSingle(args.root_dir, test_json, transform=None, return_flux=False)
        model_input_size = args.sam2_image_size if args.backbone == "sam2" else args.tile
        dices = []
        for i in tqdm(range(len(test_ds)), desc="test", leave=False):
            s = test_ds[i]
            logits, _ = infer_logits_full(
                model,
                s["image"],
                tile=args.tile,
                overlap=args.overlap,
                model_input_size=model_input_size,
                device=device,
                memory_tokens=None,
                flux_prev=None,
                delta_t=torch.zeros((1, 1), device=device),
            )
            d, _ = dice_iou_from_logits(logits, s["mask"][0], thr=0.5)
            dices.append(d)
        print(f"[test] mean dice = {float(np.mean(dices)) if dices else 0.0:.4f}")


if __name__ == "__main__":
    main()
