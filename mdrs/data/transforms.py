import random
from typing import Dict

import torch

from ..utils.tiling import Tile, anchor_token_params


def _sample_root_guided_crop(mask: torch.Tensor, tile: int) -> tuple[int, int] | None:
    """Return (y0,x0) for a crop window guided by positive pixels.

    Args:
        mask: (1,H,W) float/bool tensor
        tile: tile size

    Returns:
        (y0,x0) or None if no positive pixels.
    """
    # Use a cheap check first
    if mask is None:
        return None
    # mask could be float in {0,1}
    pos = torch.nonzero(mask[0] > 0.5, as_tuple=False)
    if pos.numel() == 0:
        return None
    # Randomly pick a positive pixel
    iy = random.randint(0, pos.shape[0] - 1)
    cy, cx = int(pos[iy, 0].item()), int(pos[iy, 1].item())
    H, W = mask.shape[-2:]
    th = tw = tile
    y0 = max(0, min(H - th, cy - th // 2))
    x0 = max(0, min(W - tw, cx - tw // 2))
    return y0, x0


class EnsureAnchor:
    """If no anchor is present, create a full-image anchor."""

    def __call__(self, sample: Dict):
        if "anchor" not in sample or sample["anchor"] is None:
            img = sample["image"]
            _H, _W = img.shape[-2:]
            sample["anchor"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        return sample


class RandomTileCrop:
    """Random crop for a single frame; also produces normalized anchor params.

    PRMI is extremely imbalanced (lots of background). Pure random tiles often yield
    empty crops and slow/unstable training.

    This transform therefore supports optional root-guided cropping:
      - If the tile contains any positive pixel in GT mask, with probability `pos_prob`
        choose a crop centered near a random positive pixel.
    """

    def __init__(self, tile: int = 768, pos_prob: float = 0.7):
        self.tile = tile
        self.pos_prob = float(pos_prob)

    def __call__(self, sample: Dict):
        img = sample["image"]  # (3,H,W)
        mask = sample["mask"]  # (1,H,W)
        H, W = img.shape[-2:]

        # If smaller than tile: pad (no scaling)
        if H < self.tile or W < self.tile:
            pad_h = max(0, self.tile - H)
            pad_w = max(0, self.tile - W)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out = dict(sample)
            out["image"] = img
            out["mask"] = mask
            if "flux" in out and out["flux"] is not None:
                out["flux"] = torch.nn.functional.pad(out["flux"], (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out["anchor"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            return out

        th = tw = self.tile

        # Root-guided crop (only when mask has positives)
        yx = None
        if random.random() < self.pos_prob:
            yx = _sample_root_guided_crop(mask, self.tile)
        if yx is None:
            y0 = 0 if H == th else random.randint(0, H - th)
            x0 = 0 if W == tw else random.randint(0, W - tw)
        else:
            y0, x0 = yx

        out = dict(sample)
        out["image"] = img[:, y0 : y0 + th, x0 : x0 + tw]
        out["mask"] = mask[:, y0 : y0 + th, x0 : x0 + tw]
        if "flux" in out and out["flux"] is not None:
            out["flux"] = out["flux"][:, y0 : y0 + th, x0 : x0 + tw]

        t = Tile(y=y0, x=x0, h=th, w=tw)
        out["anchor"] = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32)
        return out


class PairRandomTileCropShared:
    """Random crop for a (prev, cur) pair, with the SAME crop window and SAME anchor.

    Also supports optional root-guided crop using the *current* GT mask.
    """

    def __init__(self, tile: int = 768, pos_prob: float = 0.7):
        self.tile = tile
        self.pos_prob = float(pos_prob)

    def __call__(self, sample: Dict):
        img0 = sample["image_prev"]  # (3,H,W)
        img1 = sample["image"]
        m0 = sample["mask_prev"]
        m1 = sample["mask"]
        H, W = img0.shape[-2:]

        # Pad if needed
        if H < self.tile or W < self.tile:
            pad_h = max(0, self.tile - H)
            pad_w = max(0, self.tile - W)
            out = dict(sample)
            out["image_prev"] = torch.nn.functional.pad(img0, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out["image"] = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out["mask_prev"] = torch.nn.functional.pad(m0, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out["mask"] = torch.nn.functional.pad(m1, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            if "flux_prev" in out and out["flux_prev"] is not None:
                out["flux_prev"] = torch.nn.functional.pad(out["flux_prev"], (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            if "flux" in out and out["flux"] is not None:
                out["flux"] = torch.nn.functional.pad(out["flux"], (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            out["anchor"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            return out

        th = tw = self.tile

        yx = None
        if random.random() < self.pos_prob:
            yx = _sample_root_guided_crop(m1, self.tile)
        if yx is None:
            y0 = 0 if H == th else random.randint(0, H - th)
            x0 = 0 if W == tw else random.randint(0, W - tw)
        else:
            y0, x0 = yx

        out = dict(sample)
        out["image_prev"] = img0[:, y0 : y0 + th, x0 : x0 + tw]
        out["image"] = img1[:, y0 : y0 + th, x0 : x0 + tw]
        out["mask_prev"] = m0[:, y0 : y0 + th, x0 : x0 + tw]
        out["mask"] = m1[:, y0 : y0 + th, x0 : x0 + tw]

        if "flux_prev" in out and out["flux_prev"] is not None:
            out["flux_prev"] = out["flux_prev"][:, y0 : y0 + th, x0 : x0 + tw]
        if "flux" in out and out["flux"] is not None:
            out["flux"] = out["flux"][:, y0 : y0 + th, x0 : x0 + tw]

        t = Tile(y=y0, x=x0, h=th, w=tw)
        out["anchor"] = torch.tensor([anchor_token_params(t, H, W)], dtype=torch.float32)
        return out
