import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.skeleton import cldice_loss
from .utils.targets import (
    centerline_to_tips,
    mask_to_boundary,
    mask_to_centerline,
    pseudo_bg_hard_from_image,
)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    num = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3))
    den = torch.sum(probs + targets, dim=(1, 2, 3)) + eps
    return torch.mean(1.0 - num / den)


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def masked_flux_loss(pred_flux: torch.Tensor, gt_flux: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = mask.float()
    dot = torch.sum(pred_flux * gt_flux, dim=1, keepdim=True)
    pn = torch.sqrt(torch.sum(pred_flux**2, dim=1, keepdim=True) + eps)
    gn = torch.sqrt(torch.sum(gt_flux**2, dim=1, keepdim=True) + eps)
    cos = dot / (pn * gn + eps)
    cos_loss = (1.0 - cos) * m
    mse = torch.sum((pred_flux - gt_flux) ** 2, dim=1, keepdim=True) * m
    denom = torch.sum(m) + eps
    return (torch.sum(cos_loss) / denom) + (torch.sum(mse) / denom)


def cosine_token_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # a,b: (B,1,D) or (B,D)
    if a.ndim == 3:
        a = a[:, 0]
    if b.ndim == 3:
        b = b[:, 0]
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    cos = torch.sum(a * b, dim=1)
    return torch.mean(1.0 - cos)


class CompoundLossTL(nn.Module):
    """Final optimal loss for PRMI time-lapse.

    Keep strong structure priors; remove video-style warp.

    - seg: focal + dice
    - topo: soft-clDice
    - prompt_maps: centerline + tip + bg_hard (pseudo on has_root=0)
    - boundary
    - flux local (tangent-field) optional
    - gate sparsity
    - rootness BCE
    - mem: Δt-decayed token consistency (optional, only if prev_pred is provided)
    """

    def __init__(
        self,
        w_seg=1.0,
        w_topo=0.5,
        w_maps=0.5,
        w_bnd=0.2,
        w_flux=0.1,
        w_gate=1e-4,
        w_root=0.2,
        w_mem=0.05,
        focal_alpha=0.25,
        focal_gamma=2.0,
        cldice_iters=12,
        tau_days: float = 14.0,
    ):
        super().__init__()
        self.w_seg = w_seg
        self.w_topo = w_topo
        self.w_maps = w_maps
        self.w_bnd = w_bnd
        self.w_flux = w_flux
        self.w_gate = w_gate
        self.w_root = w_root
        self.w_mem = w_mem
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cldice_iters = cldice_iters
        self.tau_days = tau_days

    def forward(
        self,
        pred: Dict,
        gt_mask: torch.Tensor,
        has_root: torch.Tensor,
        gt_flux: Optional[torch.Tensor] = None,
        prev_pred: Optional[Dict] = None,
        delta_t_days: Optional[torch.Tensor] = None,
    ):
        # seg
        l_focal = sigmoid_focal_loss(pred["mask_logits"], gt_mask, alpha=self.focal_alpha, gamma=self.focal_gamma)
        l_dice = dice_loss_from_logits(pred["mask_logits"], gt_mask)
        l_seg = l_focal + l_dice

        # topology
        l_topo = cldice_loss(torch.sigmoid(pred["mask_logits"]), gt_mask.float(), iters=self.cldice_iters)

        # prompt maps targets
        center_t = mask_to_centerline(gt_mask.float(), iters=self.cldice_iters)
        tip_t = centerline_to_tips(center_t)
        bnd_t = mask_to_boundary(gt_mask.float())

        l_center = F.binary_cross_entropy_with_logits(pred["center_logits"], center_t)
        l_tip = F.binary_cross_entropy_with_logits(pred["tip_logits"], tip_t)

        # bg_hard pseudo only on has_root==0
        img = pred.get("input_image")
        pseudo = pseudo_bg_hard_from_image(img) if img is not None else torch.zeros_like(gt_mask)
        hr = (has_root.view(-1, 1, 1, 1) > 0.5).float()
        bg_t = (1 - hr) * pseudo
        l_bg = F.binary_cross_entropy_with_logits(pred["bg_hard_logits"], bg_t)
        l_maps = l_center + l_tip + 0.5 * l_bg

        # boundary
        l_bnd = F.binary_cross_entropy_with_logits(pred["boundary_logits"], bnd_t)

        # flux local
        l_flux = torch.tensor(0.0, device=gt_mask.device)
        if gt_flux is not None:
            gf = F.interpolate(gt_flux, size=pred["mask_logits"].shape[-2:], mode="bilinear", align_corners=False)
            pf = F.interpolate(pred["flux"], size=gf.shape[-2:], mode="bilinear", align_corners=False)
            l_flux = masked_flux_loss(pf, gf, gt_mask)

        # gate sparsity
        l_gate = pred["gate1"].abs().mean() + pred["gate2"].abs().mean()

        # rootness
        l_root = F.binary_cross_entropy_with_logits(pred["rootness_logit"], has_root)

        # time-decayed memory consistency
        l_mem = torch.tensor(0.0, device=gt_mask.device)
        if prev_pred is not None and self.w_mem > 0:
            dt = delta_t_days
            if dt is None:
                dt = torch.zeros((gt_mask.shape[0], 1), device=gt_mask.device)
            decay = torch.exp(-dt / max(1e-6, self.tau_days)).clamp(0.0, 1.0)
            mem_loss = cosine_token_loss(pred["memory_token"], prev_pred["memory_token"].detach())
            l_mem = (decay.mean()) * mem_loss

        total = (
            self.w_seg * l_seg
            + self.w_topo * l_topo
            + self.w_maps * l_maps
            + self.w_bnd * l_bnd
            + self.w_flux * l_flux
            + self.w_gate * l_gate
            + self.w_root * l_root
            + self.w_mem * l_mem
        )

        logs = {
            "total": total.detach(),
            "seg": l_seg.detach(),
            "topo": l_topo.detach(),
            "maps": l_maps.detach(),
            "bnd": l_bnd.detach(),
            "flux": l_flux.detach(),
            "gate": l_gate.detach(),
            "root": l_root.detach(),
            "mem": l_mem.detach(),
            "focal": l_focal.detach(),
            "dice": l_dice.detach(),
            "center": l_center.detach(),
            "tip": l_tip.detach(),
            "bg": l_bg.detach(),
        }
        return total, logs
