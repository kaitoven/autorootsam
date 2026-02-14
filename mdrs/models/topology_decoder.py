import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyAuxHead(nn.Module):
    """Predict skeleton logits + tangent/flux field from stage1 and mask logits."""

    def __init__(self, stage1_dim: int, mid: int = 128):
        super().__init__()
        self.stage1_proj = nn.Sequential(nn.Conv2d(stage1_dim, mid, 1), nn.GELU())
        self.net = nn.Sequential(
            nn.Conv2d(mid + 1, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GELU(),
        )
        self.skel_head = nn.Conv2d(mid, 1, 1)
        self.flux_head = nn.Conv2d(mid, 2, 1)

    def forward(self, stage1: torch.Tensor, mask_logits: torch.Tensor):
        B, _, H1, W1 = stage1.shape
        mask_up = F.interpolate(mask_logits, size=(H1, W1), mode="bilinear", align_corners=False)
        x = torch.cat([self.stage1_proj(stage1), mask_up], dim=1)
        x = self.net(x)
        return self.skel_head(x), self.flux_head(x)


class GatedFusionUnit(nn.Module):
    """Fuse skeleton prior into mask logits to bridge thin fractures."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, mask_logits: torch.Tensor, skel_prob: torch.Tensor):
        mask_prob = torch.sigmoid(mask_logits)
        gap = torch.clamp(skel_prob - mask_prob, min=0.0)
        g = self.gate(torch.cat([mask_prob, skel_prob], dim=1))
        return mask_logits + self.alpha * g * gap


class MemoryWriterPlusPlusTL(nn.Module):
    """Write (stage4, mask, flux, anchor, dt) into a memory token.

    This is **NOT** an optical-flow memory; it is a time-lapse prior token.
    """

    def __init__(self, stage4_dim: int, token_dim: int):
        super().__init__()
        self.dt_fc = nn.Sequential(nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.enc = nn.Sequential(
            nn.Conv2d(stage4_dim + 1 + 2 + token_dim + token_dim, token_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        stage4: torch.Tensor,
        mask_logits: torch.Tensor,
        flux: torch.Tensor,
        anchor_token: torch.Tensor,
        delta_t_days: torch.Tensor | None = None,
    ):
        B, _, h, w = stage4.shape
        mask = torch.sigmoid(F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False))
        flux = F.interpolate(flux, size=(h, w), mode="bilinear", align_corners=False)
        a = anchor_token.view(B, -1, 1, 1).expand(B, anchor_token.shape[1], h, w)
        if delta_t_days is None:
            dt = torch.zeros((B, 1), device=stage4.device, dtype=stage4.dtype)
        else:
            dt = delta_t_days.view(B, 1).to(stage4)
        dt_tok = self.dt_fc(dt).view(B, -1, 1, 1).expand(B, -1, h, w)
        x = torch.cat([stage4, mask, flux, a, dt_tok], dim=1)
        mem = self.pool(self.enc(x)).flatten(1)
        return mem.unsqueeze(1)


# Backward-compatible alias
MemoryWriterPlusPlus = MemoryWriterPlusPlusTL
