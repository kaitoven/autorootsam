import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import TokenCrossAttention


class AnchorMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        # anchor: (B,4) normalized (x0/W,y0/H,w/W,h/H)
        return self.mlp(anchor)


class FluxEncoder(nn.Module):
    """Encode a 2ch tangent/flux field into a vector token."""

    def __init__(self, dim: int):
        super().__init__()
        h = max(8, dim // 2)
        self.net = nn.Sequential(nn.Conv2d(2, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, dim, 1), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(flux)).flatten(1)


class TimeGapEncoder(nn.Module):
    """Encode Δt (days) to a token; used to make memory gating time-lapse aware."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        # (B,1) float
        return self.net(delta_days)


class PromptMapGeneratorTL(nn.Module):
    """Hybrid Route-3+ prompter, upgraded for PRMI time-lapse (Δt-aware memory gating).

    Outputs differentiable prompt maps:
      - centerline heatmap
      - tip heatmap
      - bg_hard heatmap

    Key idea:
      memory update is weighted by exp(-Δt/τ), so long gaps rely less on memory.
    """

    def __init__(
        self,
        dim: int,
        stage1_dim: int,
        stage2_dim: int,
        stage4_dim: int,
        num_pos: int = 8,
        num_neg: int = 4,
        num_track: int = 4,
        num_unc: int = 2,
        num_heads: int = 8,
        tau_days: float = 14.0,
        max_tex_tokens: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.nq = num_pos + num_neg + num_track + num_unc
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_track = num_track
        self.num_unc = num_unc
        self.tau_days = float(tau_days)
        self.max_tex_tokens = int(max_tex_tokens)

        self.query_bank = nn.Parameter(torch.randn(self.nq, dim) * 0.02)
        self.anchor_mlp = AnchorMLP(dim)
        self.flux_enc = FluxEncoder(dim)
        self.time_enc = TimeGapEncoder(dim)

        self.mem_attn = TokenCrossAttention(dim, num_heads=num_heads)
        self.sem_attn = TokenCrossAttention(dim, num_heads=num_heads)
        self.tex_attn = TokenCrossAttention(dim, num_heads=num_heads)

        self.proj_s1 = nn.Conv2d(stage1_dim, dim, 1)
        self.proj_s2 = nn.Conv2d(stage2_dim, dim, 1)
        self.proj_s4 = nn.Conv2d(stage4_dim, dim, 1)

        # map heads at stage1 resolution
        self.film = nn.Linear(dim, dim * 2)
        self.map_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.center_head = nn.Conv2d(dim, 1, 1)
        self.tip_head = nn.Conv2d(dim, 1, 1)
        self.bg_head = nn.Conv2d(dim, 1, 1)

        self.mem_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Sigmoid())

    def _flatten(self, feat: torch.Tensor) -> torch.Tensor:
        return rearrange(feat, "b c h w -> b (h w) c")

    def _compress_feat(self, feat: torch.Tensor, max_tokens: int) -> torch.Tensor:
        B, C, H, W = feat.shape
        if H * W <= max_tokens:
            return feat
        side = int(max(1, round(math.sqrt(max_tokens))))
        return F.adaptive_avg_pool2d(feat, (side, side))

    def _time_decay(self, delta_days: Optional[torch.Tensor]) -> torch.Tensor:
        # returns (B,1,1) in [0,1]
        if delta_days is None:
            return None
        d = torch.clamp(delta_days, min=0.0)
        return torch.exp(-d / self.tau_days)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        memory_tokens: Optional[torch.Tensor] = None,
        flux_prev: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        delta_days: Optional[torch.Tensor] = None,
    ):
        s1, s2, s4 = feats["stage1"], feats["stage2"], feats["stage4"]
        B = s4.shape[0]

        q = self.query_bank.unsqueeze(0).repeat(B, 1, 1)

        # anchor
        anchor_token = torch.zeros((B, self.dim), device=s4.device, dtype=s4.dtype)
        if anchor is not None:
            a = self.anchor_mlp(anchor)
            anchor_token = a
            q = q + a.unsqueeze(1)

        # time embedding + decay
        time_token = torch.zeros((B, self.dim), device=s4.device, dtype=s4.dtype)
        decay = None
        if delta_days is not None:
            time_token = self.time_enc(delta_days)
            decay = self._time_decay(delta_days)  # (B,1)
            q = q + time_token.unsqueeze(1)

        # physics injection (tangent field) to track queries
        if flux_prev is not None:
            pe = self.flux_enc(flux_prev)
            if decay is not None:
                pe = pe * decay  # long gaps: weaker physics prior
            q_track = q[:, self.num_pos + self.num_neg : self.num_pos + self.num_neg + self.num_track, :] + pe.unsqueeze(1)
            q = torch.cat(
                [q[:, : self.num_pos + self.num_neg, :], q_track, q[:, self.num_pos + self.num_neg + self.num_track :, :]],
                dim=1,
            )

        # memory retrieval with Δt-aware gating
        if memory_tokens is not None:
            q_mem = self.mem_attn(q, memory_tokens)
            gate_in = q_mem + time_token.unsqueeze(1)
            g = self.mem_gate(gate_in)
            if decay is not None:
                g = g * decay.view(B, 1, 1)
            q = q + g * (q_mem - q)

        # semantics (stage4)
        kv_sem = self._flatten(self.proj_s4(s4))
        q = self.sem_attn(q, kv_sem)

        # texture refinement (stage1/2), downsample tokens for speed
        s1p = self.proj_s1(s1)
        s2p = self.proj_s2(s2)
        s1c = self._compress_feat(s1p, self.max_tex_tokens)
        s2c = self._compress_feat(s2p, self.max_tex_tokens)
        kv_tex = torch.cat([self._flatten(s1c), self._flatten(s2c)], dim=1)
        q = self.tex_attn(q, kv_tex)

        # FiLM conditioning from (pos + track + unc)
        q_sum = torch.mean(
            torch.cat(
                [
                    q[:, : self.num_pos, :],
                    q[:, self.num_pos + self.num_neg : self.num_pos + self.num_neg + self.num_track, :],
                    q[:, -self.num_unc :, :],
                ],
                dim=1,
            ),
            dim=1,
        )
        gamma_beta = self.film(q_sum)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(B, self.dim, 1, 1)
        beta = beta.view(B, self.dim, 1, 1)

        f = s1p * (1.0 + gamma) + beta
        f = self.map_head(f)

        center_logits = self.center_head(f)
        tip_logits = self.tip_head(f)
        bg_logits = self.bg_head(f)

        aux = {
            "q_all": q,
            "q_pos": q[:, : self.num_pos, :],
            "q_neg": q[:, self.num_pos : self.num_pos + self.num_neg, :],
            "anchor_token": anchor_token,
            "time_token": time_token,
        }
        if decay is not None:
            aux["time_decay"] = decay
        return {"center": center_logits, "tip": tip_logits, "bg_hard": bg_logits}, aux


class PromptMapEncoder(nn.Module):
    """Encode prompt maps into a dense embedding at stage4 resolution (SAM-like dense prompt)."""

    def __init__(self, prompt_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, prompt_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 1),
        )

    def forward(self, maps: Dict[str, torch.Tensor], size_hw):
        x = torch.cat([maps["center"], maps["tip"], maps["bg_hard"]], dim=1)
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.net(x)


def sample_sparse_coords_from_center(center_logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Top-k peak sampling from centerline prob. Returns coords (B,K,2) normalized to [-1,1]."""
    with torch.no_grad():
        prob = torch.sigmoid(center_logits)
        B, _, H, W = prob.shape
        flat = prob.view(B, -1)
        k = min(k, flat.shape[1])
        _, idx = torch.topk(flat, k=k, dim=1)
        ys = (idx // W).float()
        xs = (idx % W).float()
        xn = (xs / max(1, W - 1)) * 2 - 1
        yn = (ys / max(1, H - 1)) * 2 - 1
        return torch.stack([xn, yn], dim=-1)


def _sample_topk_coords_from_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k sampling from a single-channel logits map.

    Returns coords (B,K,2) normalized to [-1,1].
    """
    if k <= 0:
        return logits.new_zeros((logits.shape[0], 0, 2))
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        B, _, H, W = prob.shape
        flat = prob.view(B, -1)
        k = min(int(k), int(flat.shape[1]))
        _, idx = torch.topk(flat, k=k, dim=1)
        ys = (idx // W).float()
        xs = (idx % W).float()
        xn = (xs / max(1, W - 1)) * 2 - 1
        yn = (ys / max(1, H - 1)) * 2 - 1
        return torch.stack([xn, yn], dim=-1)


def sample_sparse_points_from_maps(
    maps: Dict[str, torch.Tensor],
    k_pos_center: int = 6,
    k_pos_tip: int = 2,
    k_neg_bg: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hybrid Route-3+ sparse points: positive (center/tip) + negative (bg_hard).

    Returns:
        coords_norm: (B,K,2) in [-1,1]
        labels: (B,K) int32, 1=positive, 0=negative
    """
    c = _sample_topk_coords_from_logits(maps["center"], k_pos_center)
    t = _sample_topk_coords_from_logits(maps["tip"], k_pos_tip)
    b = _sample_topk_coords_from_logits(maps["bg_hard"], k_neg_bg)

    coords = torch.cat([c, t, b], dim=1)
    B = coords.shape[0]
    labels = coords.new_empty((B, coords.shape[1]), dtype=torch.int32)
    labels[:, : c.shape[1] + t.shape[1]] = 1
    labels[:, c.shape[1] + t.shape[1] :] = 0
    return coords, labels


def sample_sparse_tokens_from_maps(maps: Dict[str, torch.Tensor], k: int = 8) -> torch.Tensor:
    """Backward-compatible helper.

    The TL model samples top-k peaks from the *centerline* prompt map to
    provide optional sparse prompts for a SAM-style decoder.

    Returns coords (B,K,2) in [-1,1].
    """
    return sample_sparse_coords_from_center(maps["center"], k=k)
