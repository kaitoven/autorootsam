import torch
import torch.nn as nn
import torch.nn.functional as F


class RootnessHead(nn.Module):
    """Predict if the tile/frame contains any root (PRMI has_root weak label)."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class BoundaryHead(nn.Module):
    """Predict boundary/edge probability map for mask sharpening."""

    def __init__(self, in_dim: int, mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, feat: torch.Tensor, out_hw=None) -> torch.Tensor:
        logits = self.net(feat)
        if out_hw is not None:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits
