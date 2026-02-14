import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.autorootsam import Sam21BackboneAdapter


class ToyBackbone(Sam21BackboneAdapter):
    """A minimal backbone to validate the end-to-end pipeline without SAM2.1.

    Replace with a real SAM2.1 adapter for actual experiments.
    """

    def __init__(self, stage1_dim=64, stage2_dim=96, stage4_dim=128, prompt_dim=256):
        super().__init__()
        self.stage1_dim = stage1_dim
        self.stage2_dim = stage2_dim
        self.stage4_dim = stage4_dim
        self.prompt_dim = prompt_dim

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, stage1_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stage1_dim, stage1_dim, 3, padding=1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(stage1_dim, stage2_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(stage2_dim, stage2_dim, 3, padding=1),
            nn.GELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(stage2_dim, stage4_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(stage4_dim, stage4_dim, 3, padding=1),
            nn.GELU(),
        )

        self.prompt_fuse = nn.Sequential(nn.Conv2d(stage4_dim + prompt_dim, stage4_dim, 1), nn.GELU())
        self.mask_head = nn.Sequential(
            nn.Conv2d(stage4_dim, stage4_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(stage4_dim, 1, 1),
        )

    def encode_image(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s4 = self.enc4(s2)
        return {"stage1": s1, "stage2": s2, "stage4": s4}

    def decode_masks(self, feats: dict, dense_prompt: torch.Tensor, sparse_coords=None, sparse_labels=None, memory_tokens=None):
        s4 = feats["stage4"]
        dp = F.interpolate(dense_prompt, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.prompt_fuse(torch.cat([s4, dp], dim=1))
        logits = self.mask_head(x)
        H, W = feats["stage1"].shape[-2:]
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
