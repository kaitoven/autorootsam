"""SAM2 / SAM2.1 official adapter for Root-STC-SAM2.1++-TL.

This adapter lets our prompt-free pipeline use the official
facebookresearch/sam2 implementation (SAM 2 / SAM 2.1).

Design choice (pragmatic / robust):
- We reuse SAM2's own PromptEncoder+MaskDecoder via SAM2Base._forward_sam_heads.
- Our differentiable prompt embedding (dense_prompt) is converted to a *mask prompt*
  (mask_inputs) that SAM2 can consume.
- Optional sparse peak coords are converted to SAM2 point prompts.

Important constraints:
- The official SAM2 configs assume a fixed square `image_size` (e.g. 1024).
  For best results, feed images already padded/resized to that resolution.
  In minirhizotron PRMI, padding (no scaling) is typically preferred.

References (official):
- SAM2 usage + build_sam2 API: https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Sam2AdapterConfig:
    model_cfg: str
    checkpoint: str
    image_size: int = 1024
    bb_feat_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (256, 256),
        (128, 128),
        (64, 64),
    )
    do_normalize: bool = True


class SAM2OfficialBackboneAdapter(nn.Module):
    """Backbone adapter that exposes `encode_image` and `decode_masks`.

    The adapter internally holds an official SAM2Base model.
    """

    def __init__(self, cfg: Sam2AdapterConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device_str = device

        try:
            from sam2.build_sam import build_sam2
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SAM2 is not installed. Install with: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e

        # Resolve config path when using installed SAM2 package.
        cfg_path = str(cfg.model_cfg)
        if not os.path.exists(cfg_path):
            try:
                import sam2  # type: ignore
                pkg_dir = Path(sam2.__file__).resolve().parent
                for cand in (pkg_dir / cfg_path, pkg_dir.parent / cfg_path):
                    if cand.exists():
                        cfg_path = str(cand)
                        break
            except Exception:
                pass

        self.sam2 = build_sam2(cfg_path, cfg.checkpoint, device=device, mode="eval")
        self.sam2.eval()

        # Match predictor defaults
        self._bb_feat_sizes = list(cfg.bb_feat_sizes)

        # Normalization (same as SAM2Transforms)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @property
    def image_size(self) -> int:
        return int(getattr(self.sam2, "image_size", self.cfg.image_size))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.do_normalize:
            return x
        return (x - self._mean.to(x.device)) / self._std.to(x.device)

    def encode_image(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode an image batch.

        Args:
            x: (B,3,H,W) float tensor in [0,1] (recommended), already padded/resized
               to (image_size,image_size).

        Returns:
            dict with: stage1 (B,256,256,256), stage2 (B,256,128,128), stage4 (B,256,64,64)
        """
        B, C, H, W = x.shape
        assert C == 3
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"SAM2OfficialBackboneAdapter expects {self.image_size}x{self.image_size} input; got {H}x{W}. "
                "Pad or resize your tiles to match SAM2 config image_size."
            )

        x = self._normalize(x)
        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)

        # During video training, SAM2 adds no_mem_embed to the lowest-res feature.
        if getattr(self.sam2, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).contiguous().view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        # feats: [256x256, 128x128, 64x64]
        return {"stage1": feats[0], "stage2": feats[1], "stage4": feats[2]}

    def _coords_to_pixels(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized coords in [-1,1] to absolute pixel coords (x,y)."""
        # coords_norm: (B,K,2)
        s = float(self.image_size - 1)
        xy01 = (coords_norm + 1.0) * 0.5
        xy = xy01 * s
        return xy

    def _dense_prompt_to_mask_prompt(self, dense_prompt: torch.Tensor) -> torch.Tensor:
        """Convert (B,C,64,64) dense embedding to a (B,1,1024,1024) mask prompt."""
        # Very lightweight conversion: mean over channels -> [0,1] -> upsample
        m = torch.sigmoid(dense_prompt.mean(dim=1, keepdim=True))
        m = F.interpolate(m, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return m

    def decode_masks(
        self,
        feats: Dict[str, torch.Tensor],
        dense_prompt: torch.Tensor,
        sparse_coords: Optional[torch.Tensor] = None,
        sparse_labels: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode mask logits.

        Args:
            feats: output of encode_image
            dense_prompt: (B,C,64,64)
            sparse_coords: optional (B,K,2) in [-1,1]
            memory_tokens: ignored here; memory is handled in our outer TL stack.

        Returns:
            mask_logits: (B,1,image_size,image_size)
        """
        backbone_features = feats["stage4"]
        high_res_features = [feats["stage1"], feats["stage2"]]

        point_inputs = None
        if sparse_coords is not None and sparse_coords.numel() > 0:
            xy = self._coords_to_pixels(sparse_coords)
            if sparse_labels is None:
                labels = torch.ones(xy.shape[:2], dtype=torch.int32, device=xy.device)
            else:
                labels = sparse_labels.to(torch.int32)
                if labels.ndim == 3:
                    labels = labels[:, :, 0]
                labels = labels.to(device=xy.device)
            point_inputs = {"point_coords": xy, "point_labels": labels}

        mask_inputs = self._dense_prompt_to_mask_prompt(dense_prompt)

        (
            _low_res_multimasks,
            _high_res_multimasks,
            _ious,
            _low_res_masks,
            high_res_masks,
            _obj_ptr,
            _object_score_logits,
        ) = self.sam2._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=False,
        )

        # high_res_masks: (B,1,1024,1024) logits
        return high_res_masks
