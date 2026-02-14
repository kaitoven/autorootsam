import torch
import torch.nn as nn
import torch.nn.functional as F

from .lwa_pp import LWAPlusPlus
from .prompt_maps_tl import (
    PromptMapGeneratorTL,
    PromptMapEncoder,
    sample_sparse_tokens_from_maps,
    sample_sparse_points_from_maps,
)
from .topology_decoder import TopologyAuxHead, GatedFusionUnit, MemoryWriterPlusPlus
from .heads import RootnessHead, BoundaryHead


class Sam21BackboneAdapter(nn.Module):
    """Adapter interface for an external SAM2.1 implementation.

    You must implement:
      - encode_image(x) -> dict: stage1, stage2, stage4
      - decode_masks(feats, dense_prompt, sparse_coords=None, sparse_labels=None, memory_tokens=None)
        -> mask_logits (B,1,H,W)

    dense_prompt: (B, prompt_dim, h4, w4) from our prompt maps.
    sparse_coords: optional (B,K,2) in [-1,1] from prompt peaks.
    memory_tokens: optional (B,N,token_dim) from previous frames.
    """

    def encode_image(self, x: torch.Tensor) -> dict:
        raise NotImplementedError

    def decode_masks(
        self,
        feats: dict,
        dense_prompt: torch.Tensor,
        sparse_coords=None,
        sparse_labels=None,
        memory_tokens=None,
    ) -> torch.Tensor:
        raise NotImplementedError


class BoundaryRefiner(nn.Module):
    def __init__(self, beta: float = 0.75):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, mask_logits: torch.Tensor, boundary_logits: torch.Tensor) -> torch.Tensor:
        b = torch.sigmoid(boundary_logits)
        return mask_logits + self.beta * (b - 0.5)


class RootSTCSAM21PPTL(nn.Module):
    """Root-STC-SAM 2.1++-TL (Final / Optimal).

    Key difference vs video-style STC:
    - PRMI is *time-lapse* (Δt in days, often 10–30+ days), so memory usage MUST be
      Δt-aware (decayed), not optical-flow warping.

    Pipeline:
    - Backbone encode -> (stage1/2 enhanced by LWA++)
    - PromptMapGeneratorTL -> center/tip/bg_hard (Δt-aware memory gating)
    - PromptMapEncoder -> dense prompt embedding
    - SAM2.1 decoder -> mask_logits_raw
    - Topology head -> skel + tangent/flux
    - GFU bridge -> fuse
    - Boundary refine -> mask_logits
    - Rootness head -> suppress false positives
    - MemoryWriter++ -> memory_token (includes anchor + Δt embedding)
    """

    def __init__(
        self,
        backbone: Sam21BackboneAdapter,
        stage1_dim: int,
        stage2_dim: int,
        stage4_dim: int,
        prompt_dim: int = 256,
        num_heads: int = 8,
        # Hybrid Route-3+ sparse prompt sampling
        pos_k_center: int = 6,
        pos_k_tip: int = 2,
        neg_k_bg: int = 4,
        # Backward-compat: if >0 and (pos/neg not overridden), we sample k from centerline
        sample_k: int = 0,
        tau_days: float = 14.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.sample_k = sample_k
        self.pos_k_center = int(pos_k_center)
        self.pos_k_tip = int(pos_k_tip)
        self.neg_k_bg = int(neg_k_bg)
        self.prompt_dim = prompt_dim

        self.lwa1 = LWAPlusPlus(stage1_dim)
        self.lwa2 = LWAPlusPlus(stage2_dim)

        self.rootness = RootnessHead(stage4_dim)

        self.prompter = PromptMapGeneratorTL(
            dim=prompt_dim,
            stage1_dim=stage1_dim,
            stage2_dim=stage2_dim,
            stage4_dim=stage4_dim,
            num_heads=num_heads,
            tau_days=tau_days,
        )
        self.map_encoder = PromptMapEncoder(prompt_dim)

        self.topo = TopologyAuxHead(stage1_dim=stage1_dim)
        self.gfu = GatedFusionUnit()

        self.boundary = BoundaryHead(stage1_dim)
        self.boundary_refiner = BoundaryRefiner()

        self.mem_writer = MemoryWriterPlusPlus(stage4_dim=stage4_dim, token_dim=prompt_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory_tokens=None,
        flux_prev=None,
        anchor=None,
        delta_t: torch.Tensor | None = None,
        rootness_thresh: float | None = None,
        mem_write_thresh: float | None = 0.25,
        return_aux: bool = True,
    ):
        feats = self.backbone.encode_image(x)
        s1, s2, s4 = feats["stage1"], feats["stage2"], feats["stage4"]

        s1e, gate1 = self.lwa1(s1)
        s2e, gate2 = self.lwa2(s2)

        root_logit = self.rootness(s4)

        maps, aux = self.prompter(
            {"stage1": s1e, "stage2": s2e, "stage4": s4},
            memory_tokens=memory_tokens,
            flux_prev=flux_prev,
            anchor=anchor,
            delta_days=delta_t,
        )
        dense_prompt = self.map_encoder(maps, size_hw=s4.shape[-2:])

        sparse_coords = None
        sparse_labels = None
        # Hybrid Route-3+ (pos center/tip + neg bg_hard)
        if (self.pos_k_center + self.pos_k_tip + self.neg_k_bg) > 0:
            sparse_coords, sparse_labels = sample_sparse_points_from_maps(
                maps,
                k_pos_center=self.pos_k_center,
                k_pos_tip=self.pos_k_tip,
                k_neg_bg=self.neg_k_bg,
            )
        # Backward compatible: sample from centerline only
        elif self.sample_k > 0:
            sparse_coords = sample_sparse_tokens_from_maps(maps, k=self.sample_k)

        mask_logits_raw = self.backbone.decode_masks(
            {**feats, "stage1": s1e, "stage2": s2e, "stage4": s4},
            dense_prompt,
            sparse_coords=sparse_coords,
            sparse_labels=sparse_labels,
            memory_tokens=memory_tokens,
        )

        skel_logits, flux = self.topo(s1e, mask_logits_raw)
        fused_logits = self.gfu(mask_logits_raw, torch.sigmoid(skel_logits))

        bnd_logits = self.boundary(s1e, out_hw=fused_logits.shape[-2:])
        mask_logits = self.boundary_refiner(fused_logits, bnd_logits)

        # Rootness suppression at inference-time
        if rootness_thresh is not None:
            root_prob = torch.sigmoid(root_logit)
            suppress = (root_prob < rootness_thresh).float().view(-1, 1, 1, 1)
            mask_logits = mask_logits * (1.0 - suppress) + (-20.0) * suppress

        if anchor is None:
            anchor_token = torch.zeros((x.shape[0], self.prompt_dim), device=x.device, dtype=x.dtype)
        else:
            anchor_token = aux.get("anchor_token", torch.zeros((x.shape[0], self.prompt_dim), device=x.device, dtype=x.dtype))

        mem_token = self.mem_writer(s4, mask_logits, flux, anchor_token, delta_t_days=delta_t)

        # Rootness-gated memory writing (critical for PRMI has_root=0 abundance)
        if mem_write_thresh is not None:
            root_prob = torch.sigmoid(root_logit).view(-1, 1, 1)
            if mem_write_thresh <= 0:
                gate = root_prob
            else:
                gate = (root_prob >= float(mem_write_thresh)).float()
            mem_token = mem_token * gate

        out = {
            "mask_logits": mask_logits,
            "mask_logits_raw": mask_logits_raw,
            "center_logits": maps["center"],
            "tip_logits": maps["tip"],
            "bg_hard_logits": maps["bg_hard"],
            "skel_logits": skel_logits,
            "flux": flux,
            "boundary_logits": bnd_logits,
            "rootness_logit": root_logit,
            "memory_token": mem_token,
            "gate1": gate1,
            "gate2": gate2,
        }
        if return_aux:
            out.update(aux)
            out["sparse_coords"] = sparse_coords
            out["sparse_labels"] = sparse_labels
        return out
