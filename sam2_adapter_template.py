"""Template for integrating facebookresearch/sam2 (SAM 2.1).

You must wrap your SAM2.1 model as a subclass of `Sam21BackboneAdapter`.

Required methods:
- encode_image(x) -> {stage1, stage2, stage4}
- decode_masks(feats, dense_prompt, sparse_coords=None, sparse_labels=None, memory_tokens=None) -> mask_logits

Notes:
- You may fuse `dense_prompt` into stage4 embeddings before calling the SAM2.1 mask decoder,
  OR convert prompt maps into SAM2.1 mask-prompt pathway.
- For stage1/2 features, if not easy to expose, start with stage4 only and duplicate,
  but performance on root hairs may drop.
"""

import torch
from mdrs.models.autorootsam import Sam21BackboneAdapter

class Sam2Adapter(Sam21BackboneAdapter):
    def __init__(self, sam2_model):
        super().__init__()
        self.sam2 = sam2_model

    def encode_image(self, x: torch.Tensor) -> dict:
        # TODO: return multi-scale features from SAM2.1 image encoder
        raise NotImplementedError

    def decode_masks(self, feats: dict, dense_prompt: torch.Tensor, sparse_coords=None, sparse_labels=None, memory_tokens=None) -> torch.Tensor:
        # TODO: call SAM2.1 mask decoder using dense_prompt (+ optional sparse peaks)
        raise NotImplementedError
