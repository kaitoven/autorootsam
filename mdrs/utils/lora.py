import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    y = xW + (alpha/r) * (xA)B
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        lora = (self.drop(x) @ self.A.t()) @ self.B.t()
        return y + self.scaling * lora


def replace_linear_with_lora(module: nn.Module, name: str, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> bool:
    child = getattr(module, name)
    if isinstance(child, nn.Linear):
        setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        return True
    return False


def apply_lora(
    model: nn.Module,
    keywords: Iterable[str] = ("q", "v"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> int:
    """Best-effort LoRA injection by attribute name match (e.g., q_proj, v_proj)."""
    hit = 0
    for m in model.modules():
        for n, _ in list(m.named_children()):
            if any(k in n.lower() for k in keywords):
                hit += int(replace_linear_with_lora(m, n, r=r, alpha=alpha, dropout=dropout))
    return hit
