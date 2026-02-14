import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenCrossAttention(nn.Module):
    """Pre-norm cross attention: Q attends to KV."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(dim, dropout=dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.ffn(self.norm_out(q))
        return q
