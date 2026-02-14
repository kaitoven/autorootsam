import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Haar DWT implemented by fixed group conv."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if (h % 2) or (w % 2):
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
            b, c, h, w = x.shape
        weight = self.filt.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(b, c, 4, h // 2, w // 2)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """Haar inverse DWT."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor):
        b, c, h, w = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).view(b, 4 * c, h, w)
        weight = self.filt.repeat(c, 1, 1, 1)
        x = F.conv_transpose2d(y, weight, stride=2, padding=0, groups=c)
        return x


def _line_kernel(size: int, angle_deg: float, thickness: float = 1.0) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    c = (size - 1) / 2.0
    ang = np.deg2rad(angle_deg)
    for i in range(size):
        for j in range(size):
            x = j - c
            y = i - c
            dist = abs(-np.sin(ang) * x + np.cos(ang) * y)
            if dist <= thickness * 0.5:
                k[i, j] = 1.0
    s = float(k.sum())
    if s > 0:
        k /= s
    return k


class OrientedLineEnhancer(nn.Module):
    """Light depthwise oriented-line filter bank (initialized as line kernels, learnable)."""

    def __init__(self, channels: int, ksize: int = 7, n_orient: int = 8):
        super().__init__()
        self.convs = nn.ModuleList()
        angles = [i * (180.0 / n_orient) for i in range(n_orient)]
        for a in angles:
            conv = nn.Conv2d(channels, channels, ksize, padding=ksize // 2, groups=channels, bias=False)
            w = torch.from_numpy(_line_kernel(ksize, a)).view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1)
            with torch.no_grad():
                conv.weight.copy_(w)
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        return torch.stack(res, dim=0).amax(dim=0)


class LWAPlusPlus(nn.Module):
    """LWA++ = Wavelet-HF branch + Oriented-Line branch + Unified gate + Zero-init residual."""

    def __init__(self, channels: int, gate_hidden: int = 64, ksize: int = 7, n_orient: int = 8):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.line = OrientedLineEnhancer(channels, ksize=ksize, n_orient=n_orient)

        # gate computed from [HF(3C) + line_ds(C)] => 4C
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, gate_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, channels * 4, 1),
            nn.Sigmoid(),
        )
        # zero-init residual strength
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        ll, lh, hl, hh = self.dwt(x)
        hf = torch.cat([lh, hl, hh], dim=1)  # (B,3C,H/2,W/2)

        line_r = self.line(x)
        line_ds = F.avg_pool2d(line_r, kernel_size=2, stride=2)

        gate_in = torch.cat([hf, line_ds], dim=1)  # (B,4C,H/2,W/2)
        g = self.gate(gate_in)
        ghf, gline = g[:, : hf.shape[1]], g[:, hf.shape[1] :]

        # HF enhancement
        hf_enh = hf * (1.0 + ghf)
        lh2, hl2, hh2 = torch.chunk(hf_enh, 3, dim=1)
        x_hf = self.idwt(ll, lh2, hl2, hh2)
        x_hf = x_hf[..., : x.shape[-2], : x.shape[-1]]

        # line enhancement
        gline_up = F.interpolate(gline, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_line = x + gline_up * line_r

        x_enh = 0.5 * (x_hf + x_line)
        out = x + self.scale * (x_enh - x)
        return out, g
