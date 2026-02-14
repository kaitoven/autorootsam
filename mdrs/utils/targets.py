import torch
import torch.nn.functional as F

from .skeleton import soft_skel


def mask_to_boundary(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B,1,H,W) in {0,1}. returns boundary map in {0,1}."""
    m = mask.float()
    dil = F.max_pool2d(m, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-m, 3, stride=1, padding=1)
    b = torch.clamp(dil - ero, 0.0, 1.0)
    return b


def mask_to_centerline(mask: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """Soft skeleton as centerline target."""
    m = mask.float()
    return torch.clamp(soft_skel(m, iters=iters), 0.0, 1.0)


def centerline_to_tips(center: torch.Tensor, thr: float = 0.15) -> torch.Tensor:
    """Approx endpoints from (soft) centerline.

    center: (B,1,H,W) in [0,1]
    Returns tips (B,1,H,W) in [0,1]
    """
    c = (center > thr).float()
    # count 8-neighbors
    k = torch.ones((1, 1, 3, 3), device=c.device, dtype=c.dtype)
    neigh = F.conv2d(c, k, padding=1)
    # For a skeleton pixel, neigh includes itself; endpoints have ~2 (self + 1 neighbor)
    tips = ((c > 0.5) & (neigh <= 2.5)).float()
    tips = F.max_pool2d(tips, 3, stride=1, padding=1)
    return tips


def pseudo_bg_hard_from_image(img: torch.Tensor) -> torch.Tensor:
    """Generate pseudo hard-negative map from an RGB image.

    img: (B,3,H,W) in [0,1]
    returns: (B,1,H,W) in {0,1}, high where line-like edges exist.

    Intended for has_root=0 tiles to mine cracks/scratches.
    """
    gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mmax = mag.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    mag = mag / mmax
    return (mag > 0.35).float()
