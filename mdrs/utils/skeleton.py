import torch
import torch.nn.functional as F

# Differentiable morphology used by soft-skeleton / clDice

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    p1 = F.max_pool2d(img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = F.max_pool2d(img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.max(p1, p2)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft-skeleton. img: (B,1,H,W) in [0,1]."""
    skel = torch.zeros_like(img)
    cur = img
    for _ in range(iters):
        opened = soft_open(cur)
        delta = torch.relu(cur - opened)
        skel = torch.max(skel, delta)
        cur = soft_erode(cur)
    return skel


def cldice_loss(pred_prob: torch.Tensor, gt_prob: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Soft-clDice for connectivity (thin structures)."""
    skel_pred = soft_skel(pred_prob, iters)
    skel_gt = soft_skel(gt_prob, iters)
    tprec = (torch.sum(skel_pred * gt_prob) + eps) / (torch.sum(skel_pred) + eps)
    trec = (torch.sum(skel_gt * pred_prob) + eps) / (torch.sum(skel_gt) + eps)
    cl_dice = 2.0 * tprec * trec / (tprec + trec + eps)
    return 1.0 - cl_dice
