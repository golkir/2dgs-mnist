"""Per-image optimisation of N 2D Gaussians (batched; no neural network)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .gaussians import grid_init, pixel_grid, render


@dataclass
class FitConfig:
    n_gaussians: int = 256
    iters: int = 1500
    lr_mu: float = 3e-3
    lr_shape: float = 1e-2
    lr_color: float = 1e-2
    sigma_frac: float = 0.6       # initial radius as a fraction of grid spacing
    min_sigma_px: float = 0.3     # floor on each radius, in pixels
    max_sigma_frac: float = 0.5   # cap, as a fraction of the image width
    nonneg_color: bool = True     # clamp colours >= 0 (natural for additive images)
    mu_bound: float = 1.1         # keep centres within [-b, b]
    dead_thresh: float = 1e-3     # Gaussians with max |colour| below this count as unused
    reset_dead: bool = True       # put unused Gaussians back at their initial slot


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def fit_batch(images: torch.Tensor, cfg: FitConfig) -> dict[str, torch.Tensor]:
    """Fit one batch of images.

    images (B,C,H,W) in [0,1]. Returns a dict of CPU float32 tensors:
      mu (B,N,2), shape (B,N,3), color (B,N,C), alive (B,N) bool, psnr (B,)
    """
    B, C, H, W = images.shape
    dev = images.device
    grid = pixel_grid(H, W, dev)
    mu, shape, color = grid_init(images, cfg.n_gaussians, cfg.sigma_frac)
    mu_init, shape_init = mu.clone(), shape.clone()
    mu.requires_grad_(True); shape.requires_grad_(True); color.requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [mu], "lr": cfg.lr_mu},
        {"params": [shape], "lr": cfg.lr_shape},
        {"params": [color], "lr": cfg.lr_color},
    ])
    base = [g["lr"] for g in opt.param_groups]
    log_min = math.log(cfg.min_sigma_px * 2.0 / W)
    log_max = math.log(cfg.max_sigma_frac * 2.0)

    for it in range(cfg.iters):
        # cosine decay to 5% of the starting learning rate
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / cfg.iters))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * f
        pred = render(mu, shape, color, H, W, grid)
        # sum over the batch, so each image's gradient doesn't depend on B
        loss = ((pred - images) ** 2).flatten(1).mean(1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu.clamp_(-cfg.mu_bound, cfg.mu_bound)
            shape[..., :2].clamp_(log_min, log_max)
            if cfg.nonneg_color:
                color.clamp_(min=0)

    with torch.no_grad():
        alive = color.abs().amax(-1) > cfg.dead_thresh               # (B,N)
        if cfg.reset_dead:
            # An unused Gaussian barely changes the image, but its leftover position/shape is
            # arbitrary. Resetting it to its slot (colour 0) makes the dataset deterministic.
            mu[~alive] = mu_init[~alive]
            shape[~alive] = shape_init[~alive]
            color[~alive] = 0
        # wrap the angle into [-pi/2, pi/2): theta and theta+pi give the same ellipse
        shape[..., 2] = torch.remainder(shape[..., 2] + math.pi / 2, math.pi) - math.pi / 2
        pred = render(mu, shape, color, H, W, grid).clamp(0, 1)
        p = psnr(pred, images)
    return {
        "mu": mu.detach().float().cpu(),
        "shape": shape.detach().float().cpu(),
        "color": color.detach().float().cpu(),
        "alive": alive.cpu(),
        "psnr": p.float().cpu(),
    }
