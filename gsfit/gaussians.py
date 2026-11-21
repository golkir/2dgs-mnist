"""2D Gaussian image model: parameterisation, initialisation and rendering.

Conventions (shared with the later flow-matching stage):
  * Coordinates are normalised to [-1, 1] on both axes; x goes right, y goes down
    (pixel (row=0, col=0) is the top-left corner at about (-1, -1)).
  * Each Gaussian i has
        mu_i     (2,)  centre (x, y)
        Sigma_i  (2,2) covariance = R(theta_i) diag(s1_i^2, s2_i^2) R(theta_i)^T
        color_i  (C,)  amplitude per channel (C = 1 for MNIST, 3 for RGB)
  * The image is an additive sum (no depth sorting, no alpha blending):
        I(p) = sum_i color_i * exp(-0.5 (p - mu_i)^T Sigma_i^{-1} (p - mu_i))
    Opacity is not a separate parameter; it would duplicate the colour
    amplitude and give the same image many different parameter vectors,
    which makes the generative stage harder.

Raw (optimised) parameters per Gaussian:
    mu     (2,)  centre
    shape  (3,)  (log s1, log s2, theta): radii are stored as logs, so Sigma is always
                 positive definite, and each radius can be clamped directly to a pixel floor
    color  (C,)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pixel_grid(h: int, w: int, device=None) -> torch.Tensor:
    """Pixel centres in normalised coordinates, shape (h*w, 2) as (x, y)."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def shape_to_cov(shape: torch.Tensor) -> torch.Tensor:
    """(..., 3) raw (log s1, log s2, theta) -> (..., 3) covariance entries (s11, s12, s22)."""
    v1, v2 = (2 * shape[..., 0]).exp(), (2 * shape[..., 1]).exp()
    c, s = shape[..., 2].cos(), shape[..., 2].sin()
    return torch.stack([v1 * c * c + v2 * s * s, (v1 - v2) * c * s, v1 * s * s + v2 * c * c], dim=-1)


def cov_to_matrix(cov: torch.Tensor) -> torch.Tensor:
    """(..., 3) (s11, s12, s22) -> (..., 2, 2) symmetric matrix."""
    s11, s12, s22 = cov.unbind(-1)
    row0 = torch.stack([s11, s12], -1)
    row1 = torch.stack([s12, s22], -1)
    return torch.stack([row0, row1], -2)


def render(mu: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
           h: int, w: int, grid: torch.Tensor | None = None) -> torch.Tensor:
    """Render a batch of Gaussian sets.

    mu (B,N,2), shape (B,N,3), color (B,N,C)  ->  image (B,C,h,w)

    Memory is O(B*N*h*w); for 256 Gaussians on 28x28 a batch of 512 is ~0.4 GB
    forward. Lower the batch size for bigger images.
    """
    if grid is None:
        grid = pixel_grid(h, w, mu.device)
    inv_s1 = (-shape[..., 0]).exp()[..., None]             # (B,N,1)
    inv_s2 = (-shape[..., 1]).exp()[..., None]
    c, s = shape[..., 2].cos()[..., None], shape[..., 2].sin()[..., None]
    dx = grid[None, None, :, 0] - mu[..., 0:1]             # (B,N,P)
    dy = grid[None, None, :, 1] - mu[..., 1:2]
    u = (c * dx + s * dy) * inv_s1                         # coordinates along the two axes,
    v = (-s * dx + c * dy) * inv_s2                        # in units of each radius
    wgt = torch.exp(-0.5 * (u * u + v * v))                # (B,N,P)
    img = torch.einsum("bnp,bnc->bcp", wgt, color)        # (B,C,P)
    return img.reshape(img.shape[0], img.shape[1], h, w)


def grid_init(images: torch.Tensor, n_gaussians: int, sigma_frac: float = 0.6):
    """Same deterministic initialisation for every image.

    Gaussians start on a regular g x g grid (g = sqrt(N)), round, with radius
    sigma_frac * grid spacing, and colour = image value under the centre,
    rescaled so a flat image stays flat. Slot k always starts at the same
    place, so "Gaussian k" means roughly the same region in every image. This
    keeps the dataset consistent for the generative stage.

    images (B,C,H,W) in [0,1]  ->  mu (B,N,2), shape (B,N,3), color (B,N,C)
    """
    g = int(round(math.sqrt(n_gaussians)))
    if g * g != n_gaussians:
        raise ValueError(f"n_gaussians must be a perfect square, got {n_gaussians}")
    B, C = images.shape[:2]
    dev = images.device
    spacing = 2.0 / g
    c = (torch.arange(g, device=dev, dtype=torch.float32) + 0.5) * spacing - 1
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    mu0 = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(B, -1, -1).clone()

    sigma0 = sigma_frac * spacing
    shape = torch.zeros(B, n_gaussians, 3, device=dev)
    shape[..., 0] = math.log(sigma0)
    shape[..., 1] = math.log(sigma0)

    # grid_sample uses the same [-1,1] (x, y) convention with align_corners=False
    samp = F.grid_sample(images, mu0[:, None], mode="bilinear", align_corners=False)
    samp = samp[:, :, 0].permute(0, 2, 1)                 # (B,N,C)
    # A grid of Gaussians of radius s with spacing a sums to 2*pi*s^2/a^2 per unit amplitude
    norm = spacing ** 2 / (2 * math.pi * sigma0 ** 2)
    color = samp * norm
    return mu0, shape, color
