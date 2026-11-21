"""Visual check of a fitted dataset: original | re-rendered | ellipses, plus stats.

  python preview.py --out data/mnist_g256 --dataset mnist --n 8
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from gsfit.data import make_dataset
from gsfit.gaussians import cov_to_matrix, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder with gaussians.pt and meta.json")
    ap.add_argument("--dataset", default=None, help="to show originals (same args as build_dataset)")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    data = torch.load(os.path.join(args.out, "gaussians.pt"))
    meta = json.load(open(os.path.join(args.out, "meta.json")))
    H, W = meta["image_size"]
    n = min(args.n, data["mu"].shape[0])
    sel = torch.linspace(0, data["mu"].shape[0] - 1, n).long()

    with torch.no_grad():
        rec = render(data["mu"][sel], data["shape"][sel], data["color"][sel], H, W).clamp(0, 1)

    orig = None
    if args.dataset:
        ds = make_dataset(args.dataset, args.root, meta["split"], args.size, args.crop,
                          n_synth=meta["num_images"])
        orig = [ds[int(data["index"][i])][0] for i in sel]

    cols = 3 if orig is not None else 2
    fig, ax = plt.subplots(n, cols, figsize=(2.4 * cols, 2.4 * n), squeeze=False)
    for r, i in enumerate(sel):
        c = 0
        if orig is not None:
            show(ax[r, c], orig[r]); ax[r, c].set_title("original", fontsize=8); c += 1
        show(ax[r, c], rec[r]); ax[r, c].set_title(f"256 Gaussians, {data['psnr'][i]:.1f} dB", fontsize=8)
        c += 1
        draw_ellipses(ax[r, c], data["mu"][i], data["cov"][i], data["color"][i])
    plt.tight_layout()
    path = os.path.join(args.out, "preview.png")
    plt.savefig(path, dpi=110)
    print("wrote", path)

    sig = torch.linalg.eigvalsh(cov_to_matrix(data["cov"])).clamp_min(0).sqrt() * W / 2  # radii in px
    print(f"PSNR mean {data['psnr'].mean():.2f} dB, 5th pct {data['psnr'].quantile(.05):.2f} dB")
    print(f"radius (px): min {sig.min():.2f}  median {sig.median():.2f}  max {sig.max():.2f}")
    print(f"anisotropy (max/min radius): median {(sig[..., 1] / sig[..., 0]).median():.2f}")
    alive = data["alive"]
    print(f"used Gaussians per image: mean {alive.sum(1).float().mean():.0f} of {alive.shape[1]}")
    sa = sig[alive]
    print(f"used only -> radius (px): min {sa.min():.2f}  median {sa.median():.2f}; "
          f"anisotropy median {(sa[:, 1] / sa[:, 0]).median():.2f}")


def show(a, img):
    img = img.numpy()
    a.imshow(img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    a.axis("off")


def draw_ellipses(a, mu, cov, color):
    M = cov_to_matrix(cov).numpy()
    amp = color.abs().amax(-1).numpy()
    amp = amp / (amp.max() + 1e-8)
    for k in range(mu.shape[0]):
        w, v = np.linalg.eigh(M[k])
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        e = Ellipse(mu[k].numpy(), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang,
                    fill=False, lw=0.5, alpha=float(0.15 + 0.85 * amp[k]), color="tab:blue")
        a.add_patch(e)
    a.set_xlim(-1, 1); a.set_ylim(1, -1); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.set_title("ellipses (1 sigma)", fontsize=8)


if __name__ == "__main__":
    main()
