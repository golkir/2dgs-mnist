"""Fit every image with N 2D Gaussians and save the Gaussian-set dataset.

Examples
  python build_dataset.py --dataset mnist --out data/mnist_g256
  python build_dataset.py --dataset folder --root ~/data/celeba/img_align_celeba \
      --size 64 --crop 140 --batch 64 --limit 30000 --out data/celeba64_g256
  python build_dataset.py --dataset synthetic --limit 64 --iters 400 --out data/smoke

Output (in --out):
  shard_00000.pt ...   one per batch; re-running skips finished shards (resume)
  gaussians.pt         merged dataset (written by the final step)
  meta.json            settings, conventions and PSNR stats
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from gsfit.data import make_dataset
from gsfit.fit import FitConfig, fit_batch
from gsfit.gaussians import shape_to_cov


def to_int_label(y):
    if isinstance(y, torch.Tensor) and y.ndim == 0:
        return int(y)
    if isinstance(y, int):
        return y
    return -1


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([to_int_label(b[1]) for b in batch], dtype=torch.long)
    return xs, ys


def merge(out_dir: str, meta: dict):
    files = sorted(glob.glob(os.path.join(out_dir, "shard_*.pt")))
    parts = [torch.load(f) for f in files]
    data = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    order = data["index"].argsort()
    data = {k: v[order] for k, v in data.items()}
    data["cov"] = shape_to_cov(data["shape"])            # (M,N,3): s11, s12, s22
    torch.save(data, os.path.join(out_dir, "gaussians.pt"))

    cov = data["cov"]
    det = cov[..., 0] * cov[..., 2] - cov[..., 1] ** 2
    meta.update({
        "num_images": int(data["mu"].shape[0]),
        "psnr_mean": float(data["psnr"].mean()),
        "psnr_p05": float(data["psnr"].quantile(0.05)),
        "psnr_min": float(data["psnr"].min()),
        "all_cov_positive_definite": bool(((cov[..., 0] > 0) & (det > 0)).all()),
        "alive_fraction": float(data["alive"].float().mean()),
        "tensors": {k: list(v.shape) for k, v in data.items()},
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "celeba", "folder", "synthetic"])
    ap.add_argument("--root", default="data/raw", help="download dir, or image folder for --dataset folder")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--size", type=int, default=None, help="resize to size x size (default 28 MNIST, 64 others)")
    ap.add_argument("--crop", type=int, default=None, help="center crop before resize (CelebA: 140)")
    ap.add_argument("--limit", type=int, default=None, help="only fit the first K images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gaussians", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256, help="images optimised together (lower if out of memory)")
    ap.add_argument("--allow-negative", action="store_true", help="allow negative colour amplitudes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ds = make_dataset(args.dataset, args.root, args.split, args.size, args.crop,
                      n_synth=args.limit or 1000)
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate)
    cfg = FitConfig(n_gaussians=args.n_gaussians, iters=args.iters,
                    nonneg_color=not args.allow_negative)

    n_batches = len(dl)
    t0 = time.time()
    start = 0
    for b, (x, y) in enumerate(dl):
        path = os.path.join(args.out, f"shard_{b:05d}.pt")
        idx = torch.arange(start, start + x.shape[0])
        start += x.shape[0]
        if os.path.exists(path):
            continue
        res = fit_batch(x.to(args.device), cfg)
        res["label"] = y
        res["index"] = idx
        torch.save(res, path + ".tmp")
        os.replace(path + ".tmp", path)          # atomic, so resume never sees half a shard
        el = time.time() - t0
        print(f"[{b + 1}/{n_batches}] psnr {res['psnr'].mean():.2f} dB "
              f"(min {res['psnr'].min():.2f})  {el / 60:.1f} min", flush=True)

    H, W = x.shape[-2:]
    meta = {
        "dataset": args.dataset, "split": args.split, "image_size": [int(H), int(W)],
        "channels": int(x.shape[1]), "n_gaussians": args.n_gaussians,
        "fit_config": vars(cfg),
        "conventions": {
            "coords": "normalised [-1,1]; x right, y down; pixel centre j -> (j+0.5)/W*2-1",
            "render": "I(p) = sum_i color_i * exp(-0.5 (p-mu_i)^T Sigma_i^-1 (p-mu_i)); no opacity, no sorting",
            "shape": "(log s1, log s2, theta); Sigma = R(theta) diag(s1^2, s2^2) R(theta)^T; theta in [-pi/2, pi/2)",
            "cov": "(s11, s12, s22) of Sigma",
            "order": "slot k started at grid cell k (row-major, sqrt(N) x sqrt(N)) for every image",
            "alive": "False = unused Gaussian, reset to its slot with colour 0 (renders nothing)",
        },
    }
    meta = merge(args.out, meta)
    print(json.dumps({k: meta[k] for k in ("num_images", "psnr_mean", "psnr_p05",
                                            "all_cov_positive_definite", "alive_fraction")}, indent=2))


if __name__ == "__main__":
    main()
