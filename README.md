# gs2d — Stage 0: images → sets of 2D Gaussians

Each image is fitted with N = 256 2D Gaussians by per-image gradient descent (no neural
network). The output is a dataset of Gaussian sets for the next stage: flat flow
matching vs product-BW flow matching.

## Install
```bash
pip install -r requirements.txt     # torch, torchvision, pillow, matplotlib, numpy
```

## Run
```bash
# 1. Smoke test, fully offline (font-drawn digits), CPU ok, ~2 min
python build_dataset.py --dataset synthetic --limit 32 --batch 16 --iters 600 --out data/smoke
python preview.py --out data/smoke --dataset synthetic

# 2. MNIST (downloads automatically), all 60k train images
python build_dataset.py --dataset mnist --out data/mnist_g256
python preview.py --out data/mnist_g256 --dataset mnist

# 3. CelebA 64x64: download img_align_celeba yourself (torchvision's downloader often fails)
python build_dataset.py --dataset folder --root /path/to/img_align_celeba \
    --size 64 --crop 140 --batch 64 --limit 30000 --out data/celeba64_g256
python preview.py --out data/celeba64_g256 --dataset folder \
    --root /path/to/img_align_celeba --size 64 --crop 140
```
Runs resume: each batch is saved as a shard, and finished shards are skipped.
If you run out of GPU memory, lower `--batch`. Memory grows with batch × N × pixels;
64×64 images need about 5× smaller batches than 28×28.

## Output: `gaussians.pt`
| key | shape | meaning |
|---|---|---|
| `mu` | (M, N, 2) | centres (x, y) in [-1, 1], with y pointing down |
| `shape` | (M, N, 3) | (log s1, log s2, θ): Σ = R(θ) diag(s1², s2²) R(θ)ᵀ |
| `cov` | (M, N, 3) | (Σ11, Σ12, Σ22): use this for BW maths |
| `color` | (M, N, C) | amplitude per channel (C = 1 or 3), ≥ 0 |
| `alive` | (M, N) | False = unused Gaussian (colour 0, back at its grid slot) |
| `psnr`, `label`, `index` | (M,) | fit quality, class label (−1 if none), dataset index |

Rendering is an additive sum, with no opacity and no depth sorting:
`I(p) = Σ_i color_i · exp(−½ (p−μ_i)ᵀ Σ_i⁻¹ (p−μ_i))`.
Use `gsfit.gaussians.render(mu, shape, color, H, W)` to turn generated sets back into
images. If your generator outputs covariances directly, convert them to shape with an
eigendecomposition (log of the square roots of the eigenvalues, plus the angle of the
first eigenvector).

## Design choices that matter for the flow-matching stage
- **Same initialisation for every image:** a 16×16 grid of round Gaussians. Slot k
  covers roughly the same region in every image, so similar images get similar sets.
  Set-level OT matching in stage 1 still works on any order.
- **No opacity parameter.** It would duplicate the colour amplitude, giving the same
  image many parameter vectors.
- **Radius floor of 0.3 px and cap of half the image.** This keeps every Σ well
  conditioned, so BW square roots and inverses are stable.
- **Unused Gaussians are reset**, not left wherever they drifted. On MNIST only about
  20% of the 256 are used, because the black background needs none. Options:
  generate all 256 and let the model learn colour 0; or treat `alive` as a variable
  set size (masking, unbalanced OT); or fit with `--n-gaussians 64` for MNIST.
- θ is wrapped to [-π/2, π/2). The angle still wraps at the ends, so flat FM should
  use `cov` (or log-scales plus a cos 2θ / sin 2θ encoding), not raw θ.

## Rough cost
On a single modern GPU, MNIST at 60k images × 1500 iterations takes on the order of
an hour, and 30k CelebA 64×64 images take a few hours. Measure on your GPU with
`--limit 1024` first. A CPU only works for smoke tests.

## Files
```
build_dataset.py   CLI: fit, shard, resume, merge, write meta.json
preview.py         original | render | ellipses grid, plus stats
gsfit/gaussians.py parameterisation, grid init, renderer (reused in stage 1)
gsfit/fit.py       batched Adam fitting with clamps and dead-Gaussian reset
gsfit/data.py      MNIST, FashionMNIST, CelebA, image folder, synthetic digits
```
