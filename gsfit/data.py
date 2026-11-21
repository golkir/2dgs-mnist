"""Image sources. Every source returns float tensors (C,H,W) in [0,1] plus an int label."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FolderImages(Dataset):
    """Any folder of images (e.g. CelebA's img_align_celeba/). Label = -1.

    Center-crops to a square (crop px, default = shorter side), resizes to `size`.
    For aligned CelebA the usual setting is crop=140 (face region), size=64.
    """

    def __init__(self, root: str, size: int = 64, crop: int | None = None, gray: bool = False):
        self.files = sorted(
            os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
            if f.lower().endswith(IMG_EXT))
        if not self.files:
            raise FileNotFoundError(f"no images under {root}")
        self.size, self.crop, self.gray = size, crop, gray

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("L" if self.gray else "RGB")
        w, h = im.size
        c = self.crop or min(w, h)
        left, top = (w - c) // 2, (h - c) // 2
        im = im.crop((left, top, left + c, top + c)).resize((self.size, self.size), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
        return torch.from_numpy(a.copy()), -1


class SyntheticDigits(Dataset):
    """Offline smoke-test data: digits drawn with a font, randomly moved, scaled and
    rotated, then blurred slightly. Looks roughly like MNIST (28x28, white on black)."""

    def __init__(self, n: int = 1000, size: int = 28, seed: int = 0):
        # The font is not stored here: PIL fonts can't be pickled, and DataLoader workers
        # on macOS/Windows receive the dataset by pickling. Each worker loads its own font.
        self.n, self.size, self.seed = n, size, seed
        self._fonts = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_fonts"] = {}
        return d

    def _font(self, px: int):
        if px not in self._fonts:
            try:
                self._fonts[px] = ImageFont.load_default(size=px)
            except TypeError:  # Pillow < 10.1 has only a tiny bitmap font
                self._fonts[px] = ImageFont.load_default()
        return self._fonts[px]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        label = rng.randrange(10)
        S = self.size * 4
        im = Image.new("L", (S, S), 0)
        dr = ImageDraw.Draw(im)
        font = self._font(int(S * rng.uniform(0.6, 0.8)))
        dr.text((S / 2 + rng.uniform(-6, 6), S / 2 + rng.uniform(-6, 6)), str(label),
                fill=255, font=font, anchor="mm", stroke_width=rng.randint(2, 5), stroke_fill=255)
        im = im.rotate(rng.uniform(-15, 15), resample=Image.BILINEAR)
        im = im.resize((self.size, self.size), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32)[None] / 255.0
        return torch.from_numpy(a.copy()), label


def make_dataset(name: str, root: str, split: str = "train", size: int | None = None,
                 crop: int | None = None, n_synth: int = 1000) -> Dataset:
    name = name.lower()
    if name in ("mnist", "fashion"):
        from torchvision import datasets, transforms
        cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        tf = [transforms.ToTensor()]
        if size and size != 28:
            tf.insert(0, transforms.Resize(size, antialias=True))
        return cls(root, train=(split == "train"), download=True, transform=transforms.Compose(tf))
    if name == "celeba":
        # torchvision's CelebA downloader (Google Drive) often fails; if it does,
        # download img_align_celeba yourself and use --dataset folder.
        from torchvision import datasets, transforms
        s = size or 64
        tf = transforms.Compose([transforms.CenterCrop(crop or 140),
                                 transforms.Resize(s, antialias=True),
                                 transforms.ToTensor()])
        return datasets.CelebA(root, split="train" if split == "train" else "test",
                               download=True, transform=tf)
    if name == "folder":
        return FolderImages(root, size=size or 64, crop=crop)
    if name == "synthetic":
        return SyntheticDigits(n=n_synth, size=size or 28)
    raise ValueError(f"unknown dataset {name}")
