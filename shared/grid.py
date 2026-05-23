"""Tiny helpers for image grids and tensor<->PIL. Used across every lesson."""
from pathlib import Path

import torch
from PIL import Image


def to_uint8_grid(imgs: torch.Tensor, nrow: int = 5) -> Image.Image:
    """imgs: (N, 3, H, W) in [-1, 1]. Returns a PIL grid with `nrow` columns."""
    imgs = imgs.detach().clamp(-1, 1).add(1).div(2)             # [-1,1] -> [0,1]
    imgs = imgs.mul(255).round().to(torch.uint8).cpu()
    n, c, h, w = imgs.shape
    ncol = (n + nrow - 1) // nrow
    canvas = torch.zeros(c, ncol * h, nrow * w, dtype=torch.uint8)
    for i, im in enumerate(imgs):
        r, k = divmod(i, nrow)
        canvas[:, r * h:(r + 1) * h, k * w:(k + 1) * w] = im
    return Image.fromarray(canvas.permute(1, 2, 0).numpy())


def save_grid(imgs: torch.Tensor, path: str | Path, nrow: int = 5):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    to_uint8_grid(imgs, nrow=nrow).save(p)
    print(f"saved {p}  ({imgs.shape[0]} imgs)")
