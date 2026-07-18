"""
Shared utilities for Phase 2 style-transfer experiments.

Images are handled as float tensors in [0, 1] with shape (1, 3, H, W).
Any model-specific normalisation (e.g. ImageNet mean/std for the Gatys VGG,
or the learned first-layer normalisation baked into the AdaIN encoder) lives
inside the respective model module, not here.
"""

import time

import numpy as np
import torch
from PIL import Image


def get_device():
    """Prefer Apple MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path, short_side=None, square=None):
    """Load an RGB image as a (1, 3, H, W) tensor in [0, 1].

    short_side: if given, resize so the *shorter* side equals this many pixels
                (aspect ratio preserved). Used to standardise resolution so the
                Gatys/AdaIN runtime comparison is fair.
    square:     if given, resize to (square, square) ignoring aspect ratio.
    """
    img = Image.open(path).convert("RGB")
    if square is not None:
        img = img.resize((square, square), Image.LANCZOS)
    elif short_side is not None:
        w, h = img.size
        scale = short_side / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # .contiguous() matters: LBFGS (Gatys) calls grad.view(-1), which requires a
    # contiguous tensor — a bare permute() leaves non-contiguous strides.
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_pil(t):
    """(1, 3, H, W) tensor in [0, 1] -> PIL.Image."""
    t = t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((t * 255.0).round().astype(np.uint8))


def save_image(t, path):
    to_pil(t).save(path)


def match_size(src, ref):
    """Resize src tensor (1,3,h,w) to ref's spatial size via bilinear interp."""
    return torch.nn.functional.interpolate(
        src, size=ref.shape[-2:], mode="bilinear", align_corners=False
    )


class Timer:
    """Context manager that records wall-clock seconds, syncing the accelerator
    first so GPU/MPS timings are accurate (kernels are async)."""

    def __init__(self, device):
        self.device = device
        self.seconds = 0.0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self._t0
        return False
