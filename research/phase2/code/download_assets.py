"""
Fetch the assets needed to run Phase 2 that are intentionally NOT committed:
  * AdaIN pretrained encoder (vgg_normalised.pth, 80 MB) + decoder (14 MB)
  * the content/style images are already committed under ../inputs, but this
    script can re-fetch them too if that folder is empty.

The torchvision VGG-19 used by the Gatys method downloads itself automatically
on first use and is cached under ~/.cache/torch, so it is not handled here.

Usage:  python download_assets.py
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS = os.path.join(HERE, "weights")

# GitHub release assets from naoto0804/pytorch-AdaIN (the reference PyTorch
# implementation of Huang & Belongie 2017).
ADAIN_BASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0"
WEIGHT_FILES = ["vgg_normalised.pth", "decoder.pth"]

IMG_BASE = "https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/master/input"
CONTENT = ["chicago", "sailboat", "cornell", "golden_gate"]
STYLE = ["la_muse", "brushstrokes", "en_campo_gris", "antimonocromatismo",
         "woman_with_hat_matisse"]


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  exists: {os.path.relpath(dst, ROOT)}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  downloading {url} -> {os.path.relpath(dst, ROOT)}")
    urllib.request.urlretrieve(url, dst)


def main():
    print("AdaIN pretrained weights:")
    for name in WEIGHT_FILES:
        fetch(f"{ADAIN_BASE}/{name}", os.path.join(WEIGHTS, name))

    content_dir = os.path.join(ROOT, "inputs", "content")
    if not os.path.isdir(content_dir) or not os.listdir(content_dir):
        print("Content/style images:")
        for n in CONTENT:
            fetch(f"{IMG_BASE}/content/{n}.jpg",
                  os.path.join(ROOT, "inputs", "content", f"{n}.jpg"))
        for n in STYLE:
            fetch(f"{IMG_BASE}/style/{n}.jpg",
                  os.path.join(ROOT, "inputs", "style", f"{n}.jpg"))
    else:
        print("Content/style images already present under inputs/.")
    print("Done.")


if __name__ == "__main__":
    main()
