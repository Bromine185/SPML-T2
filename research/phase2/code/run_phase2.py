"""
Phase 2 driver — runs BOTH style-transfer methods on a shared set of
content/style pairs and writes every figure used in analysis.pdf.

Outputs (all in ../generated_outputs):
  gatys_<c>__<s>.png / adain_<c>__<s>.png   per-pair stylised images
  comparison_grid.png                        content | style | Gatys | AdaIN
  gatys_optimization_progression.png         Gatys' iterative trajectory
  adain_flexibility_grid.png                 one AdaIN model, many styles
  adain_alpha_tradeoff.png                   content<->style knob (alpha)
  adain_style_interpolation.png              blending two styles in feature space
  runtime_comparison.png + runtime.json      timing + config

Run:  python run_phase2.py
(Set NST_WEIGHTS to override the weights directory; default ./weights.)
"""

import json
import os
import platform
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import adain_nst
import gatys_nst
from utils import Timer, get_device, load_image, save_image, to_pil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUTS = os.path.join(ROOT, "inputs")
OUT = os.path.join(ROOT, "generated_outputs")
WEIGHTS = os.environ.get("NST_WEIGHTS", os.path.join(HERE, "weights"))
os.makedirs(OUT, exist_ok=True)

SHORT_SIDE = 512          # both methods run at this resolution (fair timing)
GATYS_STEPS = 300         # optimisation iterations per image
ADAIN_TIMING_RUNS = 25    # feed-forward passes to average for timing

# Shared comparison pairs (content, style) run through BOTH methods.
PAIRS = [
    ("chicago", "la_muse"),
    ("sailboat", "brushstrokes"),
    ("cornell", "en_campo_gris"),
]
# For AdaIN's "arbitrary style" flexibility grid.
GRID_CONTENTS = ["chicago", "sailboat", "cornell"]
GRID_STYLES = ["la_muse", "brushstrokes", "en_campo_gris", "woman_with_hat_matisse"]


def cpath(name):
    return os.path.join(INPUTS, "content", f"{name}.jpg")


def spath(name):
    return os.path.join(INPUTS, "style", f"{name}.jpg")


def imshow(ax, t, title=None):
    ax.imshow(np.asarray(to_pil(t)))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)


def device_name():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return platform.processor() or platform.machine()


def main():
    torch.manual_seed(0)
    device = get_device()
    dev_str = f"{device.type} ({device_name()})"
    print(f"Device: {dev_str}\nResolution: short side {SHORT_SIDE}px\n")

    print("Loading AdaIN model (pretrained encoder + decoder)...")
    adain = adain_nst.AdaINStyleTransfer(
        os.path.join(WEIGHTS, "vgg_normalised.pth"),
        os.path.join(WEIGHTS, "decoder.pth"),
        device,
    )

    results = {
        "device": dev_str,
        "short_side": SHORT_SIDE,
        "gatys_steps": GATYS_STEPS,
        "pairs": [],
        "adain_timing_runs": ADAIN_TIMING_RUNS,
    }
    gatys_imgs, adain_imgs = {}, {}

    # ---- main comparison: run both methods on each shared pair ----
    for i, (c, s) in enumerate(PAIRS):
        print(f"\n=== Pair {i + 1}/{len(PAIRS)}: {c} + {s} ===")
        content = load_image(cpath(c), short_side=SHORT_SIDE)
        style = load_image(spath(s), short_side=SHORT_SIDE)

        # AdaIN (feed-forward) — averaged timing over several runs.
        _ = adain.stylize(content, style)  # warmup
        t0 = time.perf_counter()
        for _ in range(ADAIN_TIMING_RUNS):
            with Timer(device):
                a_out = adain.stylize(content, style, alpha=1.0)
        adain_sec = (time.perf_counter() - t0) / ADAIN_TIMING_RUNS
        save_image(a_out, os.path.join(OUT, f"adain_{c}__{s}.png"))
        adain_imgs[(c, s)] = a_out
        print(f"  AdaIN:  {adain_sec * 1000:.1f} ms/image")

        # Gatys (optimisation) — capture the full trajectory on the first pair.
        snap_steps = (1, 10, 25, 50, 100, 200, 300) if i == 0 else (GATYS_STEPS,)
        g0 = time.perf_counter()
        g_out, snaps, hist = gatys_nst.run_gatys(
            content, style, device, num_steps=GATYS_STEPS, snapshot_steps=snap_steps
        )
        gatys_sec = time.perf_counter() - g0
        save_image(g_out, os.path.join(OUT, f"gatys_{c}__{s}.png"))
        gatys_imgs[(c, s)] = g_out
        print(f"  Gatys:  {gatys_sec:.1f} s/image ({GATYS_STEPS} steps)")

        results["pairs"].append(
            {
                "content": c,
                "style": s,
                "adain_ms": round(adain_sec * 1000, 2),
                "gatys_s": round(gatys_sec, 2),
                "gatys_final_loss": round(hist[-1][1], 4) if hist else None,
            }
        )

        if i == 0:  # Gatys optimisation-progression montage
            make_progression(content, style, snaps)

    # ---- comparison grid: content | style | Gatys | AdaIN ----
    make_comparison_grid(gatys_imgs, adain_imgs)

    # ---- AdaIN-only demos (fast, showcase flexibility) ----
    make_flexibility_grid(adain)
    make_alpha_tradeoff(adain, "sailboat", "la_muse")
    make_interpolation(adain, "golden_gate", ["la_muse", "brushstrokes"])

    # ---- runtime summary ----
    make_runtime_chart(results)
    with open(os.path.join(OUT, "runtime.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll outputs written to", OUT)
    print(json.dumps(results, indent=2))


def make_progression(content, style, snaps):
    n = len(snaps) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    imshow(axes[0], content, "content (init)")
    for ax, (step, img) in zip(axes[1:], snaps):
        imshow(ax, img, f"step {step}")
    fig.suptitle(
        "Gatys: image emerges via iterative optimisation (start = content image)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gatys_optimization_progression.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def make_comparison_grid(gatys_imgs, adain_imgs):
    rows = len(PAIRS)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = axes[None, :]
    cols = ["Content", "Style", "Gatys (optimisation)", "AdaIN (feed-forward)"]
    for r, (c, s) in enumerate(PAIRS):
        content = load_image(cpath(c), short_side=SHORT_SIDE)
        style = load_image(spath(s), short_side=SHORT_SIDE)
        imshow(axes[r][0], content, cols[0] if r == 0 else None)
        imshow(axes[r][1], style, cols[1] if r == 0 else None)
        imshow(axes[r][2], gatys_imgs[(c, s)], cols[2] if r == 0 else None)
        imshow(axes[r][3], adain_imgs[(c, s)], cols[3] if r == 0 else None)
        axes[r][0].set_ylabel(f"{c}\n+ {s}", fontsize=10, rotation=90,
                              labelpad=10, va="center")
    fig.suptitle("Gatys vs AdaIN on identical content/style pairs", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "comparison_grid.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_flexibility_grid(adain):
    """One frozen AdaIN model stylises many content x style combos — no retraining."""
    nr, nc = len(GRID_CONTENTS), len(GRID_STYLES) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr))
    for r, c in enumerate(GRID_CONTENTS):
        content = load_image(cpath(c), short_side=SHORT_SIDE)
        imshow(axes[r][0], content, "content" if r == 0 else None)
        for k, s in enumerate(GRID_STYLES):
            style = load_image(spath(s), short_side=SHORT_SIDE)
            out = adain.stylize(content, style, alpha=1.0)
            imshow(axes[r][k + 1], out, s if r == 0 else None)
    fig.suptitle("AdaIN is 'arbitrary': one model, any style, single forward pass",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "adain_flexibility_grid.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def make_alpha_tradeoff(adain, c, s):
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    content = load_image(cpath(c), short_side=SHORT_SIDE)
    style = load_image(spath(s), short_side=SHORT_SIDE)
    fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(3 * (len(alphas) + 1), 3.4))
    imshow(axes[0], style, "style")
    for ax, a in zip(axes[1:], alphas):
        out = adain.stylize(content, style, alpha=a)
        imshow(ax, out, f"alpha={a}")
    fig.suptitle(f"AdaIN content-style trade-off ({c} + {s}) — free at test time",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "adain_alpha_tradeoff.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def make_interpolation(adain, c, styles):
    content = load_image(cpath(c), short_side=SHORT_SIDE)
    sty = [load_image(spath(s), short_side=SHORT_SIDE) for s in styles]
    ws = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
    fig, axes = plt.subplots(1, len(ws), figsize=(3 * len(ws), 3.4))
    for ax, w in zip(axes, ws):
        out = adain.stylize_interpolate(content, sty, list(w))
        imshow(ax, out, f"{w[0]:.2f}/{w[1]:.2f}")
    fig.suptitle(
        f"AdaIN style interpolation: {styles[0]} <-> {styles[1]} (feature-space blend)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "adain_style_interpolation.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


def make_runtime_chart(results):
    labels = [f"{p['content']}\n+{p['style']}" for p in results["pairs"]]
    gatys = [p["gatys_s"] for p in results["pairs"]]
    adain = [p["adain_ms"] / 1000.0 for p in results["pairs"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, gatys, 0.4, label="Gatys (optimisation)", color="#c0392b")
    ax.bar(x + 0.2, adain, 0.4, label="AdaIN (feed-forward)", color="#2980b9")
    ax.set_yscale("log")
    ax.set_ylabel("seconds per image (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Runtime per image @ {results['short_side']}px — {results['device']}")
    ax.legend()
    for xi, (g, a) in enumerate(zip(gatys, adain)):
        ax.text(xi - 0.2, g, f"{g:.0f}s", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, a, f"{a * 1000:.0f}ms", ha="center", va="bottom", fontsize=8)
    speedups = [p["gatys_s"] / (p["adain_ms"] / 1000.0) for p in results["pairs"]]
    results["mean_speedup"] = round(float(np.mean(speedups)), 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "runtime_comparison.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
