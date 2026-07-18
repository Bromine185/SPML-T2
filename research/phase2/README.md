# Research Track — Phase 2: Implement & Validate Both Style-Transfer Methods

Phase 2 implements the two foundational neural style transfer (NST) methods and
validates them on a shared set of content/style pairs so they can be compared
head-to-head:

1. **Gatys et al. (2015)** — *optimisation-based*: the output image itself is the
   optimisation variable, iteratively refined against a VGG content loss and a
   Gram-matrix style loss.
2. **Huang & Belongie / AdaIN (ICCV 2017)** — *feed-forward*: a single pass
   through an **encoder → AdaIN → decoder** network, where AdaIN aligns the
   content features' channel-wise mean/variance to the style's.

The AdaIN operation, the Gatys losses, and both forward passes are implemented
from scratch. The AdaIN **decoder** uses published pretrained weights — training
it requires MS-COCO + WikiArt and many GPU-hours, which is outside the scope of a
validation exercise (the task states the emphasis is on *validation, not perfect
reproduction*).

## Layout

```
Phase_2/
├── code/
│   ├── gatys_nst.py            # Gatys optimisation-based NST (from scratch)
│   ├── adain_nst.py            # AdaIN op + encoder/decoder pipeline (from scratch)
│   ├── utils.py                # image IO, device, timing helpers
│   ├── run_phase2.py           # runs BOTH methods, writes every figure + runtime.json
│   ├── make_analysis_pdf.py    # compiles analysis.pdf from figures + runtime.json
│   ├── download_assets.py      # fetches pretrained AdaIN weights (not committed)
│   ├── phase2_style_transfer.ipynb   # executed narrative demo
│   ├── requirements.txt
│   └── weights/                # pretrained AdaIN weights (git-ignored; see below)
├── inputs/                     # content/ + style/ images (the paper example images)
├── generated_outputs/          # all result PNGs + runtime.json
├── analysis.pdf                # the write-up (figures + discussion)
└── README.md
```

## How to run

```bash
cd Research/Phase_2/code
pip install -r requirements.txt
python download_assets.py     # fetches vgg_normalised.pth + decoder.pth into weights/
python run_phase2.py          # generates all figures into ../generated_outputs/
python make_analysis_pdf.py   # builds ../analysis.pdf
```

`run_phase2.py` runs on Apple MPS, CUDA, or CPU automatically (see
`utils.get_device`). The Gatys VGG-19 downloads itself via torchvision on first
use and is cached under `~/.cache/torch`.

## Results (Apple M4, MPS, 512 px short side, 300 L-BFGS steps)

| Content + Style | Gatys | AdaIN | Speed-up |
|---|--:|--:|--:|
| chicago + la_muse | 88.1 s | 190 ms | ~460× |
| sailboat + brushstrokes | 65.9 s | 148 ms | ~440× |
| cornell + en_campo_gris | 69.0 s | 183 ms | ~380× |

**~428× faster on average.** The gap is structural, not a constant factor: Gatys
pays a full optimisation loop *per image and per style*, while AdaIN amortises all
learning into one decoder and then pays only a single forward pass.

### Key qualitative findings (see `analysis.pdf`)

- **Content preservation** — Gatys keeps scene structure sharper (it optimises an
  explicit content loss for *this* image); AdaIN's fidelity is fixed by the
  decoder and controlled at test time by the `alpha` knob.
- **Stylisation strength** — Gatys matches Gram matrices at 5 VGG depths (rich
  multi-scale texture); AdaIN matches only mean/variance at one level (strong,
  uniform colour/texture transfer, occasionally smoother).
- **A clean mechanism demo** — for the very dark `en_campo_gris` style, AdaIN
  rescales each channel to the style's dark mean and drives the whole output
  toward black, while Gatys keeps the building legible. This concretely shows
  AdaIN transfers *global feature statistics*.
- **Flexibility** — AdaIN gives arbitrary style, a free content/style `alpha`
  knob, and style interpolation, all at inference; Gatys would have to re-run the
  optimisation for each.

## Note on committed weights

`code/weights/` (the 80 MB `vgg_normalised.pth` and 14 MB `decoder.pth`) is
git-ignored to keep the repo clean; run `download_assets.py` to fetch it. The
generated figures and the executed notebook already contain the visual results,
so the deliverable is fully inspectable without re-running anything.
