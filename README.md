# Task 2 — Base ML Submission

Custom deep-learning architectures implemented **from scratch** (no pre-built
ResNet / LSTM / GRU / RNN / Transformer modules) in **PyTorch**.

## Contents

```
Task2_Submission/
└── Base_ML/
    ├── Level_1_ResNet/
    │   ├── code/level1_resnet_cifar10.ipynb
    │   ├── model_weights/      # custom_resnet_cifar10.pth (written on run)
    │   └── outputs/            # training_curves.png, confusion_matrix.png, metrics.json
    ├── Level_2_LSTM/
    │   ├── code/level2_lstm_jena.ipynb
    │   ├── model_weights/      # custom_lstm_jena.pth
    │   └── outputs/            # loss_curve.png, forecast_examples.png, metrics.json
    └── Level_3_Transformer/
        ├── code/level3_transformer_jena.ipynb
        ├── model_weights/      # transformer_jena.pth, lstm_compare_jena.pth
        └── outputs/            # comparison_pred_vs_actual.png, training_curves.png, metrics_comparison.json
```

## Levels

| Level | Architecture (from scratch) | Dataset | Task |
|---|---|---|---|
| 1 | Custom ResNet — manual residual blocks, skip connections, projection shortcuts, GAP | CIFAR-10 | 10-class image classification |
| 2 | Custom LSTM — hand-written input/forget/output gates, candidate & cell updates | Jena Climate | 72 h → 12 h temperature forecast |
| 3 | Encoder-only Transformer — manual sinusoidal PE, multi-head self-attention, encoder blocks | Jena Climate | 720 h → 24 h forecast, compared vs the Level-2 LSTM |

## How to run

Each notebook is self-contained and runs top-to-bottom on a **GPU-enabled**
Python environment with `torch`, `torchvision`, `numpy`, `pandas`,
`matplotlib`, `scikit-learn`.

- Datasets auto-download to `~/datasets` (override with the `DATA_DIR` env var).
  CIFAR-10 via `torchvision`; Jena Climate via the Keras mirror.
- Trained weights are written to each level's `model_weights/`; all figures and
  metric JSONs to each level's `outputs/`.
- Tune `EPOCHS` (and `BATCH` in Level 3) for your hardware. Defaults already
  produce strong results; increase epochs for higher CIFAR-10 accuracy.

## Submission notes (per updated instructions)

- **Level 1:** training/validation curves, final accuracy, classification
  report, confusion matrix, and saved weights — as specified in the PDF.
- **Levels 2 & 3:** a single predicted-vs-actual temperature comparison plot for
  both models on the same test window, with evaluation by **Huber loss, MAE and
  MSE** (Level 3 notebook, section 7). Each level also saves its own weights and
  loss curves.

## Constraints satisfied

- **Level 1:** 3 residual stages (limit 2–4), 2 blocks/stage (limit 1–3),
  increasing width 64→128→256, Global Average Pooling before the classifier,
  manual dimension-mismatch handling via strided 1×1 projection shortcuts.
- **Level 2:** all gates implemented by hand, hidden dim 96 (range 32–256),
  2 stacked layers (max 2); no built-in recurrent modules.
- **Level 3:** encoder-only, `d_model=128` (range 64–128), 4 heads (2 or 4),
  3 encoder layers (max 3); manual positional encoding and attention.
