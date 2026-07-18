"""
Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive
Instance Normalization" (ICCV 2017) — the pipeline implemented from scratch.

Key idea: instead of optimising the image, do a single feed-forward pass:

    content --VGG encoder--> f_c ┐
                                 ├── AdaIN ──> t ──> decoder ──> stylised image
    style   --VGG encoder--> f_s ┘

AdaIN aligns the *channel-wise mean and variance* of the content features to
those of the style features. That one operation transfers style, so any style
image works with no per-style training — hence "arbitrary" and "real-time".

The AdaIN op, the encoder/decoder structure, and the forward pass are written
here. The encoder is a fixed VGG-19 and the decoder is the released pretrained
model from Huang & Belongie / naoto0804 (training the decoder would need
MS-COCO + WikiArt and many GPU-hours; Phase 2 is about *validating* the method,
so we load the published weights and reproduce the pipeline end-to-end).
"""

import torch
import torch.nn as nn

# --- The AdaIN operation (the heart of the method), from scratch ------------


def calc_mean_std(feat, eps=1e-5):
    """Per-sample, per-channel mean and std of a (N, C, H, W) feature map."""
    n, c = feat.shape[:2]
    var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(n, c, 1, 1)
    mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    """Normalise content features, then rescale to the style's mean/std.

    out = style_std * (content - content_mean) / content_std + style_mean
    """
    size = content_feat.size()
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    normalized = (content_feat - c_mean.expand(size)) / c_std.expand(size)
    return normalized * s_std.expand(size) + s_mean.expand(size)


# --- Encoder / decoder architecture (matches the pretrained weights) --------

# Full normalised VGG-19 (reflection padding, a learned 1x1 "normalisation"
# first layer). We only run it up to relu4_1 as the encoder.
def _build_vgg():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),  # relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),  # relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),  # relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # relu4_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    )


# Decoder mirrors the encoder: upsample + conv blocks, 512 -> 3 channels.
def _build_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


class AdaINStyleTransfer(nn.Module):
    """Encoder (VGG to relu4_1) + AdaIN + pretrained decoder."""

    def __init__(self, vgg_weights, decoder_weights, device):
        super().__init__()
        vgg = _build_vgg()
        vgg.load_state_dict(torch.load(vgg_weights, map_location="cpu"))
        # Encoder = layers up to and including relu4_1 (first 31 children).
        self.encoder = nn.Sequential(*list(vgg.children())[:31])

        self.decoder = _build_decoder()
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location="cpu"))

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x.to(self.device))

    @torch.no_grad()
    def stylize(self, content, style, alpha=1.0):
        """Single feed-forward stylisation. alpha in [0,1] trades content<->style."""
        f_c = self.encode(content)
        f_s = self.encode(style)
        t = adaptive_instance_normalization(f_c, f_s)
        t = alpha * t + (1 - alpha) * f_c  # content-style trade-off
        return self.decoder(t).cpu().clamp(0, 1)

    @torch.no_grad()
    def stylize_interpolate(self, content, styles, weights, alpha=1.0):
        """Blend several styles in AdaIN feature space (convex combination)."""
        f_c = self.encode(content)
        weights = [w / sum(weights) for w in weights]
        t = torch.zeros_like(f_c)
        for style, w in zip(styles, weights):
            t = t + w * adaptive_instance_normalization(f_c, self.encode(style))
        t = alpha * t + (1 - alpha) * f_c
        return self.decoder(t).cpu().clamp(0, 1)
