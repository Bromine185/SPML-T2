"""
Gatys et al., "A Neural Algorithm of Artistic Style" (2015) — from scratch.

Optimisation-based neural style transfer:
  * a frozen VGG-19 is used purely as a fixed feature extractor;
  * CONTENT is matched by an MSE loss on a deep feature map (relu4_2);
  * STYLE  is matched by an MSE loss on Gram matrices (feature correlations)
    at relu1_1, relu2_1, relu3_1, relu4_1, relu5_1;
  * the *image itself* is the optimisation variable — we start from the content
    image and run L-BFGS, so every step is a forward+backward pass through VGG.

This is the slow, iterative regime: one optimisation loop per (content, style)
pair. Nothing is learned/amortised, which is exactly the limitation AdaIN fixes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

# VGG-19 `features` indices for the layers Gatys uses (post-ReLU activations).
STYLE_LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}
CONTENT_LAYER = 22  # relu4_2

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gram_matrix(feat):
    """Normalised Gram matrix of a (1, C, H, W) feature map.

    G[i, j] = <channel_i, channel_j> captures which features co-activate,
    discarding spatial layout — this is precisely what encodes 'style'/texture.
    Dividing by the number of elements keeps the loss scale-invariant to size.
    """
    b, c, h, w = feat.size()
    f = feat.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


class VGGFeatures(nn.Module):
    """Frozen VGG-19 feature extractor returning the chosen style/content maps.

    Max-pooling is swapped for average-pooling, as recommended in the paper for
    smoother gradients and more visually pleasing results.
    """

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        layers = []
        for m in vgg:
            if isinstance(m, nn.MaxPool2d):
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(m)
        self.model = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def forward(self, x):
        x = (x - self.mean) / self.std
        style_feats, content_feat = {}, None
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in STYLE_LAYERS:
                style_feats[i] = x
            if i == CONTENT_LAYER:
                content_feat = x
            if i >= CONTENT_LAYER and len(style_feats) == len(STYLE_LAYERS):
                break
        return style_feats, content_feat


def run_gatys(
    content,
    style,
    device,
    num_steps=300,
    style_weight=1e6,
    content_weight=1.0,
    snapshot_steps=(1, 25, 50, 100, 200, 300),
    log_every=50,
):
    """Run optimisation-based style transfer.

    Returns (final_image, snapshots, history) where snapshots is a list of
    (step, image_tensor_cpu) capturing the optimisation trajectory and history
    is a list of (step, total, content, style) loss values.
    """
    extractor = VGGFeatures().to(device)
    content = content.to(device)
    style = style.to(device)

    # Fixed targets (no grad): style Grams from the style image, content map
    # from the content image.
    with torch.no_grad():
        style_feats, _ = extractor(style)
        style_targets = {i: gram_matrix(f) for i, f in style_feats.items()}
        _, content_target = extractor(content)

    # The optimisation variable IS the image; initialise from the content image.
    # .contiguous() is required so L-BFGS's internal grad.view(-1) works on MPS.
    img = content.detach().clone().contiguous().requires_grad_(True)
    optimizer = torch.optim.LBFGS([img], max_iter=20, line_search_fn="strong_wolfe")

    step = [0]
    snapshots, history = [], []
    pending_snaps = sorted(snapshot_steps)
    n_style = len(style_targets)

    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            img.clamp_(0, 1)  # keep pixels valid throughout
        style_feats, content_feat = extractor(img)

        c_loss = F.mse_loss(content_feat, content_target)
        s_loss = sum(
            F.mse_loss(gram_matrix(style_feats[i]), style_targets[i])
            for i in style_targets
        ) / n_style

        total = content_weight * c_loss + style_weight * s_loss
        total.backward()

        step[0] += 1
        # Threshold-based capture: L-BFGS's max_iter makes step[0] jump by up to
        # 20 per optimizer.step(), so we snapshot the first eval at/after each
        # requested milestone rather than requiring an exact match.
        while pending_snaps and step[0] >= pending_snaps[0]:
            milestone = pending_snaps.pop(0)
            snapshots.append((milestone, img.detach().cpu().clamp(0, 1).clone()))
        if step[0] % log_every == 0 or step[0] == 1:
            history.append(
                (step[0], total.item(), c_loss.item(), (style_weight * s_loss).item())
            )
            print(
                f"  [gatys] step {step[0]:4d}  total={total.item():.2f}  "
                f"content={c_loss.item():.4f}  style={(style_weight * s_loss).item():.2f}"
            )
        return total

    while step[0] < num_steps:
        optimizer.step(closure)

    with torch.no_grad():
        img.clamp_(0, 1)
    return img.detach().cpu(), snapshots, history
