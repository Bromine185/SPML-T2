"""
Build analysis.pdf from the generated figures + measured runtimes.

Reads ../generated_outputs/runtime.json so every number quoted in the report
matches the actual run. Run AFTER run_phase2.py.

Usage:  python make_analysis_pdf.py
"""

import json
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "generated_outputs")
PDF = os.path.join(ROOT, "analysis.pdf")
MAX_W = 16 * cm  # usable content width

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceBefore=6,
                    spaceAfter=8, textColor=colors.HexColor("#1a2a4a"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10,
                    spaceAfter=4, textColor=colors.HexColor("#28406b"))
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=ss["BodyText"], fontSize=8.5, leading=11,
                     textColor=colors.HexColor("#555555"), spaceAfter=12)
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontSize=20,
                       textColor=colors.HexColor("#1a2a4a"))
SUB = ParagraphStyle("Sub", parent=ss["BodyText"], fontSize=11,
                     textColor=colors.HexColor("#555555"), spaceAfter=2)


def fig(story, name, caption, max_w=MAX_W):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        story.append(Paragraph(f"[missing figure: {name}]", CAP))
        return
    iw, ih = PILImage.open(path).size
    w = min(max_w, MAX_W)
    h = w * ih / iw
    max_h = 20 * cm
    if h > max_h:
        h = max_h
        w = h * iw / ih
    story.append(Image(path, width=w, height=h))
    story.append(Paragraph(caption, CAP))


def para(story, text):
    story.append(Paragraph(text, BODY))


def load_runtime():
    with open(os.path.join(OUT, "runtime.json")) as f:
        return json.load(f)


def build():
    rt = load_runtime()
    pairs = rt["pairs"]
    mean_g = sum(p["gatys_s"] for p in pairs) / len(pairs)
    mean_a = sum(p["adain_ms"] for p in pairs) / len(pairs)
    speedup = rt.get("mean_speedup", round(mean_g / (mean_a / 1000.0), 1))

    doc = SimpleDocTemplate(PDF, pagesize=A4, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, leftMargin=2.5 * cm,
                            rightMargin=2.5 * cm, title="Phase 2 — NST Analysis")
    s = []

    # --- title ---
    s.append(Paragraph("Neural Style Transfer — Phase 2", TITLE))
    s.append(Paragraph("Implementation &amp; Validation of Gatys et al. and AdaIN", SUB))
    s.append(Paragraph(
        f"Both methods run on identical content/style pairs at {rt['short_side']}px on "
        f"{rt['device']}.", SUB))
    s.append(Spacer(1, 0.3 * cm))

    # --- overview ---
    s.append(Paragraph("1. What this phase does", H1))
    para(s,
         "Phase 2 implements the two foundational neural style transfer (NST) methods "
         "from Phase 1 and validates them on the same images so they can be compared "
         "directly. <b>Gatys et al. (2015)</b> is the original optimisation-based method: "
         "it treats the output image itself as the free variable and iteratively "
         "optimises it. <b>Huang &amp; Belongie's AdaIN (ICCV 2017)</b> replaces that slow "
         "per-image loop with a single feed-forward pass through an encoder&ndash;AdaIN&ndash;"
         "decoder network. The AdaIN operation, the Gatys losses, and both forward passes "
         "are implemented from scratch in <font face='Courier'>gatys_nst.py</font> and "
         "<font face='Courier'>adain_nst.py</font>; the AdaIN decoder uses the published "
         "pretrained weights (training it needs MS-COCO + WikiArt and many GPU-hours, "
         "which is out of scope for a validation exercise).")

    # --- Gatys ---
    s.append(Paragraph("2. Method 1 &mdash; Gatys: optimisation-based NST", H1))
    para(s,
         "A frozen VGG-19 is used purely as a fixed feature extractor. Two losses are "
         "defined on its activations: a <b>content loss</b> (MSE between the deep feature "
         "maps at relu4_2 of the output and the content image) that pins down semantic "
         "layout, and a <b>style loss</b> (MSE between <b>Gram matrices</b> at "
         "relu1_1&hellip;relu5_1). The Gram matrix G = FF<sup>T</sup> holds the "
         "correlations between feature channels; because it sums over all spatial "
         "positions it throws away <i>where</i> things are and keeps only <i>what "
         "textures/colours co-occur</i> &mdash; i.e. style. Starting from the content "
         "image, we minimise <i>content_weight&middot;L_content + style_weight&middot;"
         "L_style</i> over the pixels with L-BFGS. Every iteration is a full forward + "
         "backward pass through VGG, which is why the method is slow: nothing is reused "
         "between images.")
    fig(s, "gatys_optimization_progression.png",
        "Figure 1. The Gatys output is <i>optimised into existence</i>. It begins as the "
        "content image (left) and, step by step, VGG-feature gradients paint the style's "
        "texture onto it while the content loss preserves the scene.")

    # --- AdaIN ---
    s.append(Paragraph("3. Method 2 &mdash; AdaIN: real-time feed-forward NST", H1))
    para(s,
         "AdaIN starts from a simple observation: in instance normalisation, the affine "
         "scale/shift applied after normalising each channel effectively controls style. "
         "So instead of learning per-style parameters, AdaIN <b>copies them from the style "
         "image at run time</b>. Concretely, content and style are both pushed through a "
         "fixed VGG encoder (up to relu4_1); the AdaIN layer normalises each content "
         "feature channel to zero mean/unit variance and then rescales it to the "
         "<b>channel-wise mean and standard deviation of the style features</b>: "
         "AdaIN(x,y) = &sigma;(y)&middot;(x&minus;&mu;(x))/&sigma;(x) + &mu;(y). A decoder "
         "(trained once, on many styles) inverts these aligned features back to an image. "
         "The whole thing is one forward pass &mdash; no optimisation loop &mdash; which is "
         "what makes it real-time and, because the style only enters through its "
         "statistics, applicable to <b>arbitrary</b> styles the decoder never saw.")
    fig(s, "adain_flexibility_grid.png",
        "Figure 2. One frozen AdaIN model stylises every content image with every style "
        "in a single forward pass each &mdash; no per-style training. This 'arbitrary "
        "style' flexibility is impossible with Gatys without re-running the whole "
        "optimisation for each pair.")
    fig(s, "adain_alpha_tradeoff.png",
        "Figure 3. Because AdaIN blends aligned and original content features "
        "(t = &alpha;&middot;AdaIN + (1&minus;&alpha;)&middot;f_content), the "
        "content&harr;style trade-off is a free, continuous knob at test time &mdash; "
        "&alpha;=0 reconstructs the content (decoder pass-through), &alpha;=1 is full "
        "stylisation.")
    fig(s, "adain_style_interpolation.png",
        "Figure 4. Styles can be interpolated by convex-combining their AdaIN targets in "
        "feature space, yielding smooth blends between two styles &mdash; again, for free "
        "at inference.")

    s.append(PageBreak())

    # --- head to head ---
    s.append(Paragraph("4. Head-to-head on identical pairs", H1))
    fig(s, "comparison_grid.png",
        "Figure 5. Same content and style fed to both methods. Gatys keeps edges and "
        "scene structure sharper (explicit content loss at test time); AdaIN transfers "
        "the global colour/texture statistics more aggressively and is more painterly / "
        "abstract, occasionally washing out fine content detail.")

    # --- discussion ---
    s.append(Paragraph("5. Discussion", H1))
    s.append(Paragraph("5.1 Content preservation", H2))
    para(s,
         "Gatys optimises an explicit content loss for <i>this specific image</i> at test "
         "time, so structural fidelity is generally higher and directly tunable via the "
         "content/style weight ratio. AdaIN's content fidelity is fixed by a decoder that "
         "was trained to balance content and style across a whole dataset; at &alpha;=1 it "
         "can over-stylise and abstract away fine detail, but the &alpha; knob (Figure 3) "
         "recovers content cheaply without any re-optimisation. This is vivid in the "
         "<i>cornell + en_campo_gris</i> row of Figure 5: because AdaIN literally rescales "
         "each feature channel to the style's mean, a very dark, low-key style drags the "
         "whole output toward black and buries the building, whereas Gatys &mdash; "
         "balancing an explicit content loss &mdash; keeps the tower legible. It is a "
         "clean demonstration that AdaIN transfers <i>global feature statistics</i>, for "
         "better or worse.")
    s.append(Paragraph("5.2 Stylisation strength", H2))
    para(s,
         "Gatys matches Gram matrices at five VGG depths, so it can reproduce "
         "multi-scale brush texture and fine stroke structure quite faithfully &mdash; at "
         "the cost of sensitivity to loss weights and initialisation. AdaIN matches only "
         "first- and second-order feature statistics (mean/variance) at a single level, "
         "which transfers overall palette and coarse texture very strongly and uniformly, "
         "but is a lower-order match, so intricate style micro-structure is sometimes "
         "smoothed.")
    s.append(Paragraph("5.3 Runtime", H2))
    para(s,
         f"On {rt['device']} at {rt['short_side']}px, Gatys averaged "
         f"<b>{mean_g:.1f} s/image</b> ({rt['gatys_steps']} L-BFGS iterations) while AdaIN "
         f"averaged <b>{mean_a:.0f} ms/image</b> for a single forward pass &mdash; roughly "
         f"a <b>{speedup:.0f}&times; speed-up</b>. Crucially the gap is structural, not "
         "constant-factor: Gatys pays the full optimisation cost <i>per image and per "
         "style</i>, whereas AdaIN amortises all training into one decoder and then pays "
         "only inference.")
    fig(s, "runtime_comparison.png",
        "Figure 6. Per-image runtime (log scale). The optimisation loop is orders of "
        "magnitude slower than a single feed-forward pass.", max_w=13 * cm)

    s.append(Paragraph("5.4 Implementation difficulty", H2))
    para(s,
         "Gatys is easier to <i>implement</i> end-to-end: a pretrained VGG, two losses, "
         "and an optimiser &mdash; no training data or decoder required. Its difficulty is "
         "operational (slow, weight-tuning). AdaIN is trivial at the AdaIN layer itself "
         "(a few lines) but its decoder must be <i>trained once</i> on a large "
         "content/style corpus; we side-stepped that by loading published weights, which "
         "is the standard way to validate the method.")

    # --- summary table ---
    s.append(Paragraph("6. Summary: Gatys vs AdaIN", H1))
    data = [
        ["", "Gatys (2015)", "AdaIN (2017)"],
        ["Regime", "Per-image optimisation", "Single feed-forward pass"],
        ["Style encoding", "Gram matrices (5 layers)", "Channel mean/var (relu4_1)"],
        ["Speed / image", f"{mean_g:.0f} s", f"{mean_a:.0f} ms"],
        ["Arbitrary style", "Yes (re-optimise each time)", "Yes (no retraining)"],
        ["Content/style knob", "Retune weights + rerun", "Free α at test time"],
        ["Style interpolation", "No (naturally)", "Yes (feature-space blend)"],
        ["Needs trained decoder", "No", "Yes (pretrained here)"],
        ["Content fidelity", "Higher, tunable", "Good; α-dependent"],
        ["Best when", "Max quality, few images", "Interactive / many images"],
    ]
    t = Table(data, colWidths=[4.2 * cm, 5.9 * cm, 5.9 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28406b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3ccdb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    s.append(t)
    s.append(Spacer(1, 0.3 * cm))

    # --- reflection ---
    s.append(Paragraph("7. What I learned", H1))
    para(s,
         "Reproducing both methods made the conceptual jump between them concrete. Gatys "
         "shows that a discriminative CNN's features already separate content (deep "
         "activations) from style (feature correlations) &mdash; style transfer is then "
         "just optimisation against those two targets. AdaIN shows that this expensive "
         "optimisation can be <i>amortised</i>: aligning feature statistics is a cheap, "
         "closed-form operation, and a decoder trained once turns it into a real-time, "
         "arbitrary-style method. The trade-off is the recurring one in deep learning "
         "&mdash; per-instance optimisation (slow, flexible, high quality) versus a learned "
         "feed-forward approximation (fast, amortised, slightly lower fidelity). AdaIN "
         "does not strictly beat Gatys on quality; it makes a different, and for most "
         "applications far more practical, trade-off.")

    # --- refs ---
    s.append(Paragraph("References", H1))
    for r in [
        "L. A. Gatys, A. S. Ecker, M. Bethge. <i>A Neural Algorithm of Artistic Style</i>. "
        "arXiv:1508.06576, 2015.",
        "X. Huang, S. Belongie. <i>Arbitrary Style Transfer in Real-time with Adaptive "
        "Instance Normalization</i>. ICCV 2017.",
        "Reference AdaIN weights &amp; architecture: naoto0804/pytorch-AdaIN (GitHub).",
        "K. Simonyan, A. Zisserman. <i>Very Deep Convolutional Networks (VGG)</i>. "
        "ICLR 2015.",
    ]:
        s.append(Paragraph("&bull; " + r, BODY))

    doc.build(s)
    print("Wrote", PDF)


if __name__ == "__main__":
    build()
