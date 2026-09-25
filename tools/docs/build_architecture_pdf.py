"""Rebuild the scientific architecture note from executable default configuration.

Run with the project's optional docs dependencies installed.
All figures and equations are vector paths; no model training is performed.
"""

# ruff: noqa: E402
import io
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tram-matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from svglib.svglib import svg2rlg

from tram_forecast.config import FIELDS, Config
from tram_forecast.model import ForecastNetwork

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/tram_cnn_scientific_architecture.pdf"
W, H = 1000, 740
INK = "#17202B"
BLUE = "#DCEAF4"
PURPLE = "#E9E2F1"
GREEN = "#E2EDE3"
ORANGE = "#F5E6D3"
GREY = "#F1F2F3"
cfg = Config()
model = ForecastNetwork([x + 3 for x in cfg.category_caps], 1, cfg)
assert model.parameter_report()["total"] == 199268
REV = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
).strip()
plt.rcParams.update(
    {"mathtext.fontset": "stix", "font.family": "STIXGeneral", "svg.fonttype": "path"}
)
OUT.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
c.setTitle("Multiscale tram ridership CNN: executable architecture")
c.setAuthor("Tram forecasting project")


def text(x, y, s, size=10, font="Times-Roman", color=INK, align="left"):
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    getattr(
        c,
        {
            "left": "drawString",
            "center": "drawCentredString",
            "right": "drawRightString",
        }[align],
    )(x, y, s)


def para(x, y, w, s, size=10, leading=13):
    style = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=size,
        leading=leading,
        textColor=HexColor(INK),
    )
    p = Paragraph(s, style)
    _, height = p.wrap(w, 1000)
    p.drawOn(c, x, y - height)
    return y - height


def eq(x, y, s, size=15, maxwidth=900):
    fig = plt.figure(figsize=(1, 0.3))
    fig.text(0, 0, s, fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    buf.seek(0)
    drawing = svg2rlg(buf)
    if drawing.width > maxwidth:
        scale = maxwidth / drawing.width
        drawing.scale(scale, scale)
        drawing.width *= scale
        drawing.height *= scale
    renderPDF.draw(drawing, c, x, y)


def line(points, color=INK, dash=False, width=0.8, arrow=False):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    c.setDash(3, 2) if dash else c.setDash()
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p)
    c.setDash()
    if arrow:
        x, y = points[-1]
        a, b = points[-2]
        angle = math.atan2(y - b, x - a)
        length = 5
        p = c.beginPath()
        p.moveTo(x, y)
        for delta in (2.65, -2.65):
            p.lineTo(
                x + length * math.cos(angle + delta),
                y + length * math.sin(angle + delta),
            )
        p.close()
        c.setFillColor(HexColor(color))
        c.drawPath(p, fill=1, stroke=0)


def box(x, y, w, h, label, sub=None, fill=BLUE, size=11):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(INK))
    c.setLineWidth(0.8)
    c.rect(x, y, w, h, fill=1)
    text(
        x + w / 2,
        y + h / 2 + (3 if sub else -3),
        label,
        size,
        "Times-Bold",
        align="center",
    )
    if sub:
        text(x + w / 2, y + h / 2 - 10, sub, 8.5, "Helvetica", align="center")


def frame(x, y, w, h, label, fill=None):
    c.setStrokeColor(HexColor("#727C88"))
    c.setLineWidth(0.6)
    if fill:
        c.setFillColor(HexColor(fill))
    c.rect(x, y, w, h, fill=bool(fill))
    text(x + 10, y + h - 16, label, 11, "Times-Bold")


def header(n, title, subtitle):
    text(
        36,
        709,
        "TRAM RIDERSHIP FORECASTING | TECHNICAL ARCHITECTURE",
        9,
        "Helvetica-Bold",
    )
    text(964, 709, f"IMPLEMENTATION NOTE / {n:02d}", 9, "Helvetica", align="right")
    line([(36, 701), (964, 701)], color="#7A8590", width=0.6)
    text(36, 675, title, 23, "Times-Bold")
    text(36, 657, subtitle, 10)
    line([(36, 39), (964, 39)], color="#7A8590", width=0.6)
    text(
        36,
        25,
        f"Default implementation at {REV} | 25 September 2026 | No trained-performance claims",
        8,
        "Helvetica",
    )
    text(964, 25, f"{n} / 5", 9, "Helvetica", align="right")


def table(x, y, width, columns, rows, weights=None, rowh=28):
    weights = weights or [1] * len(columns)
    unit = width / sum(weights)
    xs = [x]
    for weight in weights:
        xs.append(xs[-1] + weight * unit)
    c.setFillColor(HexColor(GREY))
    c.rect(x, y - rowh, width, rowh, fill=1, stroke=0)
    for i, col in enumerate(columns):
        text(xs[i] + 7, y - rowh + 10, col, 10, "Times-Bold")
    for r, row in enumerate(rows):
        baseline = y - rowh * (r + 2) + 10
        if r % 2 == 1:
            c.setFillColor(HexColor("#F9FAFB"))
            c.rect(x, baseline - 10, width, rowh, fill=1, stroke=0)
        for i, value in enumerate(row):
            text(xs[i] + 7, baseline, str(value), 9.5)
        line(
            [(x, baseline - 10), (x + width, baseline - 10)],
            color="#CDD2D8",
            width=0.35,
        )
    return y - rowh * (len(rows) + 1)


# PAGE 1 -----------------------------------------------------------------
header(
    1,
    "A. Complete computation graph",
    "One route, one fixed 504-hour observed history, one requested future day; B denotes sample batch; M denotes the current chunk of nonempty hours.",
)
# Main bottom-up graph, modeled on grouped scientific architecture figures.
frame(36, 91, 605, 550, "Full model")
box(66, 110, 186, 29, "Categorical events", "per hour: [N, 5]", ORANGE)
box(66, 151, 186, 29, "Five embedding tables", "concatenate: [M, 28, Npad]", ORANGE)
box(66, 192, 186, 29, "E1: multiscale event block", "28 -> 32; mask after GELU")
box(66, 233, 186, 29, "E2: multiscale event block", "32 -> 32; mask after GELU")
box(66, 274, 186, 29, "Masked maximum over events", "scatter: Z [B, 32, 504]", GREEN)
box(315, 274, 178, 29, "Observed boarding history", "Y [B, 1, 504]", ORANGE)
box(66, 315, 186, 29, "Concat [Z, K]", "raw input [B, 38, 504]", GREY)
box(315, 315, 178, 29, "Concat [Y / s, K]", "count input [B, 7, 504]", GREY)
box(520, 274, 104, 29, "Calendar K", "[B, 6, 504]", ORANGE, size=10)
box(66, 356, 186, 29, "A: hourly multiscale block", "[B, 32, 504]")
box(315, 356, 178, 29, "B: hourly multiscale block", "[B, 32, 504]")
box(181, 397, 230, 26, "Channel concatenation: [B, 64, 504]", fill=GREY, size=10)
box(
    181,
    435,
    230,
    26,
    "S1 (64 -> 64) + max pool (2, 2)",
    "[B, 64, 252]",
    PURPLE,
    size=10,
)
box(
    181,
    473,
    230,
    26,
    "S2 (64 -> 32) + max pool (2, 2)",
    "[B, 32, 126]",
    PURPLE,
    size=10,
)
box(
    181, 511, 230, 26, "S3 (32 -> 16) + max pool (3, 3)", "[B, 16, 42]", PURPLE, size=10
)
box(181, 549, 230, 23, "Flatten: [B, 672]", fill=GREY, size=10)
box(
    181,
    584,
    230,
    29,
    "Concat request q; shared MLP",
    "685 -> 128 -> 24; s x softplus",
    GREEN,
    size=10,
)
text(
    296,
    623,
    "24 nonnegative hourly boarding predictions",
    11,
    "Times-Bold",
    align="center",
)
for lower, upper in [
    (139, 151),
    (180, 192),
    (221, 233),
    (262, 274),
    (303, 315),
    (344, 356),
]:
    line([(159, lower), (159, upper)], arrow=True)
line([(404, 303), (404, 315)], arrow=True)
line([(404, 344), (404, 356)], arrow=True)
line([(159, 385), (159, 390), (251, 390), (251, 397)], arrow=True)
line([(404, 385), (404, 390), (341, 390), (341, 397)], arrow=True)
for lower, upper in [
    (423, 435),
    (461, 473),
    (499, 511),
    (537, 549),
    (572, 584),
    (613, 620),
]:
    line([(296, lower), (296, upper)], arrow=True)
line([(572, 303), (572, 308), (410, 308)], color="#A36420")
line([(398, 308), (244, 308), (244, 315)], color="#A36420", arrow=True)
c.setStrokeColor(HexColor("#A36420"))
c.arc(398, 303, 410, 313, 0, 180)
line([(572, 308), (480, 308), (480, 315)], color="#A36420", arrow=True)
box(
    449,
    486,
    173,
    50,
    "Request conditioning q",
    "route (8) + calendar (4) + lead (1)",
    ORANGE,
    size=10,
)
line([(535, 536), (535, 598), (411, 598)], color="#A36420", arrow=True)
text(449, 467, "No requested target enters the encoder.", 8.5, "Times-Italic")
# Right hand operator definitions.
frame(662, 428, 302, 213, "Inset A1. Universal multiscale operator")
box(764, 454, 98, 22, "Input X", fill=GREY, size=10)
for x, label, sub in [
    (680, "Conv + GELU", "path 1"),
    (774, "Conv + GELU", "path 2"),
    (868, "Conv + GELU", "path 3"),
]:
    box(x, 505, 78, 36, label, sub, size=9)
    line([(813, 476), (813, 488), (x + 39, 488), (x + 39, 505)], arrow=True)
    line([(x + 39, 541), (x + 39, 557), (813, 557), (813, 572)], arrow=True)
box(718, 572, 190, 25, "Concatenate output channels", fill=GREY, size=10)
text(813, 610, "Independent weights; same temporal length", 9, align="center")
frame(662, 259, 302, 153, "Inset A2. Head and conditioning")
eq(675, 365, r"$q=[e_r;\ \kappa(d);\ (h-1)/60]\in\mathbb{R}^{13}$", 14, 275)
eq(
    675,
    324,
    r"$\hat{y}=s\,\mathrm{softplus}(W_2\,D_{0.1}(\mathrm{GELU}(W_1[v;q]+b_1))+b_2)$",
    12,
    275,
)
para(
    675,
    308,
    274,
    "W1: 685 to 128; W2: 128 to 24. Dropout is active only during training. Output index identifies forecast hour 0-23.",
    10,
    13,
)
frame(662, 91, 302, 152, "Notation and reading convention")
para(
    675,
    220,
    274,
    "<b>M</b>: nonempty hours in the current event chunk.<br/><b>Npad</b>: padded event sequence length.<br/><b>K</b>: historical calendar, including empty hours.<br/><b>s</b>: fitting-period mean count, lower-bounded by 1.<br/><b>v</b>: compressed history vector, 672 features.<br/><b>[ ; ]</b>: channel/feature concatenation.<br/><b>Arrows</b>: bottom-to-top data flow; amber denotes calendar/request context.",
    10,
    15,
)
para(
    36,
    79,
    928,
    "<b>Figure 1.</b> Executable full-model graph. Every blue/purple convolution block expands into three parallel kernel/dilation paths. The raw and boarding branches share hour coordinates but have independent weights. The event encoder is reused for each hour; all routes share the full network.",
    10,
    12,
)
c.showPage()

# PAGE 2 -----------------------------------------------------------------
header(
    2,
    "B. Exact multiscale operators and parameterization",
    "Triples below mean (output channels, kernel length k, dilation d). Concatenation order is the listed path order.",
)
frame(
    36,
    433,
    928,
    208,
    "Figure 2. Expanded E1 example; the same construction is used at every convolutional stage",
)
box(60, 506, 118, 42, "Embedded events", "[M, 28, Npad]", ORANGE)
for y, label in [
    (583, "Conv(28, 12, k=3, d=1, p=1)"),
    (523, "Conv(28, 12, k=5, d=1, p=2)"),
    (463, "Conv(28, 8, k=3, d=2, p=2)"),
]:
    box(235, y, 250, 27, label, size=10)
    box(522, y, 105, 27, "GELU", size=10)
    line([(178, 527), (207, 527), (207, y + 13), (235, y + 13)], arrow=True)
    line([(485, y + 13), (522, y + 13)], arrow=True)
    line([(627, y + 13), (658, y + 13), (658, 527), (701, 527)], arrow=True)
box(701, 506, 114, 42, "Concat channels", "[M, 32, Npad]", GREY)
box(848, 506, 93, 42, "Apply mask", "mask: [M,1,Npad]", GREEN, size=10)
line([(815, 527), (848, 527)], arrow=True)
text(
    54,
    447,
    "Mask invalid inputs before E1; repeat the mask after every event block. Hourly/shared blocks have no event padding mask.",
    9,
)
rows = []
inputs = [28, 32, 38, 7, 64, 64, 32]
names = ["E1", "E2", "A (raw)", "B (counts)", "S1", "S2", "S3"]
lengths = ["Npad", "Npad", "504", "504", "504 -> 252", "252 -> 126", "126 -> 42"]
paths = [
    cfg.event1,
    cfg.event2,
    cfg.hourly,
    cfg.hourly,
    cfg.shared1,
    cfg.shared2,
    cfg.shared3,
]
counts = []
for name, ci, ps, L in zip(names, inputs, paths, lengths):
    number = sum(co * (ci * k + 1) for co, k, _ in ps)
    counts.append(number)
    rows.append(
        (name, ci, " / ".join(f"({co},{k},{d})" for co, k, d in ps), L, f"{number:,}")
    )
table(
    36,
    408,
    928,
    [
        "Block",
        "Cin",
        "Parallel paths (Cout, k, d)",
        "Length before -> after pooling",
        "Parameters",
    ],
    rows,
    [1, 0.6, 3.4, 2.5, 1.1],
    rowh=27,
)
eq(
    48,
    135,
    r"$\operatorname{MS}(X)=\operatorname{Concat}_{i=1}^{3}\left[\operatorname{GELU}\!\left(\operatorname{Conv1d}_{k_i,d_i,p_i}(X)\right)\right],\qquad p_i=\frac{d_i(k_i-1)}{2}$",
    17,
    900,
)
para(
    48,
    121,
    902,
    "All convolutions: stride 1, trainable bias, odd kernel, symmetric zero padding. No trainable normalization, residual projection or attention. Shared max pools use padding 0, dilation 1 and ceil_mode=False. E2 is followed by max over events, not by temporal downsampling.",
    10.5,
    14,
)
text(
    48,
    59,
    "Source: Config paths; MultiscaleConv1d.forward; EventEncoder.features; ForecastNetwork.shared.",
    9,
    "Times-Italic",
)
c.showPage()

# PAGE 3 -----------------------------------------------------------------
header(
    3,
    "C. Temporal support, alignment and compression",
    "Dilated support is sparse. Receptive-field span, output spacing and the head's full-history coverage are different quantities.",
)
frame(36, 280, 529, 361, "Figure 3. Hourly branch filters on a common scale")
left, right = 83, 525


def mapping(v):
    return left + (v + 168) / 336 * (right - left)


for y, k, d, label in [
    (555, 3, 1, "Local: k=3, d=1"),
    (475, 5, 24, "Daily offsets: k=5, d=24"),
    (395, 3, 168, "Weekly offsets: k=3, d=168"),
]:
    text(52, y + 30, label, 11, "Times-Bold")
    line([(left, y), (right, y)], color="#77818B")
    for tap in range(-(k // 2), k // 2 + 1):
        xx = mapping(tap * d)
        c.setFillColor(HexColor("#295575"))
        c.circle(xx, y, 3.2, fill=1, stroke=0)
    text(52, y - 20, f"Span = {1 + (k - 1) * d} hourly positions", 9)
for v in (-168, -48, -24, 0, 24, 48, 168):
    xx = mapping(v)
    line([(xx, 365), (xx, 359)], color="#7A8590")
    text(xx, 345, str(v), 8, "Helvetica", align="center")
text(304, 326, "Offset from output position (hours)", 10, align="center")
# local inset prevents tiny d1 taps from being mistaken for one point
frame(365, 567, 177, 53, "Local detail (separate scale)")
line([(387, 584), (521, 584)])
for x, lab in [(389, "-1"), (454, "0"), (519, "+1")]:
    c.setFillColor(HexColor("#295575"))
    c.circle(x, 584, 2.7, fill=1, stroke=0)
    text(x, 573, lab, 8, align="center")
frame(585, 280, 379, 361, "Figure 4. Propagated maximum nominal span")
eq(600, 587, r"$R_{\ell+1}=R_\ell+(k-1)dJ_\ell$", 17, 340)
eq(600, 553, r"$J_{\ell+1}=s_\ell J_\ell$", 17, 340)
table(
    599,
    531,
    351,
    ["Output", "Length", "J (h)", "R (h)"],
    [
        ("A / B", 504, 1, 337),
        ("S1 + pool", 252, 2, 362),
        ("S2 + pool", 126, 4, 412),
        ("S3 + pool", 42, 12, 468),
    ],
    [1.6, 1, 1, 1],
    rowh=29,
)
para(
    601,
    370,
    345,
    "R follows the widest support path at each block, including pooling. J is the separation between adjacent outputs on the original hourly grid. Spans count positions, not elapsed endpoint differences. Padding supplies no observed data.",
    10,
    13,
)
frame(36, 91, 928, 168, "Alignment and boundary implications")
para(
    53,
    232,
    433,
    "<b>Before fusion.</b> Both branches retain 504 positions. Position t refers to the same route, date and hour. Calendar K is appended after hourly event pooling, so an empty event hour still has a calendar representation.<br/><br/><b>At boundaries.</b> For k=3, d=168, only positions 168 through 335 (168 of 504) have all three taps within observed history.",
    11,
    15,
)
para(
    520,
    232,
    425,
    "<b>After compression.</b> 16 channels x 42 positions flatten to 672 features. The dense head receives the entire compressed sequence; its input coverage is therefore wider than any single local convolutional output.<br/><br/><b>Leakage.</b> Symmetric kernels may read later positions within the observed window. They never read at or after the observation cutoff.",
    11,
    15,
)
text(
    48,
    59,
    "All support calculations refer to default kernels and pooling. The 468-hour span does not imply dense sampling of every intervening hour.",
    9,
    "Times-Italic",
)
c.showPage()

# PAGE 4 -----------------------------------------------------------------
header(
    4,
    "D. Variable event cardinality and exact tiled execution",
    "The event axis is an ordered transaction index within one hour. Event dilation is not elapsed time.",
)
frame(36, 380, 928, 261, "Figure 5. Hour-local masking and exact core/halo tiling")
eq(
    50,
    583,
    r"$Z_{b,j,t}=\max_{0\leq n<N_{b,t}}\Phi_\theta(E_{b,t})_{j,n},\qquad Z_{b,:,t}=0\ \mathrm{if}\ N_{b,t}=0$",
    18,
    900,
)
text(
    53,
    560,
    "E1 radius = 2 events; E2 radius = 4 events; combined maximum radius = 6 events.",
    11,
)
# coordinate bar
x0 = 90
unit = 22
y = 495
for i in range(36):
    fill = BLUE if 6 <= i < 30 else ORANGE
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor("#777777"))
    c.rect(x0 + i * unit, y, unit, 25, fill=1, stroke=1)
text(156, 530, "left halo: 6 events", 10, align="center")
text(486, 530, "retained core (24 events shown schematically)", 10, align="center")
text(816, 530, "right halo: 6 events", 10, align="center")
line([(222, 485), (750, 485)], width=1.5)
line([(222, 481), (222, 489)])
line([(750, 481), (750, 489)])
text(
    486,
    470,
    "Encode the extended tile; retain only core outputs; take channelwise maximum.",
    10,
    align="center",
)
para(
    53,
    450,
    439,
    "At internal tile edges the halo contains actual neighboring events. At true hour boundaries normal zero padding applies. No convolution crosses an hour boundary. Channelwise maxima from successive cores are combined in chronological order.",
    10.5,
    14,
)
para(
    525,
    450,
    420,
    "Invalid activations are zeroed before/after event convolutions. Only the final masked maximum uses negative infinity. Empty hours bypass reduction. A strict comparison retains the earliest event on exact ties, matching full-sequence max gradients.",
    10.5,
    14,
)
text(36, 354, "Embedding schema (trainable categorical tables)", 13, "Times-Bold")
rows = [
    (f, cap, dim, cap + 3)
    for f, cap, dim in zip(FIELDS, cfg.category_caps, cfg.embedding_dims)
]
table(
    36,
    337,
    505,
    ["Field", "Retained cap", "Width", "Max rows"],
    rows,
    [2.5, 1.2, 0.8, 1],
    rowh=27,
)
para(
    47,
    161,
    483,
    "Reserved rows: PAD=0, MISSING=1, UNK=2. Keep training categories with frequency at least 5, capped by descending frequency with lexical tie-breaking. Rare and unseen nonmissing categories share UNK. Concatenated event width: 28.",
    10,
    13,
)
frame(565, 108, 399, 247, "Memory execution contract")
para(
    580,
    329,
    367,
    "<b>Hour packing.</b> Sort by length, break ties by original hour index; at most 32 hours and 32,768 padded positions per chunk. Scatter pooled vectors back to original coordinates.<br/><br/><b>Oversized hours.</b> The default core budget is 32,768 - 2 x 6 = 32,756 events. Preserve every event; overlap only the required halo.<br/><br/><b>Backward computation.</b> Non-reentrant checkpointing recomputes embeddings, E1, E2 and pooling. Compact IDs and pooled vectors remain live; memory still depends on batch event volume.<br/><br/><b>Inference.</b> Disable checkpointing under no_grad; reuse one encoded history per route/cutoff for all requested days.",
    10.5,
    14,
)
text(
    48,
    59,
    "Tiled/untiled output and parameter-gradient equivalence is tested on synthetic inputs, including ties. Real-data/GPU memory has not been measured.",
    9,
    "Times-Italic",
)
c.showPage()

# PAGE 5 -----------------------------------------------------------------
header(
    5,
    "E. Forecast protocol, objective and reproducibility",
    "The architecture is implemented; data-scale runtime, trained forecast accuracy and the benefit of the event branch remain empirical questions.",
)
frame(36, 422, 570, 219, "Figure 6. Direct requests from a frozen observation cutoff")
for y, hist, target, cut in [
    (566, "Aug 11-31, 2025", "Sep 1-Oct 31, 2025", "c = Sep 1"),
    (493, "Oct 11-31, 2025", "Nov 1-Dec 31, 2025", "c = Nov 1"),
]:
    box(56, y, 177, 31, hist, "21 observed days", BLUE, size=10)
    box(
        254,
        y,
        327,
        31,
        target,
        "61 independent requested days; shared history",
        ORANGE,
        size=10,
    )
    line([(243, y - 7), (243, y + 40)], width=1.1)
    text(243, y - 20, cut, 9, align="center")
    line([(233, y + 15), (254, y + 15)], arrow=True)
text(
    55, 442, "Lead h = 1 requests cutoff day; requested date d = c + (h - 1) days.", 10
)
frame(626, 422, 338, 219, "Exact parameter budget")
table(
    638,
    616,
    314,
    ["Component", "Parameters"],
    [
        ("E1 + E2", "7,264"),
        ("A + B", "5,464"),
        ("S1 + S2 + S3", "25,072"),
        ("Dense head", "90,904"),
        ("Nonembedding total", "128,704"),
        ("Maximum embeddings", "70,564"),
    ],
    [2.3, 1.2],
    rowh=22,
)
text(650, 440, "Maximum full model: 199,268", 12, "Times-Bold")
text(36, 393, "Fitting objective and calendar representation", 13, "Times-Bold")
eq(
    45,
    333,
    r"$\mathcal{L}=\frac{1}{24B}\sum_{b=1}^{B}\sum_{u=0}^{23}|\hat{y}_{b,u}-y_{b,u}|,\qquad \mathrm{WAPE}=\frac{\sum|\hat{y}-y|}{\sum y}$",
    18,
    565,
)
para(
    48,
    329,
    552,
    "Counts are scaled by a single training-grid mean s >= 1 before the boarding encoder; outputs are multiplied by s. MAE is evaluated in original count units. WAPE is aggregated from sums, never averaged across route/batch ratios. A zero denominator yields null plus its reason.",
    10.5,
    14,
)
eq(45, 250, r"$K_t=[\sin a_h,\cos a_h,\sin a_w,\cos a_w,\sin a_m,\cos a_m]$", 16, 565)
eq(45, 216, r"$a_h=2\pi h_t/24,\quad a_w=2\pi w_t/7,\quad a_m=2\pi(m_t-1)/31$", 16, 565)
para(
    48,
    202,
    552,
    "Weekday uses Monday=0. Requested calendar contains weekday/day-of-month pairs only. The fixed day-of-month period does not encode annual seasonality. Route embedding: 10 rows x 8 dimensions, including reserved row 0; neural inference uses the nine supported route IDs.",
    10.5,
    14,
)
frame(626, 160, 338, 245, "Training and implementation evidence")
para(
    640,
    382,
    310,
    "<b>Defaults.</b> AdamW, lr 0.001, weight decay 0.0001; batch 1, accumulation 8, gradient norm limit 1. Up to 30 epochs; patience 5 on validation WAPE.<br/><br/><b>Boundaries.</b> Initial fit: Jan-Aug; final fresh refit: Jan-Oct. Fit vocabularies/scaling only on the allowed period. Route 5 uses an external zero fallback.<br/><br/><b>Comparisons.</b> Weekly weekday/hour mean, boarding-only CNN, full CNN. No prediction is fed back as observed history.<br/><br/><b>Evidence.</b> 26 fixture tests passed before this document. Actual model training was not run in the code-only delivery; no accuracy claim is made.",
    10,
    13,
)
line([(36, 145), (964, 145)], color="#A0A7AF")
para(
    36,
    130,
    928,
    "<b>Source of truth.</b> src/tram_forecast/config.py; model/{blocks,events,network}.py; schema.py; settings.py; train.py; evaluate.py. Parameter totals include every embedding row. Embedding count = sum_i (V_i + 3)e_i + 80, where V_i is the retained nonmissing vocabulary size. The boarding-only ablation removes E1/E2 and A and changes S1 input width from 64 to 32.",
    10,
    13,
)
para(
    36,
    82,
    928,
    "<b>Figure design reference.</b> User-supplied Google TFT architecture screenshot: grouped stages, explicit data-flow arrows and operator insets. This note depicts the implemented CNN, with its own operators and tensor contracts; no TFT attention, recurrence or gating is implied.",
    9.5,
    12,
)
c.showPage()
c.save()
print(OUT)
