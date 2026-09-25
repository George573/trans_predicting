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
BLUE = "#D7E9F5"
PURPLE = "#E4DCF3"
GREEN = "#D8EBE3"
ORANGE = "#FAE5C7"
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
    c.setFillColor(HexColor("#295575"))
    c.rect(x, y + h - 23, 3, 23, fill=1, stroke=0)
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
    line([(36, 701), (158, 701)], color="#295575", width=3)
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
    text(964, 25, f"{n} / 10", 9, "Helvetica", align="right")


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


def junction(x, y, radius=11):
    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(HexColor("#475569"))
    c.setLineWidth(1)
    c.circle(x, y, radius, fill=1, stroke=1)
    text(x, y - 3, "||", 11, "Helvetica-Bold", align="center")


def operation(x, y, w, h, label, sub=None, kind="pool"):
    # Taper = reduction; slanted sides = fixed tensor operation. Neither has learned weights.
    points = (
        [(x, y + h), (x + w, y + h), (x + w - 9, y), (x + 9, y)]
        if kind == "pool"
        else [(x + 9, y + h), (x + w, y + h), (x + w - 9, y), (x, y)]
    )
    c.setFillColor(HexColor(GREEN if kind == "pool" else GREY))
    c.setStrokeColor(HexColor("#475569"))
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.drawPath(path, fill=1, stroke=1)
    text(
        x + w / 2,
        y + h / 2 + (3 if sub else -3),
        label,
        9,
        "Times-Bold",
        align="center",
    )
    if sub:
        text(x + w / 2, y + h / 2 - 9, sub, 8, "Helvetica", align="center")


def grid(x, y, values, width=190, rowh=17, fill=ORANGE):
    cw = width / len(values[0])
    for r, row in enumerate(values):
        for j, value in enumerate(row):
            c.setFillColor(HexColor(fill if r else GREY))
            c.setStrokeColor(HexColor("#B5BDC5"))
            c.setLineWidth(0.4)
            c.rect(x + j * cw, y - (r + 1) * rowh, cw, rowh, fill=1, stroke=1)
            text(
                x + (j + 0.5) * cw,
                y - (r + 1) * rowh + 5,
                str(value),
                8.5,
                "Helvetica",
                align="center",
            )


def note(y, title, body):
    text(620, y, title, 13, "Times-Bold")
    para(620, y - 17, 325, body, 11, 15)


def stage(y, title, shape, explanation, kind="learned"):
    if kind == "learned":
        box(65, y, 470, 48, title, shape, BLUE, 12)
    else:
        operation(65, y, 470, 48, title, shape, kind=kind)
    para(620, y + 43, 325, explanation, 11, 15)


header(
    1,
    "1. What are we predicting?",
    "A reading guide: start with the observations, follow one sample, then inspect the architecture.",
)
para(
    55,
    623,
    880,
    "<b>Task.</b> For one tram route, predict the number of boardings in each of the 24 hours of a requested future day. The model sees 21 complete days of past observations. It does not receive transactions or boarding counts from the future day.",
    15,
    21,
)
text(55, 527, "A single example used throughout this explanation", 15, "Times-Bold")
box(
    55,
    423,
    330,
    70,
    "Observed history: route 17",
    "Oct 11, 00:00 through Oct 31, 23:00",
    ORANGE,
    14,
)
box(460, 423, 180, 70, "CNN model", "one shared model for all routes", BLUE, 14)
box(
    715,
    423,
    230,
    70,
    "Requested day: Nov 8",
    "24 nonnegative hourly predictions",
    GREEN,
    13,
)
line([(385, 458), (460, 458)], arrow=True)
line([(640, 458), (715, 458)], arrow=True)
para(
    55,
    395,
    330,
    "<b>Observation cutoff: Nov 1, 00:00.</b><br/>21 days x 24 hours = 504 historical positions. Everything at or after this cutoff is outside the observed window.",
    11,
    15,
)
para(
    715,
    395,
    230,
    "<b>Output positions:</b> 00:00, 01:00, ..., 23:00 on Nov 8. These are counts, not transaction categories.",
    11,
    15,
)
frame(55, 149, 420, 147, "Two complementary descriptions of the same past")
para(
    72,
    259,
    385,
    "<b>Raw path:</b> individual validation transactions, grouped into hours. It can learn patterns in transaction categories and their order.<br/><br/><b>Boarding path:</b> one observed boarding count per hour. It directly preserves how much demand occurred.",
    12,
    17,
)
frame(505, 149, 440, 147, "The request tells the model which future day to predict")
para(
    523,
    259,
    401,
    "The request supplies route identity, the requested day's calendar, and its distance from the cutoff. In this example Nov 8 has lead 8; Nov 1 has lead 1.<br/><br/>Changing the request does not change the frozen historical window.",
    12,
    17,
)
para(
    55,
    113,
    890,
    "<b>How to read this document.</b> Pages 2-3 define the data. Page 4 shows the full network; pages 5-6 trace its stages. Pages 7-10 give exact filters, temporal coverage, memory execution and evaluation details. All numerical examples are illustrative.",
    10.5,
    14,
)
c.showPage()

header(
    2,
    "2. Raw path: from transactions to hourly lists",
    "One row is one recorded validation transaction. An hour can contain many rows, one row, or no rows.",
)
text(55, 615, "Selected source columns: before grouping", 14, "Times-Bold")
table(
    55,
    593,
    530,
    ["route", "timestamp", "result", "type", "good", "place", "pass"],
    [
        ["17", "Oct 11 00:05", "1", "52", "A", "3", "-"],
        ["17", "Oct 11 00:18", "0", "52", "B", "3", "X"],
        ["17", "Oct 11 00:47", "1", "52", "A", "3", "Y"],
        ["17", "Oct 11 01:09", "1", "52", "A", "3", "-"],
    ],
    [0.7, 2.1, 0.8, 0.7, 0.7, 0.7, 0.7],
    rowh=26,
)
note(
    607,
    "Routing and time are alignment keys",
    "Route selects the service being forecast. Timestamp determines the historical hour and the order of events within it. Neither is one of the five event-category columns.",
)
note(
    507,
    "Five category values describe each event",
    "The short headings map, in order, to validation_result, tran_type_id, good_type, place_id and pass_route. Result 1 denotes successful boarding. Other codes are categorical values; their numeric size is not a measurement.",
)
text(55, 413, "Group by hour; preserve chronological event order", 14, "Times-Bold")
grid(
    55,
    386,
    [
        ["result", "type", "good", "place", "pass"],
        [1, 52, "A", 3, "-"],
        [0, 52, "B", 3, "X"],
        [1, 52, "A", 3, "Y"],
    ],
    330,
    23,
)
text(406, 350, "hour 0: 3 x 5", 12, "Helvetica")
grid(
    55,
    271,
    [["result", "type", "good", "place", "pass"], [1, 52, "A", 3, "-"]],
    330,
    23,
)
text(406, 243, "hour 1: 1 x 5", 12, "Helvetica")
box(
    55,
    157,
    330,
    39,
    "hour 2: empty list",
    "0 events; the hour still exists",
    ORANGE,
    12,
)
note(
    382,
    "Rows = events; columns = categories",
    "Each hour has its own row count N. This is a ragged structure: there is no single fixed event count for the entire 21-day window. The first two lists shown here belong to consecutive hours.",
)
note(
    271,
    "A missing field is not an empty hour",
    "'-' here means a missing category inside a real event. A missing-category token preserves that event. By contrast, an empty hour has no events and later receives an all-zero 32-feature raw summary.",
)
para(
    55,
    111,
    890,
    "<b>One raw input sample = 504 ordered hourly lists.</b> Each list is N x 5, with N varying by hour. Values are mapped to categorical IDs using training-only vocabularies; missing and unseen values have separate tokens. Padding for computation is explained on pages 5 and 9.",
    11,
    15,
)
c.showPage()

header(
    3,
    "3. Boarding path: an aligned matrix of hours",
    "This input has a fixed length. Counts and calendar values share exactly the same 504 historical hour positions.",
)
text(55, 615, "Source: one label row per route, date and hour", 14, "Times-Bold")
table(
    55,
    593,
    530,
    ["route", "date", "hour", "boardings"],
    [
        ["17", "Oct 11", "0", "2"],
        ["17", "Oct 11", "1", "1"],
        ["17", "Oct 11", "2", "0"],
    ],
    rowh=26,
)
note(
    606,
    "A count is an aggregate, not an event",
    "The model reads boardings from the hourly label files. Here 2, 1 and 0 agree with the successful toy transactions on page 2, but the input pipeline loads the count labels separately.",
)
text(55, 446, "One sample: rows are variables, columns are hours", 14, "Times-Bold")
grid(
    55,
    423,
    [
        ["hour index", "0", "1", "2", "...", "503"],
        ["count Y", "2", "1", "0", "...", "18"],
    ],
    530,
    28,
)
note(
    444,
    "A complete grid of 504 positions",
    "Column 0 is Oct 11 at 00:00; column 503 is Oct 31 at 23:00. Absent label rows are filled with zero under this dataset convention; source-presence metadata is retained for checks.",
)
grid(
    55,
    328,
    [
        ["calendar K", "0", "1", "2", "...", "503"],
        ["sin(hour)", "0", ".259", ".500", "...", "-.259"],
        ["cos(hour)", "1", ".966", ".866", "...", ".966"],
        ["sin(weekday)", "-.975", "-.975", "-.975", "...", "-.434"],
        ["cos(weekday)", "-.223", "-.223", "-.223", "...", "-.901"],
        ["sin(monthday)", ".898", ".898", ".898", "...", "-.201"],
        ["cos(monthday)", "-.440", "-.440", "-.440", "...", ".980"],
    ],
    530,
    20,
    BLUE,
)
note(
    320,
    "Six calendar features per hour",
    "Hour-of-day, weekday and day-of-month each become a sine/cosine pair. All six rows are shown; values are rounded for the October 2025 example. Cyclic encoding places adjacent times near each other, including across a cycle boundary.",
)
operation(
    55,
    108,
    530,
    48,
    "Stack scaled count and calendar along the row axis",
    "1 count row + 6 calendar rows = 7 x 504",
    kind="fixed",
)
note(
    153,
    "What enters the boarding CNN",
    "Divide counts by a training-set scale s, then stack [Y/s; K]. Stacking has no learned weights and does not add hours. The raw path receives the same K after its events are summarized.",
)
c.showPage()

# PAGE 1 -----------------------------------------------------------------
header(
    4,
    "4. Complete computation graph",
    "Read upward: the two histories introduced on pages 2-3 become 24 hourly predictions for the requested day.",
)
# Main bottom-up graph, modeled on grouped scientific architecture figures.
frame(36, 91, 605, 550, "Full model")
box(66, 110, 186, 29, "Categorical events", "per hour: [N, 5]", ORANGE)
box(66, 151, 186, 29, "Five embedding tables", "concatenate: [M, 28, Npad]", ORANGE)
box(66, 192, 186, 29, "E1: multiscale event block", "28 -> 32; mask after GELU")
box(66, 233, 186, 29, "E2: multiscale event block", "32 -> 32; mask after GELU")
box(66, 274, 186, 29, "Masked maximum over events", "scatter: Z [B, 32, 504]", GREEN)
box(315, 274, 178, 29, "Observed boarding history", "Y [B, 1, 504]", ORANGE)
operation(66, 315, 186, 29, "Concat [Z, K]", "raw input [B, 38, 504]", kind="fixed")
operation(
    315, 315, 178, 29, "Concat [Y / s, K]", "count input [B, 7, 504]", kind="fixed"
)
box(520, 274, 104, 29, "Calendar K", "[B, 6, 504]", ORANGE, size=10)
box(66, 356, 186, 29, "A: hourly multiscale block", "[B, 32, 504]")
box(315, 356, 178, 29, "B: hourly multiscale block", "[B, 32, 504]")
operation(181, 397, 230, 26, "Channel concatenation: [B, 64, 504]", kind="fixed")
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
operation(181, 549, 230, 23, "Flatten: [B, 672]", kind="fixed")
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
# Annotations and a small alignment example follow the preferred annotated layout.
para(
    48,
    504,
    120,
    "<b>Compress</b><br/>Learn cross-branch patterns, then reduce the sequence length before the dense head.",
    9,
    12,
)
line([(169, 477), (180, 477)], color="#70549A", arrow=True)
para(
    430,
    430,
    190,
    "<b>Align and fuse</b><br/>Join features for the same historical hour; keep their temporal order.",
    9,
    12,
)
line([(428, 405), (413, 405)], color="#70549A", arrow=True)
text(320, 244, "One route, three historical hour positions", 10, "Times-Bold")
for x, hour, count, events in [
    (381, "00:00", 2, 3),
    (454, "01:00", 1, 1),
    (527, "02:00", 0, 0),
]:
    text(x, 223, hour, 9, "Helvetica", align="center")
    text(x, 177, str(count), 10, "Helvetica-Bold", align="center")
    if events:
        for j in range(events):
            c.setFillColor(HexColor("#BC802A"))
            c.rect(x - 14 + j * 11, 200, 8, 11, fill=1, stroke=0)
    else:
        text(x, 201, "empty", 8, align="center")
text(320, 202, "Events", 9)
text(320, 177, "Counts", 9)
line([(363, 165), (579, 165)], color="#7A8590", arrow=True)
para(
    320,
    150,
    290,
    "Same toy hours as pages 2-3. An empty event hour receives a zero raw summary, while its calendar remains available.",
    9,
    12,
)
frame(662, 410, 302, 231, "1 / Raw input: one event list per hour")
para(
    675,
    614,
    274,
    "<b>One event = five categorical fields.</b> Group records by route and hour, then sort by timestamp. Event count varies from hour to hour.",
    10,
    13,
)
grid(
    675,
    554,
    [
        ["result", "type", "good", "place", "pass"],
        [1, 52, "A", 3, "-"],
        [0, 52, "B", 3, "X"],
        [1, 52, "A", 3, "Y"],
    ],
    266,
    22,
)
para(
    675,
    449,
    274,
    "Illustration: hour 0 has three events. The field definitions and the next two hours are shown on page 2. A dash denotes a missing category.",
    9.5,
    12,
)
frame(662, 247, 302, 148, "2 / Hourly input: volume and calendar")
para(
    675,
    369,
    274,
    "<b>Y:</b> 504 observed counts, one per chronological hour. <b>K:</b> six calendar rows on the same hour grid.<br/><br/><b>Why two branches?</b> Events describe transaction patterns; counts retain boarding volume. Event pooling makes the raw branch compatible with this fixed grid (page 3).",
    10,
    14,
)
frame(662, 91, 302, 140, "3 / Request: which future day to predict")
para(
    675,
    205,
    274,
    "Route 17; cutoff Nov 1; requested Nov 8. Lead is 8 days. The head receives a route embedding, the requested-day calendar and normalized lead 7/60.",
    10,
    13,
)
eq(675, 124, r"$q=[e_r;\ \kappa(d);\ (h-1)/60]\in\mathbb{R}^{13}$", 14, 274)
text(675, 104, "The future boarding target is never an input.", 9, "Times-Italic")
para(
    36,
    79,
    928,
    "<b>Figure 1.</b> B = sample batch; M = nonempty hours in a chunk; Npad = padded event length. Blue/purple blocks contain parallel convolutions. Amber = data/context; green = reduction/output; slanted grey shapes = fixed tensor operations. All routes share one network.",
    9.5,
    12,
)
c.showPage()

header(
    5,
    "5. Raw encoder: turn each hour into 32 features",
    "Read downward. The event axis exists only within an hour; these convolutions never cross an hour boundary.",
)
para(
    55,
    625,
    890,
    "<b>Feature/channel:</b> one learned numerical response at each event or hour. A shape 32 x N means 32 feature rows and N event columns.",
    11,
    15,
)
stage(
    551,
    "Look up five learned category embeddings",
    "N x 5 category IDs -> 28 x N features",
    "Each category ID selects a learned vector. Their widths are 4, 4, 8, 4 and 8; joining them gives 28 numerical features for every event.",
)
stage(
    436,
    "E1: three parallel convolution filters + GELU",
    "28 x N -> 32 x N",
    "Filters inspect short neighborhoods in the event sequence. Different kernel lengths and dilations preserve multiple views of that neighborhood. Their output channels are concatenated.",
)
stage(
    321,
    "E2: three parallel convolution filters + GELU",
    "32 x N -> 32 x N",
    "A second multiscale block combines the first block's features. Symmetric padding preserves event positions. The combined maximum context reaches six events to either side.",
)
stage(
    206,
    "Maximum over the valid event positions",
    "32 x N -> 32 values for this hour",
    "For each of 32 channels, keep its largest valid event response. This removes the variable event axis. An empty hour is explicitly assigned a zero vector.",
    kind="pool",
)
for y in [551, 436, 321]:
    line([(300, y), (300, y - 67)], arrow=True)
para(
    55,
    155,
    520,
    "<b>Repeat for all 504 hours and restore their time order:</b><br/>32 features per hour x 504 hours = <b>32 x 504</b>. This is now a regular hourly matrix, ready to align with the boarding path.",
    13,
    18,
)
para(
    620,
    159,
    325,
    "<b>Implementation detail.</b> Hours are packed into chunks and padded only for computation. A mask marks real events so artificial padding cannot affect the summary. Exact chunking is on page 9.",
    10.5,
    14,
)
text(
    55,
    69,
    "Shapes here describe one hour or one sample; the implementation adds a leading batch/chunk dimension.",
    10,
    "Times-Italic",
)
c.showPage()

header(
    6,
    "6. Hourly fusion and the requested-day prediction",
    "Read downward. Both branches now use hours as their sequence axis. Shapes exclude the batch dimension.",
)
box(55, 563, 250, 54, "Raw summary + calendar", "32 + 6 rows = 38 x 504", ORANGE, 12)
box(350, 563, 250, 54, "Scaled counts + calendar", "1 + 6 rows = 7 x 504", ORANGE, 12)
box(55, 475, 250, 54, "A: multiscale hourly CNN", "32 x 504", BLUE, 12)
box(350, 475, 250, 54, "B: multiscale hourly CNN", "32 x 504", BLUE, 12)
line([(180, 563), (180, 529)], arrow=True)
line([(475, 563), (475, 529)], arrow=True)
junction(326, 441, 13)
line([(180, 475), (180, 441), (313, 441)], arrow=True)
line([(475, 475), (475, 441), (339, 441)], arrow=True)
text(352, 420, "Stack channels: 64 x 504", 11)
note(
    609,
    "Learn patterns across hours",
    "Both A and B use local, daily and weekly filter offsets. They produce different learned features at the same hour positions. A uses event composition; B uses observed demand volume.",
)
note(
    479,
    "Concatenation is a join, not a layer",
    "The small circle stacks 32 raw-path channels and 32 boarding-path channels. It has no weights. Hour t in both branches still refers to exactly the same historical hour.",
)
for y, name, shape, pool, out in [
    (347, "S1", "64 x 504", 2, "64 x 252"),
    (260, "S2", "32 x 252", 2, "32 x 126"),
    (173, "S3", "16 x 126", 3, "16 x 42"),
]:
    box(55, y, 250, 48, name + ": multiscale convolution", shape, PURPLE, 12)
    operation(350, y, 250, 48, "Max pool: groups of " + str(pool) + " positions", out)
    line([(305, y + 24), (350, y + 24)], arrow=True)
    if name == "S1":
        line([(326, 428), (326, 410), (180, 410), (180, 395)], arrow=True)
    else:
        line([(475, y + 87), (475, y + 65), (180, y + 65), (180, y + 48)], arrow=True)
note(
    354,
    "Convolutions learn; pooling compresses",
    "Each S block combines all incoming channels using parallel temporal filters. Pooling then reduces the time axis. Repeating this yields 16 channels at 42 positions, rather than 504.",
)
operation(55, 85, 205, 46, "Flatten", "16 x 42 = 672", kind="fixed")
junction(300, 108, 12)
box(350, 85, 250, 46, "Dense head + softplus x s", "685 -> 128 -> 24 counts", GREEN, 12)
line([(475, 173), (475, 149), (158, 149), (158, 131)], arrow=True)
line([(260, 108), (288, 108)], arrow=True)
line([(312, 108), (350, 108)], arrow=True)
text(278, 65, "request: 13 values", 10)
line([(300, 78), (300, 96)], arrow=True)
note(
    219,
    "The request conditions the final answer",
    "Add 8 learned route features, 4 requested-day calendar features and 1 normalized lead value: 672 + 13 = 685. The head predicts all 24 hours at once. Softplus ensures nonnegative counts; multiplication by s restores count units.",
)
c.showPage()

# PAGE 2 -----------------------------------------------------------------
header(
    7,
    "7. Exact multiscale operators and parameterization",
    "The same input is inspected at several neighborhood sizes. Concatenation preserves each path's learned responses for the next stage.",
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
junction(758, 527, 17)
text(758, 489, "[M, 32, Npad]", 9, align="center")
line([(701, 527), (741, 527)], arrow=True)
operation(833, 506, 117, 42, "Mask multiplication", "mask [M,1,Npad]", kind="reshape")
line([(775, 527), (833, 527)], arrow=True)
text(
    54,
    447,
    "Mask invalid inputs before E1; repeat the mask after every event block. Hourly/shared blocks have no event padding mask.",
    9,
)
para(
    61,
    599,
    132,
    "<b>Input axes</b><br/>M hourly sequences;<br/>28 features per event;<br/>Npad event positions.",
    9,
    12,
)
line([(119, 554), (119, 549)], color="#295575", arrow=True)
para(
    706,
    594,
    227,
    "<b>Keep all three views</b><br/>12 + 12 + 8 output channels = 32. Padding preserves event positions; masking removes artificial padded positions.",
    9,
    12,
)
text(75, 477, "Example block: E1", 9, "Times-Italic")
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
        "Paths: (channels, kernel, dilation)",
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
    86,
    "Exact head: v is the flattened 672-D history; q is the 13-D request. Dropout is training-only.",
    9,
    "Times-Italic",
)
eq(
    48,
    52,
    r"$\hat{y}=s\,\mathrm{softplus}\!\left(W_2\,\mathrm{Dropout}_{0.1}\!\left[\mathrm{GELU}(W_1[v;q]+b_1)\right]+b_2\right),\quad W_1:685\to128,\ W_2:128\to24$",
    13,
    900,
)
c.showPage()

# PAGE 3 -----------------------------------------------------------------
header(
    8,
    "8. Temporal support, alignment and compression",
    "Input here is one feature vector per hour. Local, daily and weekly taps offer complementary views of the same 21-day timeline.",
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
    purpose = {
        1: "Adjacent-hour variation",
        24: "Same clock hour on nearby days",
        168: "Same hour and weekday across weeks",
    }[d]
    text(525, y - 20, purpose, 9, "Times-Italic", align="right")
for v in (-168, -48, -24, 0, 24, 48, 168):
    xx = mapping(v)
    line([(xx, 365), (xx, 359)], color="#7A8590")
    text(xx, 345, str(v), 8, "Helvetica", align="center")
text(
    304,
    326,
    "Each dot = a sampled hourly feature vector; lines show the shared time axis",
    10,
    align="center",
)
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
    "<b>Why pool?</b> Reducing 504 positions to 42 keeps the head compact. R is the widest nominal support span; J is spacing on the original hourly grid. Here s_l is stride, not the count scale s. Spans count positions; padding adds no observations.",
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
    9,
    "9. Variable event cardinality and exact tiled execution",
    "A busy hour may contain many events. Chunking limits temporary tensors; overlapping halos preserve exactly the full encoder's context.",
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
text(156, 530, "context only: 6 events", 10, align="center")
text(486, 530, "retained core (24 events shown schematically)", 10, align="center")
text(816, 530, "context only: 6 events", 10, align="center")
for i, label in [(0, "a-6"), (6, "a"), (29, "b-1"), (35, "b+5")]:
    text(x0 + i * unit + unit / 2, 502, label, 8, "Helvetica", align="center")
line([(222, 485), (750, 485)], width=1.5)
line([(222, 481), (222, 489)])
line([(750, 481), (750, 489)])
text(
    486,
    470,
    "Each cell indexes one embedded event (28 features). Retain core [a,b); halo outputs are discarded.",
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
    10,
    "10. Forecast protocol, objective and reproducibility",
    "One observed history is reused for many requested dates. A fixed-cutoff evaluation tests the same information constraints as the final forecast.",
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
        "route + requested date + lead -> 24 counts per day",
        ORANGE,
        size=10,
    )
    line([(243, y - 7), (243, y + 40)], width=1.1)
    text(243, y - 20, cut, 9, align="center")
    line([(233, y + 15), (254, y + 15)], arrow=True)
text(
    55,
    442,
    "Changing the requested date changes q, not the observed window. Lead 1 starts at c.",
    10,
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
