"""Build a vector architecture note from the executable boarding-only defaults."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from tram_forecast.config import Config
from tram_forecast.model import ForecastNetwork


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/tram_cnn_scientific_architecture.pdf"


def build(output=OUT):
    cfg = Config()
    model = ForecastNetwork(1, cfg)
    styles = getSampleStyleSheet()
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 8))

    paragraph("Boarding-only tram forecasting CNN", "Title")
    paragraph("Bounded hourly observations, encoded once per route and cutoff.")
    paragraph("Default computation graph", "Heading2")
    length = cfg.history_days * 24
    rows = [["Stage", "Output (channels x hours)"]]
    rows.append(["Boardings / scale + six calendar channels", f"7 x {length}"])
    width = sum(c for c, _, _ in cfg.hourly)
    length = (length + cfg.hourly_stride - 1) // cfg.hourly_stride
    rows.append(["Parallel hourly Conv1d + GELU paths", f"{width} x {length}"])
    for i, (paths, factor) in enumerate(zip(
        (cfg.shared1, cfg.shared2, cfg.shared3)[:cfg.shared_depth], (2, 2, 3)
    ), 1):
        width = sum(c for c, _, _ in paths)
        length //= factor
        rows.append([f"Shared CNN {i} + {cfg.temporal_pool} pool {factor}", f"{width} x {length}"])
    if cfg.pooled_hours is not None:
        rows.append(["Adaptive temporal pooling", f"{width} x {cfg.pooled_hours}"])
    rows.extend([
        ["Flatten history", str(model.encoded_width)],
        ["Add route (8), target calendar (4), lead (1)", str(model.encoded_width + 13)],
        ["Linear + GELU + dropout", str(cfg.head_width)],
        ["Linear + Softplus, multiply by scale", "24 hourly boarding counts"],
    ])
    table = Table(rows, colWidths=[4.5 * inch, 2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7e9f5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    paragraph(f"Total parameters: <b>{model.parameter_report()['total']:,}</b>.")
    paragraph("Convolution paths", "Heading2")
    paragraph("Each tuple is (output channels, kernel length, dilation). Paths use same "
              "padding and GELU, then concatenate channels.")
    for name in ("hourly", "shared1", "shared2", "shared3"):
        paragraph(f"<b>{name}</b>: {getattr(cfg, name)}")
    paragraph("Conditioning and data contracts", "Heading2")
    paragraph("History calendar: sine/cosine of hour/24, weekday/7 and "
              "(day-of-month - 1)/31. Target calendar omits hour. The route embedding "
              "has eight values. Lead is (lead - 1)/60 for integer leads 1 through 61.")
    paragraph("Count scale is max(1, mean fitting-period counts). Only hourly boarding "
              "label CSVs are required. Artifact schema 2 stores count and presence grids, "
              "scale, metadata and separate validation targets. No raw transaction lists "
              "or event vocabularies are used.")
    paragraph("Training contexts share one history encoding across requested days. Loss "
              "and gradient accumulation weight requested days equally. Validation freezes "
              "history at September 1; final refit freezes it at November 1. Route 5 uses "
              "an external zero fallback. Forecasts are never fed back as observed history.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), title="Boarding-only tram forecasting architecture").build(story)
    return output


if __name__ == "__main__":
    print(build())
