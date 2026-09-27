import argparse
import hashlib
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tram_forecast.data import (
    REQUEST_INPUT_FEATURES,
    ROUTES,
    Boardings,
    history_inputs,
    request_inputs,
)
from tram_forecast.predict import fixed_forecast
from tram_forecast.train import load_model, train

from export_bundle import corridor, write
from metrics import wape
from postprocess import ROUTE5_SHARE, fill_route5

LABELS = ("dataset/labels/labels_day_train.csv", "dataset/labels/labels_day_test.csv")
CUTOFF = date(2025, 11, 1)
VALID_CUTOFF = date(2025, 9, 1)
DAYS = 61
HISTORY_DAYS = 14
FLAGS = ("is_holiday", "is_day_off", "is_short_working_day")
EVENTS = ("extended_night_service", "event_near_route", "is_citywide_event")
KEY = ["route", "date", "hour"]


def load_boardings(events: Path, **kwargs) -> Boardings:
    merged = None
    for path in LABELS:
        part = Boardings.load(path, events_path=events, **kwargs)
        if merged is None:
            merged = part
        else:
            mask = part.counts != 0
            merged.counts[mask] = part.counts[mask]
    return merged


def validation_model(events: Path, checkpoint: Path | None, out: Path):
    if checkpoint is not None:
        return load_model(checkpoint)
    torch.manual_seed(67)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = train(load_boardings(events, end=VALID_CUTOFF), load_boardings(events, start=date(2025, 8, 11), end=CUTOFF),
                 forecast_days=14, history_days=HISTORY_DAYS, epochs=50, batch_size=60, lr=8e-4, weight_decay=0.0,
                 device=device, output=out)
    return load_model(path)


def usual_inputs(inputs: np.ndarray, days: list[date]) -> np.ndarray:
    names = list(REQUEST_INPUT_FEATURES)
    usual = inputs.copy()
    for name in FLAGS + EVENTS:
        usual[:, names.index(name)] = 0.0
    usual[:, names.index("is_day_off")] = [float(d.weekday() >= 5) for d in days]
    return usual


@torch.inference_mode()
def head(model, boardings: Boardings) -> dict:
    first, second, third = model.head[0], model.head[3], model.head[6]
    days = [CUTOFF + timedelta(days=d) for d in range(DAYS)]
    base, inputs, usual = [], [], []
    for i, route in enumerate(ROUTES):
        history = {k: torch.as_tensor(v[None], dtype=torch.float32)
                   for k, v in history_inputs(boardings, route, CUTOFF, HISTORY_DAYS).items()}
        encoded = model.encode_history(history)[0]
        width = encoded.shape[0]
        split = width + model.route.embedding_dim
        base.append((first.weight[:, :width] @ encoded + first.weight[:, width:split] @ model.route.weight[i + 1] + first.bias).tolist())
        request = request_inputs(boardings.scheduled_events, route, days)
        inputs.append(request.tolist())
        usual.append(usual_inputs(request, days).tolist())
    return {
        "hidden_base": base, "hidden_request": first.weight[:, split:].tolist(),
        "hidden2_weight": second.weight.tolist(), "hidden2_bias": second.bias.tolist(),
        "output_weight": third.weight.tolist(), "output_bias": third.bias.tolist(),
        "inputs": inputs, "usual_inputs": usual,
    }


def replay(h: dict, scale: float) -> np.ndarray:
    t = {k: torch.tensor(h[k], dtype=torch.float64) for k in (
        "hidden_base", "hidden_request", "hidden2_weight", "hidden2_bias", "output_weight", "output_bias", "inputs")}
    out = np.zeros((len(ROUTES), DAYS, 24))
    for r in range(len(ROUTES)):
        for d in range(DAYS):
            x = torch.cat((t["inputs"][r, d], torch.tensor([d / 60], dtype=torch.float64)))
            h1 = torch.nn.functional.gelu(t["hidden_base"][r] + t["hidden_request"] @ x)
            h2 = torch.nn.functional.gelu(t["hidden2_weight"] @ h1 + t["hidden2_bias"])
            out[r, d] = (torch.nn.functional.softplus(t["output_weight"] @ h2 + t["output_bias"]) * scale).numpy()
    return out


def grid_frame(values: np.ndarray, start: date) -> pl.DataFrame:
    rows = [(route, (start + timedelta(days=d)).isoformat(), h, float(values[i, d, h]))
            for i, route in enumerate(ROUTES) for d in range(values.shape[1]) for h in range(24)]
    return pl.DataFrame(rows, schema={"route": pl.Int32, "date": pl.String, "hour": pl.Int8, "prediction": pl.Float64}, orient="row")


def main(a: argparse.Namespace) -> None:
    model = load_model(a.checkpoint)
    boardings = load_boardings(a.events)
    raw = fixed_forecast(model, boardings, CUTOFF, DAYS, HISTORY_DAYS)
    h = head(model, boardings)
    scale = float(model.scale)
    replayed = replay(h, scale)
    drift = np.abs(replayed - raw).max()
    assert np.allclose(replayed, raw, rtol=1e-4, atol=1e-3), drift

    vmodel = validation_model(a.events, a.validation_checkpoint, a.validation_out)
    vboard = load_boardings(a.events, start=date(2025, 8, 11), end=CUTOFF)
    vpred = fixed_forecast(vmodel, vboard, VALID_CUTOFF, DAYS, HISTORY_DAYS)
    oof = grid_frame(vpred, VALID_CUTOFF).rename({"prediction": "pred"}).with_columns(
        pl.col("date").str.to_date(),
        pl.Series("target", [float(v) for i, route in enumerate(ROUTES) for d in range(DAYS)
                             for v in vboard.target(route, VALID_CUTOFF + timedelta(days=d))]),
        pl.lit("valid").alias("fold"))
    oof = oof.with_columns(((pl.col("date") - pl.lit(VALID_CUTOFF)).dt.total_days() + 1).alias("lead"))
    score = round(1 - wape(oof["target"], oof["pred"]), 4)

    grid = grid_frame(raw, CUTOFF)
    reference = (fill_route5(grid).rename({"prediction": "final"})
                 .join(grid.rename({"prediction": "raw"}), on=KEY, how="left").select(KEY + ["raw", "final"]).sort(KEY))
    assert reference.height == 14640

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    reference.write_csv(out / "reference.csv")
    features = list(REQUEST_INPUT_FEATURES) + ["lead"]
    parameters = sum(p.numel() for p in model.parameters())
    write(out / "head.json", {
        "model": a.name, "cutoff": CUTOFF.isoformat(), "days": DAYS, "history_days": HISTORY_DAYS,
        "routes": list(ROUTES), "scale": scale, "parameters": parameters, "features": features, **h,
    })
    write(out / "corridor.json", corridor(oof))
    write(out / "postprocess.json", {"route5_share": ROUTE5_SHARE, "new_year": {}})
    created = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    write(out / "meta.json", {
        "version": f"{created}/{digest[:6]}", "model": a.name, "mode": "service",
        "train_period": ["2025-01-01", a.train_to], "horizon": [CUTOFF.isoformat(), (CUTOFF + timedelta(days=DAYS - 1)).isoformat()],
        "wape_score": score, "features": features, "cat_features": [], "cutoff": CUTOFF.isoformat(),
        "history_days": HISTORY_DAYS, "parameters": parameters, "checkpoint": a.checkpoint.name, "checkpoint_sha256": digest,
        "validation": {"cutoff": VALID_CUTOFF.isoformat(), "days": DAYS,
                       "checkpoint": a.validation_checkpoint.name if a.validation_checkpoint else "обучена рецептом notebooks/train.ipynb"},
        "created_at": created,
        "reference": {"from": CUTOFF.isoformat(), "to": (CUTOFF + timedelta(days=DAYS - 1)).isoformat(),
                      "rows": reference.height, "raw_tolerance": 1e-4},
    })
    print(f"бандл {out}: валидация сен-окт WAPE-score {score}, расхождение разложения головы {drift:.2e}, "
          f"сумма ноя-дек {reference['final'].sum():,.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Кодирует контекст CNN на отсечке 2025-11-01 и пишет бандл для Go")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--validation-checkpoint", type=Path)
    ap.add_argument("--validation-out", type=Path, default=Path(tempfile.gettempdir()) / "cnn_validation.pt")
    ap.add_argument("--name", default="cnn_14days_v9_events")
    ap.add_argument("--train-to", default="2025-09-30")
    ap.add_argument("--out", type=Path, default=Path("artifacts/bundle/cnn"))
    main(ap.parse_args())
