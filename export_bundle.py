import argparse
import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import catboost as cb
import numpy as np
import polars as pl

from backtest import (
    CALIB_HI,
    CALIB_LO,
    FOLD_BOUNDS_EXT,
    apply_group_calibration,
    fit_group_calibration,
    make_folds,
)
from data_preparation import (
    CALENDAR_BLOCK_FEATURES,
    CYCLIC_INTRAWEEK,
    EXTERNAL_CAT_FEATURES,
    build_submit,
    enrich_features,
    load_calendar,
    load_labels,
)
from final_ensemble import CALIB_CUT, KEYS, ROUTES, SHRINK
from metrics import wape
from postprocess import (
    NEW_YEAR_DAYS,
    NEW_YEAR_HOURLY,
    ROUTE5_SHARE,
    adjust_new_year,
    fill_route5,
)

BASE = ["hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]
FEATURES = BASE + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES
MODEL_NAME = "catboost_cyclic_service"
TRAIN_CUT = date(2025, 10, 31)
HORIZON = ["2025-01-01", "2026-04-30"]
BANDS = {"night": [0, 5], "morning": [6, 10], "midday": [11, 16], "evening": [17, 23]}
LEADS = {"day": (1, 1), "week": (1, 7), "month": (1, 31), "season": (32, 61)}
QUANTILES = (0.1, 0.9)
FLOOR = {"day": 0.0, "week": 0.0, "month": 0.2, "season": 0.3}
MIN_PRED = 1.0
KEY = ["route", "date", "hour"]


def cat_features(meta: dict) -> list[str]:
    return [c for c in meta["cat_features"] + EXTERNAL_CAT_FEATURES if c in FEATURES]


def fit(meta: dict, train: pl.DataFrame, cut: date) -> cb.CatBoostRegressor:
    model = cb.CatBoostRegressor(**meta["params"], cat_features=cat_features(meta), verbose=0,
                                 allow_writing_files=False, random_seed=0)
    age = (np.datetime64(cut) - train["date"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]")
    frame = train.to_pandas()
    model.fit(frame[FEATURES], frame["target"], sample_weight=0.5 ** (age.astype(float) / meta["half_life_days"]))
    return model


def predict(model: cb.CatBoostRegressor, df: pl.DataFrame) -> np.ndarray:
    return model.predict(df.to_pandas()[FEATURES])


def band() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for name, (lo, hi) in BANDS.items():
        expr = pl.when(pl.col("hour").is_between(lo, hi)).then(pl.lit(name)).otherwise(expr)
    return expr.alias("band")


def quantiles(ratio: pl.Series, median: float) -> list[float]:
    lo, hi = (float(ratio.quantile(q, "linear")) / median for q in QUANTILES)
    return [round(min(lo, 1.0), 4), round(max(hi, 1.0), 4)]


def spread(ratio: pl.Series) -> float:
    return float(ratio.quantile(QUANTILES[1], "linear") - ratio.quantile(QUANTILES[0], "linear"))


def corridor(oof: pl.DataFrame) -> dict:
    cells = (oof.filter((pl.col("route") != 5) & (pl.col("pred") >= MIN_PRED))
             .with_columns(band(), (pl.col("target") / pl.col("pred")).alias("ratio")))
    days = (oof.filter(pl.col("route") != 5).group_by(["fold", "route", "date"])
            .agg(pl.col("target").sum(), pl.col("pred").sum())
            .filter(pl.col("pred") >= MIN_PRED).with_columns((pl.col("target") / pl.col("pred")).alias("ratio")))
    median = {"*": float(days["ratio"].median())} | {
        str(r): float(days.filter(pl.col("route") == r)["ratio"].median()) for r in sorted(days["route"].unique())}
    groups = {key: pl.lit(True) if key == "*" else pl.col("route") == int(key) for key, m in median.items() if m > 0}
    hourly = {key: {b: quantiles(g["ratio"], median[key]) for b in BANDS
                    if (g := cells.filter(where & (pl.col("band") == b))).height} for key, where in groups.items()}
    daily = {key: quantiles(days.filter(where)["ratio"], median[key]) for key, where in groups.items()}
    ref = spread(cells["ratio"])
    raw = {k: spread(cells.filter(pl.col("lead").is_between(lo, hi))["ratio"]) / ref for k, (lo, hi) in LEADS.items()}
    raw["day"] = raw["week"]
    widen, top = {}, 0.0
    for k in LEADS:
        top = max(top, raw[k])
        widen[k] = round(top, 3)
    pairs = [q for g in hourly.values() for q in g.values()] + list(daily.values())
    assert all(q[0] <= 1 <= q[1] for q in pairs), pairs
    assert list(widen.values()) == sorted(widen.values()), widen
    return {"quantiles": list(QUANTILES), "bands": BANDS, "hourly": hourly, "daily": daily, "widen": widen,
            "floor": FLOOR}


def new_year_table() -> dict[str, list[float]]:
    table = {}
    for d, k in NEW_YEAR_DAYS.items():
        shape = [1.0] * 24
        for hours, v in NEW_YEAR_HOURLY.get(d, {}).items():
            for h in hours:
                shape[h] = v
        table[d.isoformat()] = [k * s for s in shape]
    return table


def write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(out: Path) -> None:
    started = time.time()
    with open("artifacts/catboost/model_meta.json", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)
    calendar = load_calendar("input/calendar/2025.xml")
    train = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    all_df = pl.concat([train, test]).sort(["date", "hour", "route"])
    submit_df = enrich_features(build_submit(routes=ROUTES), calendar).sort(["date", "hour", "route"])

    folds = [make_folds(all_df, FOLD_BOUNDS_EXT)[i] for i in (2, 3)]
    assert folds[1].cut == CALIB_CUT
    parts, scores, models = [], {}, []
    for name, f in zip(("fold3", "fold4"), folds):
        model = fit(meta, f.train, f.cut)
        models.append(model)
        pred = np.clip(predict(model, f.valid), 0, None)
        scores[name] = round(1 - wape(f.valid["target"], pred), 4)
        parts.append(f.valid.select(KEY + ["target"]).with_columns(
            pl.Series("pred", pred), pl.lit(name).alias("fold"),
            (pl.col("date") - pl.lit(f.cut)).dt.total_days().alias("lead")))
    oof = pl.concat(parts)
    oof_score = round(1 - wape(oof["target"], oof["pred"]), 4)

    hold = all_df.filter(pl.col("date") > CALIB_CUT)
    k = fit_group_calibration(hold, np.clip(predict(models[1], hold), 0, None), KEYS, shrink=SHRINK)

    final = fit(meta, all_df, TRAIN_CUT)
    assert final.feature_names_ == FEATURES, final.feature_names_
    raw = predict(final, submit_df)
    cal = apply_group_calibration(submit_df, np.clip(raw, 0, None), k, KEYS)
    grid = pl.DataFrame({"route": submit_df["route"], "date": submit_df["date_str"], "hour": submit_df["hour"], "prediction": cal})
    post = adjust_new_year(fill_route5(grid)).rename({"prediction": "final"})
    reference = (grid.select(KEY).with_columns(pl.Series("raw", raw))
                 .join(post, on=KEY, how="left", validate="1:1").sort(KEY))
    assert reference.height == 14640 and reference["final"].null_count() == 0

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.cbm"
    final.save_model(str(model_path))
    reference.write_csv(out / "reference.csv")
    kmap: dict[str, dict[str, float]] = {}
    for route, weekday, v in k.filter(pl.col("route") != 5).sort("route", "weekday").select("route", "weekday", "k").iter_rows():
        kmap.setdefault(str(route), {})[str(weekday)] = v
    created = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write(out / "features.json", {"features": FEATURES, "cat_features": cat_features(meta)})
    write(out / "calibration.json", {"keys": list(KEYS), "shrink": SHRINK, "clip": [CALIB_LO, CALIB_HI], "k": kmap})
    write(out / "corridor.json", corridor(oof))
    write(out / "postprocess.json", {"route5_share": ROUTE5_SHARE, "new_year": new_year_table()})
    write(out / "meta.json", {
        "version": f"{created}/{digest[:6]}", "model": MODEL_NAME, "mode": "service",
        "train_period": ["2025-01-01", TRAIN_CUT.isoformat()], "horizon": HORIZON,
        "wape_score": oof_score, "folds": scores, "features": FEATURES, "cat_features": cat_features(meta),
        "trees": final.tree_count_, "catboost": cb.__version__,
        "created_at": created, "train_seconds": round(time.time() - started, 1),
        "reference": {"from": "2025-11-01", "to": "2025-12-31", "rows": reference.height, "raw_tolerance": 1e-6},
    })
    print(f"бандл {out}: out-of-fold WAPE-score {oof_score} {scores}, сумма ноя-дек {reference['final'].sum():,.0f}")
    old = Path("artifacts/preds/catboost_cyclic_submit.parquet")
    if old.exists():
        j = reference.join(pl.read_parquet(old).select(KEY + [pl.col("raw").alias("old")]), on=KEY)
        diff = np.abs(np.clip(j["raw"].to_numpy(), 0, None) - j["old"].to_numpy()).max()
        print(f"против {old}: max|raw - old| = {diff:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Обучает сервисную CatBoost тем же кодом, что сабмит, и пишет бандл")
    ap.add_argument("--out", type=Path, default=Path("artifacts/bundle/catboost"))
    main(ap.parse_args().out)
