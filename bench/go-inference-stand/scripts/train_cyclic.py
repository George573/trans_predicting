"""Обучает catboost_cyclic ровно так, как final_ensemble.py для сабмита, и сохраняет .cbm.

В artifacts/catboost/model.cbm лежит base-модель (7 признаков), а в ансамбль и сабмит
идёт cyclic (плюс sin/cos часа, дня недели и дня месяца). Сервису нужна именно она.

Запуск из корня репозитория: python bench/go-inference-stand/scripts/train_cyclic.py
Пишет bench/go-inference-stand/model_cyclic.cbm и сверяет прогноз ноя-дек с
artifacts/preds/catboost_cyclic_submit.parquet (колонка raw), если он есть.
"""
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import catboost as cb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from data_preparation import CYCLIC_INTRAWEEK, build_submit, enrich_features, load_calendar, load_labels  # noqa: E402

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
BASE = ["hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]
FEATS = BASE + CYCLIC_INTRAWEEK
CUT = date(2025, 10, 31)


def main():
    meta = json.load(open(REPO / "artifacts/catboost/model_meta.json", encoding="utf-8"))
    cal = load_calendar(str(REPO / "input/calendar/2025.xml"))
    train = enrich_features(load_labels(str(REPO / "dataset/labels/labels_day_train.csv"), date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), cal)
    test = enrich_features(load_labels(str(REPO / "dataset/labels/labels_day_test.csv"), date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), cal)
    all_df = pl.concat([train, test]).sort(["date", "hour", "route"])

    # как make_cb в final_ensemble: те же параметры, веса с полураспадом half_life_days, seed 0
    m = cb.CatBoostRegressor(**meta["params"], cat_features=[c for c in meta["cat_features"] if c in FEATS],
                             verbose=0, allow_writing_files=False, random_seed=0)
    frame = all_df.to_pandas()
    age = (np.datetime64(CUT) - all_df["date"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]")
    w = 0.5 ** (age.astype(float) / meta["half_life_days"])
    m.fit(frame[FEATS], frame["target"], sample_weight=w)
    m.save_model(str(OUT / "model_cyclic.cbm"))
    print("saved", OUT / "model_cyclic.cbm", "features:", m.feature_names_)

    sub = enrich_features(build_submit(routes=ROUTES), cal).sort(["date", "hour", "route"])
    p = np.clip(m.predict(sub.to_pandas()[FEATS]), 0, None)
    sub.select(["route", "date_str", "hour"]).with_columns(pl.Series("cb", p)).write_csv(OUT / "ref_grid.csv")

    ref_path = REPO / "artifacts/preds/catboost_cyclic_submit.parquet"
    if ref_path.exists():
        ref = pl.read_parquet(ref_path).rename({"date": "date_str"})
        j = sub.select(["route", "date_str", "hour"]).with_columns(pl.Series("mine", p)).join(ref, on=["route", "date_str", "hour"])
        d = np.abs(j["mine"].to_numpy() - j["raw"].to_numpy())
        print(f"vs final_ensemble catboost_cyclic raw: max|diff|={d.max():.4g}, sum|diff|/sum={d.sum() / j['raw'].sum():.2e}")


if __name__ == "__main__":
    main()
