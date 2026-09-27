"""Что именно портит прогноз на дальнем горизонте: расстояние или несовпадение режима.

Первый разбор (analyze_holdout) показал странное: превышение над потолком интерполяции
растёт до расстояния 3 месяцев, а дальше падает, и на расстоянии 8-9 месяцев почти
исчезает. Объяснение не в расстоянии. У модели нет лаговых признаков - она не «забывает»
с горизонтом, она ошибается там, где сезонный уровень целевого месяца не представлен в
обучении. Расстояние 9 в наших данных - это пара «январь -> октябрь», а у них рабочий день
отличается на 4%; расстояние 3 - это часто «апрель -> июль», где разрыв 22%.

Скрипт проверяет это количественно и печатает таблицы для отчёта.

Запуск: PYTHONPATH=. python3 experiments/analyze_horizon_drivers.py
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data_preparation import load_calendar, load_labels, enrich_features
from metrics import wape

OUT = Path("artifacts/experiments")
MODELS = ("catboost_cyclic", "harmonic", "mix_cb70_mlp30")
EVAL_ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)


def show(title: str, df: pl.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=190, float_precision=4):
        print(df)


def monthly_level() -> pl.DataFrame:
    """уровень месяца: средний объём рабочего дня по городу, нормированный на год"""
    cal = load_calendar("input/calendar/2025.xml")
    tr = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=EVAL_ROUTES), cal)
    te = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=EVAL_ROUTES), cal)
    df = pl.concat([tr, te]).with_columns(pl.col("date").dt.month().alias("m"))
    daily = (df.group_by(["m", "date"]).agg(pl.col("target").sum().alias("v"),
                                            (pl.col("is_weekend") | pl.col("is_holiday")).any().alias("off")))
    lvl = (daily.filter(~pl.col("off")).group_by("m").agg(pl.col("v").mean().alias("level")).sort("m"))
    return lvl.with_columns((pl.col("level") / pl.col("level").mean()).alias("level_idx"))


def main() -> None:
    m = pl.read_parquet(OUT / "holdout_months_months.parquet")
    lvl = monthly_level()
    show("уровень месяца: средний рабочий день, индекс к среднему по 10 месяцам", lvl)

    level_map = dict(zip(lvl["m"].to_list(), lvl["level_idx"].to_list()))

    # разрыв уровня между целью и обучением: по ближайшему обучающему месяцу и по среднему
    def gaps(struct) -> dict:
        train = [int(x) for x in struct["train_months"].split(",")]
        t = struct["target_month"]
        nearest = min(train, key=lambda s: abs(t - s))
        lt = level_map[t]
        return {
            "lr_nearest": float(np.log(lt / level_map[nearest])),
            "lr_mean": float(np.log(lt / np.mean([level_map[s] for s in train]))),
            "lr_best": float(min(abs(np.log(lt / level_map[s])) for s in train)),
        }

    m = m.with_columns(
        pl.struct(["train_months", "target_month"]).map_elements(gaps, return_dtype=pl.Struct([
            pl.Field("lr_nearest", pl.Float64), pl.Field("lr_mean", pl.Float64),
            pl.Field("lr_best", pl.Float64)])).alias("g")
    ).unnest("g").with_columns(pl.col("dist_signed").abs().alias("dist_abs"))

    floor = (m.filter(pl.col("scheme") == "lomo_flat")
             .select(["model", "target_month", "wape_hour", "wape_hour_level_route"])
             .rename({"wape_hour": "floor_hour", "wape_hour_level_route": "floor_shape"}))
    sl = (m.filter(pl.col("scheme").str.starts_with("sliding"))
          .join(floor, on=["model", "target_month"], how="left")
          .with_columns((pl.col("wape_hour") - pl.col("floor_hour")).alias("excess"),
                        (pl.col("wape_hour_level_route") - pl.col("floor_shape")).alias("excess_shape")))

    # 1. что лучше объясняет превышение: расстояние или разрыв уровня
    print("\n=== чем объясняется превышение над потолком (скользящие окна, R2 одномерной регрессии) ===")
    for model in MODELS:
        d = sl.filter(pl.col("model") == model)
        y = d["excess"].to_numpy()
        for label, x in (("расстояние |dist|", d["dist_abs"].to_numpy().astype(float)),
                         ("разрыв уровня |lr| до ближайшего", np.abs(d["lr_nearest"].to_numpy())),
                         ("разрыв уровня |lr| до лучшего месяца", d["lr_best"].to_numpy()),
                         ("разрыв уровня |lr| до среднего обучения", np.abs(d["lr_mean"].to_numpy()))):
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f"  {model:<18} {label:<38} R2={r2:.3f}  наклон={coef[1]:+.4f}")

    # 2. главная таблица: расстояние при совпадающем режиме
    show("расстояние при СОВПАДАЮЩЕМ уровне (|lr| < 0.05): превышение над потолком",
         sl.filter(pl.col("lr_best").abs() < 0.05)
           .group_by(["model", "dist_abs"]).agg(pl.col("excess").mean(), pl.len().alias("n"))
           .pivot(on="model", index="dist_abs", values="excess").sort("dist_abs"))

    show("расстояние при РАСХОДЯЩЕМСЯ уровне (|lr| > 0.12): превышение над потолком",
         sl.filter(pl.col("lr_best").abs() > 0.12)
           .group_by(["model", "dist_abs"]).agg(pl.col("excess").mean(), pl.len().alias("n"))
           .pivot(on="model", index="dist_abs", values="excess").sort("dist_abs"))

    show("превышение по корзинам разрыва уровня, независимо от расстояния",
         sl.with_columns(pl.col("lr_best").abs().cut([0.03, 0.06, 0.12, 0.20],
                                                     labels=["<3%", "3-6%", "6-12%", "12-20%", ">20%"]).alias("bucket"))
           .group_by(["model", "bucket"]).agg(pl.col("excess").mean(),
                                              pl.col("excess_shape").mean(),
                                              pl.col("dist_abs").mean().alias("dist_avg"),
                                              pl.len().alias("n"))
           .sort(["model", "bucket"]))

    # 3. уровень против формы: сколько ошибки снимается заданием уровня извне
    show("ошибка уровня против ошибки формы, скользящие окна, по корзинам разрыва",
         sl.with_columns(pl.col("lr_best").abs().cut([0.03, 0.06, 0.12, 0.20],
                                                     labels=["<3%", "3-6%", "6-12%", "12-20%", ">20%"]).alias("bucket"))
           .filter(pl.col("model") == "mix_cb70_mlp30")
           .group_by("bucket").agg(pl.col("wape_hour").mean().alias("сырой"),
                                   pl.col("wape_hour_level_city").mean().alias("уровень_города_задан"),
                                   pl.col("wape_hour_level_route").mean().alias("уровень_маршрутов_задан"),
                                   pl.col("bias_city").mean().alias("сдвиг"),
                                   pl.len().alias("n")).sort("bucket"))

    show("расширяющееся окно: уровень против формы по лагу, смесь",
         m.filter(pl.col("scheme") == "expanding", pl.col("model") == "mix_cb70_mlp30")
          .group_by("lead").agg(pl.col("wape_hour").mean().alias("сырой"),
                                pl.col("wape_hour_level_city").mean().alias("уровень_города_задан"),
                                pl.col("wape_hour_level_route").mean().alias("уровень_маршрутов_задан"),
                                pl.col("bias_city").mean().alias("сдвиг")).sort("lead"))

    # 4. эмпирические интервалы: разброс суточного итога маршрута
    preds = pl.read_parquet(OUT / "holdout_months_preds.parquet")
    daily = (preds.with_columns(pl.col("config").str.split(":").list.get(0).alias("scheme"),
                                pl.col("config").str.split(":").list.get(1).alias("train_months"))
             .group_by(["scheme", "train_months", "m", "route", "date"])
             .agg(pl.col("target").sum().alias("y"), pl.col("mix_cb70_mlp30").sum().alias("p")))
    daily = (daily.filter(pl.col("y") > 0)
             .with_columns((pl.col("p") / pl.col("y")).alias("ratio"))
             .join(m.filter(pl.col("model") == "mix_cb70_mlp30")
                    .select(["scheme", "train_months", "target_month", "dist_abs", "lr_best"])
                    .rename({"target_month": "m"}), on=["scheme", "train_months", "m"], how="inner"))

    show("эмпирический разброс суточного итога маршрута: квантили отношения прогноз/факт",
         daily.filter(pl.col("scheme").str.starts_with("sliding"))
              .with_columns(pl.col("lr_best").abs().cut([0.03, 0.06, 0.12],
                                                        labels=["<3%", "3-6%", "6-12%", ">12%"]).alias("bucket"))
              .group_by("bucket").agg(pl.col("ratio").quantile(0.1).alias("q10"),
                                      pl.col("ratio").quantile(0.5).alias("медиана"),
                                      pl.col("ratio").quantile(0.9).alias("q90"),
                                      pl.len().alias("n")).sort("bucket"))

    show("то же для потолка интерполяции (lomo_flat) - что достижимо при полном покрытии",
         daily.filter(pl.col("scheme") == "lomo_flat")
              .group_by("m").agg(pl.col("ratio").quantile(0.1).alias("q10"),
                                 pl.col("ratio").quantile(0.5).alias("медиана"),
                                 pl.col("ratio").quantile(0.9).alias("q90")).sort("m"))


if __name__ == "__main__":
    main()
