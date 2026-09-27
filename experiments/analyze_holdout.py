"""Разбор эксперимента holdout_months: горизонт, объём обучения, потолок интерполяции.

Читает artifacts/experiments/holdout_months_*.parquet и печатает таблицы, на которых
построен отчёт docs/RESEARCH-horizon.md.

Ключевой приём разложения. В расширяющемся окне горизонт и целевой месяц связаны: лаг 9
встречается только у октября по январю. Поэтому эффект горизонта снимается на скользящих
окнах, где один и тот же месяц прогнозируется с разных расстояний, а базой служит lomo_flat -
ошибка того же месяца, когда он окружён обучающими данными. Разность между ними и есть
плата за незнание, очищенная от свойств самого месяца.

Запуск: PYTHONPATH=. python3 experiments/analyze_holdout.py
"""

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("artifacts/experiments")
MODELS = ("catboost_cyclic", "harmonic", "mix_cb70_mlp30")
SUMMER = (6, 7, 8)


def show(title: str, df: pl.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=4):
        print(df)


def main() -> None:
    m = pl.read_parquet(OUT / "holdout_months_months.parquet")
    q = pl.read_parquet(OUT / "holdout_months_quantiles.parquet")
    r = pl.read_parquet(OUT / "holdout_months_routes.parquet")

    # ---- 1. расширяющееся окно: как в жизни, горизонт растёт, обучение тоже
    show("расширяющееся окно: WAPE по часам, строки - край обучения, колонки - целевой месяц",
         m.filter(pl.col("scheme") == "expanding", pl.col("model") == "mix_cb70_mlp30")
          .pivot(on="target_month", index="train_hi", values="wape_hour").sort("train_hi"))

    show("расширяющееся окно: WAPE по часам как функция лага (все модели)",
         m.filter(pl.col("scheme") == "expanding")
          .group_by(["model", "lead"]).agg(
              pl.col("wape_hour").mean().alias("hour"),
              pl.col("wape_day").mean().alias("day"),
              pl.col("wape_month_route").mean().alias("month"),
              pl.col("bias_city").mean().alias("bias"),
              pl.len().alias("n"))
          .sort(["model", "lead"])
          .pivot(on="model", index="lead", values="hour").sort("lead"))

    # ---- 2. потолок интерполяции
    show("lomo: месяц окружён обучением (flat - равные веса, обычный - затухание 60 дней)",
         m.filter(pl.col("scheme").is_in(["lomo", "lomo_flat"]))
          .pivot(on=["scheme", "model"], index="target_month", values="wape_hour")
          .sort("target_month"))

    floor = (m.filter(pl.col("scheme") == "lomo_flat")
             .select(["model", "target_month", "wape_hour", "wape_day", "wape_month_route"])
             .rename({"wape_hour": "floor_hour", "wape_day": "floor_day",
                      "wape_month_route": "floor_month"}))

    # ---- 3. плата за расстояние, очищенная от свойств месяца
    slide = (m.filter(pl.col("scheme").str.starts_with("sliding"))
             .join(floor, on=["model", "target_month"], how="left")
             .with_columns((pl.col("wape_hour") - pl.col("floor_hour")).alias("excess_hour"),
                           (pl.col("wape_day") - pl.col("floor_day")).alias("excess_day"),
                           (pl.col("wape_month_route") - pl.col("floor_month")).alias("excess_month"),
                           (pl.col("target_month").is_in(SUMMER)).alias("target_summer")))

    show("скользящее окно 1 месяц: превышение над потолком интерполяции по расстоянию",
         slide.filter(pl.col("n_train_months") == 1)
              .group_by(["model", "dist_signed"]).agg(
                  pl.col("excess_hour").mean(), pl.len().alias("n"))
              .pivot(on="model", index="dist_signed", values="excess_hour").sort("dist_signed"))

    show("вперёд против назад: среднее превышение по абсолютному расстоянию, окно 1 месяц",
         slide.filter(pl.col("n_train_months") == 1)
              .group_by(["model", "direction", "dist_abs"]).agg(pl.col("excess_hour").mean(), pl.len().alias("n"))
              .filter(pl.col("dist_abs") <= 5)
              .pivot(on="direction", index=["model", "dist_abs"], values="excess_hour")
              .sort(["model", "dist_abs"]))

    show("объём обучения при равном расстоянии: окна 1, 2, 3 месяца",
         slide.group_by(["model", "n_train_months", "dist_abs"]).agg(pl.col("excess_hour").mean())
              .filter(pl.col("dist_abs") <= 4)
              .pivot(on="n_train_months", index=["model", "dist_abs"], values="excess_hour")
              .sort(["model", "dist_abs"]))

    show("режим цели против расстояния: лето и учебный год отдельно, окно 1 месяц",
         slide.filter(pl.col("n_train_months") == 1)
              .group_by(["model", "target_summer", "dist_abs"]).agg(pl.col("excess_hour").mean())
              .filter(pl.col("dist_abs") <= 5)
              .pivot(on="target_summer", index=["model", "dist_abs"], values="excess_hour")
              .sort(["model", "dist_abs"]))

    # ---- 4. агрегация: час / день / месячный итог
    show("уровни агрегации на расширяющемся окне, смесь",
         m.filter(pl.col("scheme") == "expanding", pl.col("model") == "mix_cb70_mlp30")
          .group_by("lead").agg(pl.col("wape_hour").mean(), pl.col("wape_day").mean(),
                                pl.col("wape_month_route").mean(),
                                pl.col("bias_city").mean().alias("bias_city"))
          .sort("lead"))

    # ---- 5. интервалы
    show("квантильный интервал 10-90 на расширяющемся окне: ширина и покрытие по лагу",
         q.group_by("lead").agg(pl.col("width_rel").mean(), pl.col("coverage_hour").mean(),
                                pl.col("coverage_day").mean(), pl.len().alias("n")).sort("lead"))

    # ---- 6. аппроксимация роста ошибки по лагу (только учебный режим цели)
    print("\n=== рост ошибки по лагу: подгонка на расширяющемся окне ===")
    for model in MODELS:
        d = (m.filter(pl.col("scheme") == "expanding", pl.col("model") == model,
                      ~pl.col("target_month").is_in(SUMMER))
             .group_by("lead").agg(pl.col("wape_hour").mean()).sort("lead"))
        x, y = d["lead"].to_numpy().astype(float), d["wape_hour"].to_numpy()
        for label, basis in (("линейная a+b*h", x), ("корневая a+b*sqrt(h)", np.sqrt(x)),
                             ("логарифм a+b*ln(h)", np.log(x))):
            A = np.vstack([np.ones_like(basis), basis]).T
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            f12 = coef[0] + coef[1] * ({"линейная a+b*h": 12.0, "корневая a+b*sqrt(h)": np.sqrt(12),
                                        "логарифм a+b*ln(h)": np.log(12)}[label])
            print(f"  {model:<18} {label:<22} a={coef[0]:.4f} b={coef[1]:.4f} R2={r2:.3f} -> h=12: {f12:.4f}")

    # ---- 7. маршруты: где ошибка растёт быстрее
    show("вклад маршрутов: превышение над потолком по расстоянию, окно 1 месяц, смесь",
         r.filter(pl.col("scheme") == "sliding1", pl.col("model") == "mix_cb70_mlp30")
          .with_columns(pl.col("dist_signed").abs().alias("dist_abs"))
          .group_by(["route", "dist_abs"]).agg(pl.col("wape_local").mean())
          .filter(pl.col("dist_abs").is_in([1, 3, 5, 7]))
          .pivot(on="dist_abs", index="route", values="wape_local").sort("route"))


if __name__ == "__main__":
    main()
