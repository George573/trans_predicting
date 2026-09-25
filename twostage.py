"""Двухступенчатая модель: дневной уровень x почасовые доли.

    прогноз(route, date, hour) = D(route, date) * s(route, тип дня, hour)

Разложение подсказано диагностикой: при точных дневных суммах наши модели дают WAPE
0.0736 против 0.0714 у идеального профиля, то есть форма суток уже почти оптимальна, а
вся ошибка сидит в дневном уровне, причём именно в разрезе дня недели (2.3 пункта).
Ступень 1 работает с 3 тысячами дневных наблюдений вместо 58 тысяч почасовых - тот же
сигнал там в 24 раза менее зашумлён.

Ни day_of_year, ни month в признаках нет: ноября и декабря в обучении не существует.
Их роль играет длина светового дня - в ноябре-декабре она 7-9 часов, ровно как в
январе-феврале, поэтому модель остаётся в виденной области и деревьям не приходится
экстраполировать.
"""

from dataclasses import dataclass, asdict, field
from datetime import date, timedelta

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import catboost as cb

from harmonic import daylight_hours

HOLIDAY_HORIZON = 14


@dataclass(frozen=True)
class TwoStageSpec:
    level_kind: str = "catboost"        # catboost | poisson
    iterations: int = 600
    learning_rate: float = 0.05
    depth: int = 4
    l2_leaf_reg: float = 3.0
    alpha: float = 1e-3                 # регуляризация для poisson
    weekday_daylight: bool = True       # явное взаимодействие дня недели со световым днём
    recent_level_weeks: int = 4         # признак "недавний уровень маршрута"
    share_weeks: int = 12               # окно истории для долей
    share_by_season: bool = False       # отдельные доли на зиму и осень

    def as_dict(self) -> dict:
        return asdict(self)


def _daytype(df: pl.DataFrame) -> pl.Series:
    return (
        pl.when(pl.col("is_holiday")).then(pl.lit("hol"))
        .when(pl.col("is_weekend")).then(pl.lit("we"))
        .otherwise(pl.lit("wd"))
    )


def _holiday_distance(dates: np.ndarray, holidays: set[date]) -> tuple[np.ndarray, np.ndarray]:
    """дней до ближайшего праздника впереди и после ближайшего позади, обрезано на 14"""
    to_h = np.full(len(dates), HOLIDAY_HORIZON, dtype=float)
    from_h = np.full(len(dates), HOLIDAY_HORIZON, dtype=float)
    for i, d in enumerate(dates):
        for k in range(1, HOLIDAY_HORIZON + 1):
            if d + timedelta(days=k) in holidays:
                to_h[i] = k
                break
        for k in range(1, HOLIDAY_HORIZON + 1):
            if d - timedelta(days=k) in holidays:
                from_h[i] = k
                break
        if d in holidays:
            to_h[i] = 0.0
            from_h[i] = 0.0
    return to_h, from_h


def to_daily(df: pl.DataFrame, holidays: set[date]) -> pl.DataFrame:
    """почасовая сетка -> одна строка на route x date с признаками дня"""
    has_target = "target" in df.columns
    agg = [pl.col("target").sum().alias("y_day")] if has_target else []

    daily = (
        df
        .group_by(["route", "date"])
        .agg(*agg,
             pl.col("weekday").first(),
             pl.col("is_holiday").first(),
             pl.col("is_weekend").first(),
             pl.col("is_short_working_day").first(),
             pl.col("day_of_year").first(),
             pl.col("season").first())
        .sort(["route", "date"])
        .with_columns(_daytype(pl.DataFrame()).alias("daytype") if False else _daytype(df.head(0)).alias("daytype"))
    )
    # daytype считаем по самим колонкам фрейма
    daily = daily.with_columns(
        pl.when(pl.col("is_holiday")).then(pl.lit("hol"))
        .when(pl.col("is_weekend")).then(pl.lit("we"))
        .otherwise(pl.lit("wd")).alias("daytype")
    )

    dates = daily["date"].to_list()
    to_h, from_h = _holiday_distance(np.array(dates), holidays)

    return daily.with_columns(
        pl.Series("daylight", daylight_hours(daily["day_of_year"].to_numpy().astype(float))),
        pl.Series("days_to_holiday", to_h),
        pl.Series("days_from_holiday", from_h),
    )


LEVEL_CATS = ["route", "weekday", "daytype"]
LEVEL_NUMS = ["daylight", "days_to_holiday", "days_from_holiday", "is_short_working_day", "recent_level"]


def _level_features(daily: pl.DataFrame, spec: TwoStageSpec) -> pl.DataFrame:
    out = daily.with_columns(pl.col("is_short_working_day").cast(pl.Int8))
    if spec.weekday_daylight:
        out = out.with_columns(
            (pl.col("daylight") * pl.col("is_weekend").cast(pl.Int8)).alias("daylight_x_weekend"),
            (pl.col("daylight") * pl.col("is_holiday").cast(pl.Int8)).alias("daylight_x_holiday"),
        )
    return out


def _level_cols(spec: TwoStageSpec) -> list[str]:
    cols = LEVEL_CATS + LEVEL_NUMS
    if spec.weekday_daylight:
        cols = cols + ["daylight_x_weekend", "daylight_x_holiday"]
    return cols


def _poisson_matrix(df: pl.DataFrame, spec: TwoStageSpec, routes: list[int]) -> np.ndarray:
    blocks = [np.stack([(df["route"].to_numpy() == r).astype(float) for r in routes], axis=1)]
    blocks.append(np.stack([(df["weekday"].to_numpy() == w).astype(float) for w in range(1, 8)], axis=1))
    blocks.append(np.stack([(df["daytype"].to_numpy() == t).astype(float) for t in ("wd", "we", "hol")], axis=1))
    nums = [c for c in _level_cols(spec) if c not in LEVEL_CATS]
    blocks.append(np.stack([df[c].to_numpy().astype(float) for c in nums], axis=1))
    return np.hstack(blocks)


def fit_shares(hist: pl.DataFrame, spec: TwoStageSpec) -> pl.DataFrame:
    """средняя доля часа в дневной сумме, нормированная в единицу по route x тип дня"""
    lo = hist["date"].max() - timedelta(weeks=spec.share_weeks)
    part = hist.filter(pl.col("date") > lo).with_columns(
        pl.when(pl.col("is_holiday")).then(pl.lit("hol"))
        .when(pl.col("is_weekend")).then(pl.lit("we"))
        .otherwise(pl.lit("wd")).alias("daytype")
    )
    keys = ["route", "daytype", "hour"] + (["season"] if spec.share_by_season else [])
    group = ["route", "daytype"] + (["season"] if spec.share_by_season else [])

    day_total = part.group_by(["route", "date"]).agg(pl.col("target").sum().alias("y_day"))
    shares = (
        part.join(day_total, on=["route", "date"])
        .filter(pl.col("y_day") > 0)
        .with_columns((pl.col("target") / pl.col("y_day")).alias("share"))
        .group_by(keys)
        .agg(pl.col("share").mean().alias("share"))
    )
    return shares.with_columns(
        (pl.col("share") / pl.col("share").sum().over(group)).alias("share")
    )


def fit_twostage(train_df: pl.DataFrame, spec: TwoStageSpec, cut: date, holidays: set[date]) -> dict:
    active = (
        train_df.group_by("route").agg(pl.col("target").sum().alias("t"))
        .filter(pl.col("t") > 0).sort("route")["route"].to_list()
    )
    part = train_df.filter(pl.col("route").is_in(active))

    daily = to_daily(part, holidays)
    recent_lo = daily["date"].max() - timedelta(weeks=spec.recent_level_weeks)
    recent = (
        daily.filter(pl.col("date") > recent_lo)
        .group_by("route").agg(pl.col("y_day").mean().alias("recent_level"))
    )
    daily = _level_features(daily.join(recent, on="route", how="left"), spec)

    cols = _level_cols(spec)
    y = daily["y_day"].to_numpy().astype(float)

    if spec.level_kind == "catboost":
        model = cb.CatBoostRegressor(
            iterations=spec.iterations, learning_rate=spec.learning_rate, depth=spec.depth,
            l2_leaf_reg=spec.l2_leaf_reg, loss_function="MAE", verbose=0,
            cat_features=LEVEL_CATS, allow_writing_files=False, random_seed=0,
        )
        model.fit(daily.select(cols).to_pandas(), y)
    elif spec.level_kind == "poisson":
        model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=spec.alpha, max_iter=5000))
        model.fit(_poisson_matrix(daily, spec, active), y)
    else:
        raise ValueError(spec.level_kind)

    return {
        "level": model, "shares": fit_shares(part, spec), "spec": spec,
        "routes": active, "recent": recent, "holidays": holidays,
    }


def predict_twostage(state: dict, df: pl.DataFrame) -> np.ndarray:
    spec: TwoStageSpec = state["spec"]
    cols = _level_cols(spec)

    daily = to_daily(df, state["holidays"]).join(state["recent"], on="route", how="left")
    daily = _level_features(daily.with_columns(pl.col("recent_level").fill_null(0.0)), spec)

    if spec.level_kind == "catboost":
        level = state["level"].predict(daily.select(cols).to_pandas())
    else:
        level = state["level"].predict(_poisson_matrix(daily, spec, state["routes"]))
    daily = daily.with_columns(pl.Series("D", np.clip(level, 0, None)))

    keys = ["route", "daytype", "hour"] + (["season"] if spec.share_by_season else [])
    out = (
        df
        .with_row_index("__i")
        .with_columns(
            pl.when(pl.col("is_holiday")).then(pl.lit("hol"))
            .when(pl.col("is_weekend")).then(pl.lit("we"))
            .otherwise(pl.lit("wd")).alias("daytype")
        )
        .join(daily.select(["route", "date", "D"]), on=["route", "date"], how="left")
        .join(state["shares"], on=keys, how="left")
        .with_columns((pl.col("D").fill_null(0.0) * pl.col("share").fill_null(0.0)).alias("pred"))
        .sort("__i")
    )
    pred = out["pred"].to_numpy().astype(float)
    known = np.isin(df["route"].to_numpy(), state["routes"])
    return np.where(known, np.clip(pred, 0, None), 0.0)
