"""Общий бэктест для всех моделей: расширяющееся окно, горизонт два календарных месяца.

Калибровка считается ВНУТРИ фолда, на периоде, предшествующем валидации. Иначе фолды
систематически занижают модели, чувствительные к сдвигу уровня (деревья): их фолдовая
модель обучена по более раннему краю, чем финальная, и недобирает по уровню, а в сабмите
этого гандикапа уже нет. Мерить надо то, что реально уходит в сабмит - калиброванный
прогноз.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Protocol

import numpy as np
import polars as pl

from metrics import wape

CALIB_LO, CALIB_HI = 0.85, 1.20

FOLD_BOUNDS: list[tuple[date, date, date]] = [
    (date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
    (date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    (date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
]

# Фолд с непрерывным режимом: осень по осени, без летнего разрыва. Ближе всего к реальной
# задаче (ноя-дек по янв-окт), поэтому для моделей, чувствительных к смене режима, отбор
# имеет смысл вести по нему и по третьему фолду, а не по среднему из четырёх.
FOLD_CONTINUOUS: tuple[date, date, date] = (date(2025, 9, 30), date(2025, 10, 1), date(2025, 10, 31))
FOLD_BOUNDS_EXT: list[tuple[date, date, date]] = FOLD_BOUNDS + [FOLD_CONTINUOUS]


@dataclass(frozen=True)
class Fold:
    cut: date
    train: pl.DataFrame
    valid: pl.DataFrame
    calib_cut: date
    calib_train: pl.DataFrame
    calib_valid: pl.DataFrame

    def describe(self) -> str:
        return (
            f"край трейна {self.cut}: train {self.train.height:>6} строк, "
            f"valid {self.valid.height:>5} ({self.valid['date'].min()} - {self.valid['date'].max()}), "
            f"калибровка по {self.calib_valid.height} строкам "
            f"({self.calib_valid['date'].min()} - {self.calib_valid['date'].max()})"
        )


def make_folds(all_df: pl.DataFrame, bounds=FOLD_BOUNDS) -> list[Fold]:
    folds: list[Fold] = []
    for cut, v0, v1 in bounds:
        horizon = (v1 - v0).days + 1
        calib_cut = cut - timedelta(days=horizon)
        folds.append(Fold(
            cut=cut,
            train=all_df.filter(pl.col("date") <= cut),
            valid=all_df.filter((pl.col("date") >= v0) & (pl.col("date") <= v1)),
            calib_cut=calib_cut,
            calib_train=all_df.filter(pl.col("date") <= calib_cut),
            calib_valid=all_df.filter((pl.col("date") > calib_cut) & (pl.col("date") <= cut)),
        ))
    return folds


def fit_calibration(y_true, y_pred, routes, lo: float = CALIB_LO, hi: float = CALIB_HI) -> dict[int, float]:
    """мультипликативный коэффициент на маршрут: во сколько раз модель недобирает"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    r = np.asarray(routes)

    calib: dict[int, float] = {}
    for route in np.unique(r):
        mask = r == route
        y_sum, p_sum = float(y[mask].sum()), float(p[mask].sum())
        calib[int(route)] = float(np.clip(y_sum / p_sum, lo, hi)) if y_sum > 0 and p_sum > 0 else 1.0
    return calib


def apply_calibration(y_pred, routes, calib: dict[int, float]) -> np.ndarray:
    p = np.asarray(y_pred, dtype=float).copy()
    r = np.asarray(routes)
    for route, k in calib.items():
        p[r == route] *= k
    return p


def cv_score(
    fit_fn: Callable[[pl.DataFrame, date], object],
    predict_fn: Callable[[object, pl.DataFrame], np.ndarray],
    folds: list[Fold],
    calibrate: bool = True,
) -> tuple[float, list[float], list[dict]]:
    """средний WAPE по фолдам; внутри фолда метрика глобальная, как на лидерборде.

    fit_fn(train_df, cut) -> модель, predict_fn(модель, df) -> прогноз в порядке строк df.
    """
    scores: list[float] = []
    details: list[dict] = []

    for fold in folds:
        calib: dict[int, float] = {}
        if calibrate:
            inner = fit_fn(fold.calib_train, fold.calib_cut)
            calib = fit_calibration(
                fold.calib_valid["target"],
                predict_fn(inner, fold.calib_valid),
                fold.calib_valid["route"],
            )

        model = fit_fn(fold.train, fold.cut)
        raw = predict_fn(model, fold.valid)
        pred = apply_calibration(raw, fold.valid["route"], calib) if calib else raw

        y = fold.valid["target"]
        scores.append(wape(y, pred))
        details.append({
            "cut": str(fold.cut),
            "wape": wape(y, pred),
            "wape_raw": wape(y, raw),
            "calibration": calib,
        })

    return float(np.mean(scores)), scores, details


def fit_group_calibration(
    df: pl.DataFrame,
    y_pred,
    keys: tuple[str, ...] = ("route", "weekday"),
    shrink: float = 5.0,
    lo: float = CALIB_LO,
    hi: float = CALIB_HI,
) -> pl.DataFrame:
    """Мультипликативная поправка по группам с усадкой к единице.

    Коэффициентов много (route x weekday это 70 ячеек), наблюдений в каждой мало, поэтому
    сырые отношения шумные: k = 1 + (k_raw - 1) * n / (n + shrink), где n - число дней в
    ячейке. Замер на фолде 4: усадка 5 улучшает поправку монотонно.

    ВАЖНО: окно оценки должно лежать в том же режиме, что и цель. Калибровка через границу
    лето/учебный год измеримо вредит (фолд 3: 0.1096 -> 0.13-0.18), внутри режима помогает
    (фолд 4: 0.1051 -> 0.0997).
    """
    return (
        df.with_columns(pl.Series("__p", np.asarray(y_pred, dtype=float)))
        .group_by(list(keys))
        .agg(
            pl.col("target").sum().alias("y"),
            pl.col("__p").sum().alias("p"),
            pl.col("date").n_unique().alias("n"),
        )
        .with_columns(
            pl.when(pl.col("p") > 0).then(pl.col("y") / pl.col("p")).otherwise(1.0).alias("raw")
        )
        .with_columns(
            (1.0 + (pl.col("raw") - 1.0) * pl.col("n") / (pl.col("n") + shrink)).clip(lo, hi).alias("k")
        )
        .select(list(keys) + ["k"])
    )


def apply_group_calibration(df: pl.DataFrame, y_pred, calib: pl.DataFrame, keys: tuple[str, ...]) -> np.ndarray:
    joined = (
        df.with_row_index("__i")
        .with_columns(pl.Series("__p", np.asarray(y_pred, dtype=float)))
        .join(calib, on=list(keys), how="left")
        .sort("__i")
    )
    return (joined["__p"] * joined["k"].fill_null(1.0)).to_numpy()
