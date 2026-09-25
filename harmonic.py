"""Гармоническая регрессия: суточный и недельный профиль через ряды Фурье.

Модель строится отдельно по каждому маршруту. Признаки - синусы и косинусы
периодов 24 ч и 168 ч, взаимодействующие с типом дня, плюс календарные флаги.
Периодическая часть определена для любого момента времени, поэтому прогноз на
ноябрь-декабрь считается по формуле, а не упирается в последний лист, как у
градиентного бустинга.
"""

from dataclasses import dataclass, asdict, field
from datetime import date, timedelta

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ORIGIN = date(2025, 1, 1)
MOSCOW_LAT_RAD = np.deg2rad(55.75)


@dataclass(frozen=True)
class HarmonicSpec:
    """конфигурация модели; ровно эти поля перебирает подбор"""
    k_day: int = 10           # число гармоник периода 24 ч
    k_week: int = 5           # число гармоник периода 168 ч
    daytype_inter: bool = True   # отдельная форма суток для выходных и праздников
    trend: bool = False       # линейный тренд по времени
    daylight: bool = False    # длина светового дня и её квадрат
    kind: str = "poisson"     # poisson (мультипликативные эффекты) или ridge
    alpha: float = 1e-4       # сила L2-регуляризации
    window_days: int | None = None   # обучать только по последним N дням

    def as_dict(self) -> dict:
        return asdict(self)


def daylight_hours(day_of_year: np.ndarray) -> np.ndarray:
    """длина светового дня в Москве, часы.

    Аналитическая астрономическая формула, внешних данных не требует. Полезна тем,
    что значения ноября-декабря (7-9 ч) лежат внутри диапазона января-февраля,
    то есть модель работает в уже виденной области, без экстраполяции.
    """
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    cos_omega = np.clip(-np.tan(MOSCOW_LAT_RAD) * np.tan(decl), -1.0, 1.0)
    return 2 * np.rad2deg(np.arccos(cos_omega)) / 15.0


def _fourier(values: np.ndarray, period: float, k: int) -> np.ndarray:
    terms = []
    for i in range(1, k + 1):
        arg = 2 * np.pi * i * values / period
        terms += [np.sin(arg), np.cos(arg)]
    return np.stack(terms, axis=1)


def design_matrix(df: pl.DataFrame, spec: HarmonicSpec) -> np.ndarray:
    """матрица признаков; порядок колонок зависит только от spec"""
    hour = df["hour"].to_numpy().astype(float)
    weekday = df["weekday"].to_numpy().astype(float)
    hour_of_week = (weekday - 1) * 24 + hour

    is_weekend = df["is_weekend"].to_numpy().astype(float)
    is_holiday = df["is_holiday"].to_numpy().astype(float)
    is_short = df["is_short_working_day"].to_numpy().astype(float)

    blocks: list[np.ndarray] = []

    if spec.trend:
        days = (df["date"].to_numpy().astype("datetime64[D]") - np.datetime64(ORIGIN)).astype(float)
        blocks.append((days / 365.0)[:, None])

    daily = _fourier(hour, 24.0, spec.k_day)
    blocks.append(daily)
    blocks.append(_fourier(hour_of_week, 168.0, spec.k_week))
    blocks.append(np.stack([is_weekend, is_holiday, is_short], axis=1))

    if spec.daytype_inter:
        blocks.append(daily * is_weekend[:, None])
        blocks.append(daily * is_holiday[:, None])

    if spec.daylight:
        dl = daylight_hours(df["day_of_year"].to_numpy().astype(float))
        blocks.append(np.stack([dl, dl ** 2], axis=1))

    return np.hstack(blocks)


def _estimator(spec: HarmonicSpec) -> Pipeline:
    """стандартизация обязательна: блоки тренда и светового дня на порядки больше
    гармоник, и без неё lbfgs в пуассоновской регрессии не сходится"""
    if spec.kind == "poisson":
        core = PoissonRegressor(alpha=spec.alpha, max_iter=5000)
    elif spec.kind == "ridge":
        core = Ridge(alpha=spec.alpha)
    else:
        raise ValueError(f"неизвестный kind: {spec.kind}")
    return make_pipeline(StandardScaler(), core)


def n_features(spec: HarmonicSpec, df: pl.DataFrame) -> int:
    return design_matrix(df.head(1), spec).shape[1]


def fit_harmonic(train_df: pl.DataFrame, spec: HarmonicSpec, cut: date | None = None) -> dict[int, object]:
    """по одной модели на маршрут; маршруты без посадок в историю не попадают"""
    df = train_df
    if spec.window_days is not None:
        edge = cut or df["date"].max()
        df = df.filter(pl.col("date") > edge - timedelta(days=spec.window_days))

    models: dict[int, object] = {}
    for route in sorted(df["route"].unique().to_list()):
        part = df.filter(pl.col("route") == route)
        y = part["target"].to_numpy().astype(float)
        if y.sum() == 0:
            continue
        models[int(route)] = _estimator(spec).fit(design_matrix(part, spec), y)
    return models


def predict_harmonic(models: dict[int, object], df: pl.DataFrame, spec: HarmonicSpec) -> np.ndarray:
    """прогноз в порядке строк df; маршруты без модели получают нули"""
    routes = df["route"].to_numpy()
    out = np.zeros(df.height, dtype=float)
    for route, model in models.items():
        mask = routes == route
        if not mask.any():
            continue
        out[mask] = np.clip(model.predict(design_matrix(df.filter(pl.col("route") == route), spec)), 0, None)
    return out
