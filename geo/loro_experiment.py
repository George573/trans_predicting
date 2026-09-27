"""Холдаут по маршрутам: можно ли прогнозировать линию, которой не было в обучении.

Каждый из девяти маршрутов по очереди убирается из обучения целиком, модель учится на
восьми остальных и прогнозирует выкинутый. Маршрут 5 не участвует: он открылся 16.12.2025,
в разметке его нет, проверить на нём точность нечем.

Два информационных окна, оба считаются одним прогоном:

    A  холодный маршрут в том же периоде: обучение на 8 маршрутах за янв-окт,
       оценка на выкинутом за сен-окт. Изолирует перенос по географии.
    B  холодный маршрут плюс горизонт вперёд: обучение на 8 маршрутах за янв-авг,
       оценка на выкинутом за сен-окт. Осени в обучении нет ни у кого - ровно так же,
       как в исходной задаче нет ноября-декабря.

Внутри одного окна все конфигурации видят одно и то же, поэтому сравнимы между собой.
Разница между A и B - цена горизонта, а не географии.

Пять конфигураций отвечают каждая на свой вопрос:

    floor_*              уровень как у среднего (медианного) из восьми известных маршрутов,
                         география не используется - планка, которую надо перебить
    oracle_level         настоящий уровень выкинутого маршрута подставлен руками; в жизни
                         недостижимо, показывает потолок при идеально угаданном объёме
    geo_level            уровень предсказан из географии - рабочий вариант, ответ на вопрос
    geo_catboost_direct  текущий CatBoost без номера маршрута, но с гео-признаками
    warm_reference       текущий CatBoost с номером маршрута, маршрут в обучении есть -
                         цена отсутствия истории читается как разница с geo_level

    python3 geo/loro_experiment.py

Пишет artifacts/geo/*.csv и artifacts/geo/loro_results.json, печатает те же таблицы.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_preparation import (  # noqa: E402
    CALENDAR_BLOCK_FEATURES, CYCLIC_INTRAWEEK, densify, enrich_features, load_calendar, load_labels,
)
from metrics import wape  # noqa: E402

ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)
HISTORY = (date(2025, 1, 1), date(2025, 10, 31))
EVAL = (date(2025, 9, 1), date(2025, 10, 31))
SETUPS = {"A": date(2025, 10, 31), "B": date(2025, 8, 31)}
HALF_LIFE = 60.0        # как в продакшн-модели
RIDGE_LAMBDA = 1.0      # на стандартизованных признаках; объявлено до замеров
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 1000
SEED = 0

# Три предиктора уровня, объявленные ДО того, как посчитаны метрики. Полная таблица по всем
# гео-признакам печатается тоже - чтобы подгонка, если она есть, была видна, а не спрятана.
DECLARED = ("n_metro_transfers", "length_km", "n_residential_500m")

# Признак, выигравший в таблице LOO ПОСЛЕ того, как она посчитана. Это подгонка отбором, и
# он считается ровно для того, чтобы показать её цену: медианная ошибка у него лучше всех,
# а на маршруте 17, который весит 24% объёма, он промахивается в 2.3 раза.
POSTHOC = ("shared_track_km",)

LEVEL_CONFIGS = ("floor_mean_level", "floor_median_level", "geo_level",
                 "geo_level_posthoc", "oracle_level")
RAW_CONFIGS = ("geo_catboost_direct", "warm_reference")

SHAPE_CATS = ["weekday", "season", "is_holiday", "is_weekend", "is_short_working_day"]
SHAPE_FEATURES = ["hour"] + SHAPE_CATS + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES
BASE_FEATURES = ["hour", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]
ART = ROOT / "artifacts" / "geo"


def catboost_params(alpha: float) -> dict:
    """гиперпараметры продакшн-модели, меняется только квантиль функции потерь"""
    meta = json.load(open(ROOT / "artifacts" / "catboost" / "model_meta.json", encoding="utf-8"))
    return {**meta["params"], "loss_function": f"Quantile:alpha={alpha}"}


def load_panel() -> pl.DataFrame:
    calendar = load_calendar(str(ROOT / "input" / "calendar" / "2025.xml"))
    labels = pl.concat([
        load_labels(str(ROOT / "dataset" / "labels" / f"labels_day_{part}.csv"), dense=False)
        for part in ("train", "test")
    ]).filter(pl.col("route").is_in(ROUTES))
    return enrich_features(densify(labels, HISTORY[0], HISTORY[1], ROUTES), calendar)


def load_geo() -> pl.DataFrame:
    return pl.read_csv(Path(__file__).parent / "data" / "route_geo.csv", separator=";")


def mean_levels(df: pl.DataFrame) -> dict[int, float]:
    """средние посадки в час по маршруту - то, что модель уровня должна угадать"""
    return {
        int(row["route"]): float(row["target"])
        for row in df.group_by("route").agg(pl.col("target").mean()).iter_rows(named=True)
    }


def weights(df: pl.DataFrame, cut: date) -> np.ndarray:
    age = (np.datetime64(cut) - df["date"].to_numpy().astype("datetime64[D]")).astype(float)
    return 0.5 ** (age / HALF_LIFE)


def fit_shape(train: pl.DataFrame, levels: dict[int, float], cut: date, alpha: float):
    """Форма суток и недели, общая для всех маршрутов: цель нормирована на уровень маршрута.

    Номера маршрута среди признаков нет, поэтому модель применима к линии, которой не
    видела. Нормировка уравнивает вклад маршрутов в градиент независимо от их объёма.
    """
    import catboost as cb

    z = train["target"].to_numpy() / np.array([levels[r] for r in train["route"].to_list()])
    model = cb.CatBoostRegressor(**catboost_params(alpha), verbose=0, allow_writing_files=False,
                                 cat_features=SHAPE_CATS, random_seed=SEED)
    model.fit(train.to_pandas()[SHAPE_FEATURES], z, sample_weight=weights(train, cut))
    return model


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA):
    """Гребневая регрессия на стандартизованных признаках, свободный член не штрафуется.

    Обучающих маршрутов восемь, поэтому только такая: бустинг на восьми строках выучит их
    наизусть, а не зависимость.
    """
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    penalty = np.eye(design.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def predict(new: np.ndarray) -> np.ndarray:
        new = np.atleast_2d(np.asarray(new, dtype=float))
        return np.column_stack([np.ones(len(new)), (new - mean) / std]) @ beta

    return predict


def geo_matrix(geo: pl.DataFrame, routes, features) -> np.ndarray:
    return (geo.filter(pl.col("route").is_in(list(routes))).sort("route")
            .select(list(features)).to_numpy().astype(float))


def level_model(geo: pl.DataFrame, levels: dict[int, float], features=DECLARED):
    """уровень маршрута из географии; цель - логарифм средних посадок в час"""
    routes = sorted(levels)
    y = np.log([levels[r] for r in routes])
    return fit_ridge(geo_matrix(geo, routes, features), y)


def predict_level(geo: pl.DataFrame, levels: dict[int, float], route: int, features=DECLARED) -> float:
    return float(np.exp(level_model(geo, levels, features)(geo_matrix(geo, [route], features))[0]))


def level_loo_residuals(geo: pl.DataFrame, levels: dict[int, float], features=DECLARED) -> np.ndarray:
    """собственный разброс модели уровня: LOO внутри обучающих маршрутов, в логарифмах"""
    routes = sorted(levels)
    return np.array([
        np.log(levels[held]) - np.log(predict_level(geo, {r: levels[r] for r in routes if r != held},
                                                    held, features))
        for held in routes
    ])


def fit_raw(panel: pl.DataFrame, geo: pl.DataFrame, held: int, cut: date,
            valid: pl.DataFrame, kind: str, geo_features) -> np.ndarray:
    """Опорные модели на сырой цели, без разделения на уровень и форму.

    direct - текущий CatBoost без номера маршрута, но с гео-признаками: так сделал бы
    почти каждый. warm - текущий CatBoost с номером маршрута, обученный включая выкинутый
    маршрут; цена отсутствия истории читается как разница между ним и geo_level.
    """
    import catboost as cb

    if kind == "warm":
        features = BASE_FEATURES + ["route"] + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES
        cats, train = SHAPE_CATS + ["route"], panel.filter(pl.col("date") <= cut)
    else:
        features = BASE_FEATURES + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES + list(geo_features)
        cats = SHAPE_CATS
        train = panel.filter((pl.col("date") <= cut) & (pl.col("route") != held))
    train, valid = train.join(geo, on="route", how="left"), valid.join(geo, on="route", how="left")

    model = cb.CatBoostRegressor(**catboost_params(0.49), verbose=0, allow_writing_files=False,
                                 cat_features=cats, random_seed=SEED)
    model.fit(train.to_pandas()[features], train["target"].to_numpy().astype(float),
              sample_weight=weights(train, cut))
    return np.clip(model.predict(valid.to_pandas()[features]), 0, None)


WARMUP_DAYS = (0, 7, 14, 31)
WARMUP_TRAIN_END = date(2025, 10, 1)
WARMUP_EVAL = (date(2025, 10, 2), date(2025, 10, 31))


def warmup_curve(panel: pl.DataFrame, geo: pl.DataFrame) -> list[dict]:
    """Сколько собственной истории нужно новой линии, чтобы прогноз стал рабочим.

    Обучение: восемь известных маршрутов до 1 октября плюс первые N дней выкинутого
    маршрута начиная с 1 сентября. Оценка всегда на одном и том же окне, 2-31 октября,
    поэтому строки таблицы сравнимы между собой. N = 0 - это холодный старт: у линии нет
    ни одного дня, и продакшн-архитектура с номером маршрута применяется к категории,
    которой не видела.

    Этой части в исходном плане эксперимента не было; она добавлена потому, что переводит
    ответ из "нельзя" в "нельзя сразу, а с такого-то дня можно".
    """
    evaluation = panel.filter((pl.col("date") >= WARMUP_EVAL[0]) & (pl.col("date") <= WARMUP_EVAL[1]))
    features = BASE_FEATURES + ["route"] + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES
    rows = []
    for days in WARMUP_DAYS:
        own_end = EVAL[0] + timedelta(days=days - 1)
        per_route = []
        for held in ROUTES:
            train = panel.filter(
                ((pl.col("route") != held) & (pl.col("date") <= WARMUP_TRAIN_END))
                | ((pl.col("route") == held) & (pl.col("date") >= EVAL[0]) & (pl.col("date") <= own_end))
            )
            valid = evaluation.filter(pl.col("route") == held).sort(["date", "hour"])
            import catboost as cb
            model = cb.CatBoostRegressor(**catboost_params(0.49), verbose=0,
                                         allow_writing_files=False,
                                         cat_features=SHAPE_CATS + ["route"], random_seed=SEED)
            model.fit(train.to_pandas()[features], train["target"].to_numpy().astype(float),
                      sample_weight=weights(train, WARMUP_TRAIN_END))
            p = np.clip(model.predict(valid.to_pandas()[features]), 0, None)
            per_route.append((valid["target"].to_numpy().astype(float), p))
        y = np.concatenate([a for a, _ in per_route])
        p = np.concatenate([b for _, b in per_route])
        rows.append({
            "own_history_days": days,
            "wape_score": round(max(0.0, 1 - wape(y, p)), 4),
            "worst_route_wape_score": round(min(
                max(0.0, 1 - wape(a, b)) for a, b in per_route), 4),
            "level_bias": round(float(p.sum() / y.sum() - 1), 4),
        })
        print(f"  своей истории {days:>2} дн: WAPE-score {rows[-1]['wape_score']}")
    return rows


def run_setup(panel: pl.DataFrame, geo: pl.DataFrame, setup: str, cut: date) -> dict:
    """девять фолдов одного информационного окна"""
    geo_features = tuple(c for c in geo.columns if c != "route")
    evaluation = panel.filter((pl.col("date") >= EVAL[0]) & (pl.col("date") <= EVAL[1]))
    # окно оценки уровня: в A совпадает с оценочным (уровень остальных маршрутов известен),
    # в B заканчивается на краю обучения - будущего не знает никто
    window = EVAL if setup == "A" else (HISTORY[0], cut)
    levels_all = mean_levels(panel.filter((pl.col("date") >= window[0]) & (pl.col("date") <= window[1])))
    oracle = mean_levels(evaluation)
    train_all = panel.filter(pl.col("date") <= cut)

    folds = []
    for held in ROUTES:
        train = train_all.filter(pl.col("route") != held)
        valid = evaluation.filter(pl.col("route") == held).sort(["date", "hour"])
        y = valid["target"].to_numpy().astype(float)
        levels = {r: v for r, v in levels_all.items() if r != held}

        shape = {
            alpha: np.clip(fit_shape(train, levels, cut, alpha)
                           .predict(valid.to_pandas()[SHAPE_FEATURES]), 0, None)
            for alpha in (0.1, 0.49, 0.9)
        }
        geo_level = predict_level(geo, levels, held)
        level_values = {
            "floor_mean_level": float(np.mean(list(levels.values()))),
            "floor_median_level": float(np.median(list(levels.values()))),
            "geo_level": geo_level,
            "geo_level_posthoc": predict_level(geo, levels, held, POSTHOC),
            "oracle_level": oracle[held],
        }
        preds = {name: level * shape[0.49] for name, level in level_values.items()}
        for name, kind in (("geo_catboost_direct", "direct"), ("warm_reference", "warm")):
            preds[name] = fit_raw(panel, geo, held, cut, valid, kind, geo_features)

        residuals = level_loo_residuals(geo, levels)
        folds.append({
            "route": held,
            "dates": valid["date"].to_numpy().astype("datetime64[D]").astype(int),
            "y": y,
            "shape": shape,
            "levels": levels,
            "level_values": level_values,
            "level_true": oracle[held],
            "preds": preds,
            "interval_lo": np.exp(np.log(geo_level) + np.percentile(residuals, 10)) * shape[0.1],
            "interval_hi": np.exp(np.log(geo_level) + np.percentile(residuals, 90)) * shape[0.9],
        })
        print(f"  {setup}, выкинут маршрут {held:>2}: уровень факт {oracle[held]:7.1f}, "
              f"гео {geo_level:7.1f}")
    return {"folds": folds, "levels_all": levels_all}


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """WAPE-score, смещение по уровню и WAPE-score формы - после подгонки суммы под факт"""
    total_y, total_p = y.sum(), p.sum()
    rescaled = p * total_y / total_p if total_p > 0 else p
    return {
        "wape_score": round(max(0.0, 1 - wape(y, p)), 4),
        "level_bias": round(total_p / total_y - 1, 4),
        "shape_wape_score": round(max(0.0, 1 - wape(y, rescaled)), 4),
    }


def pooled(folds: list[dict], config: str) -> dict[str, float]:
    """Все девять выкинутых маршрутов одной дробью, как считается лидерборд.

    shape_wape_score пересчитывается ВНУТРИ каждого маршрута, а не по общей сумме: иначе
    ошибка уровня одного маршрута маскируется ошибкой другого, и разделение уровня и формы
    перестаёт работать.
    """
    y = np.concatenate([f["y"] for f in folds])
    p = np.concatenate([f["preds"][config] for f in folds])
    rescaled = np.concatenate([
        f["preds"][config] * (f["y"].sum() / f["preds"][config].sum()
                              if f["preds"][config].sum() > 0 else 1.0)
        for f in folds
    ])
    level_errors = [
        abs(f["preds"][config].sum() / f["y"].sum() - 1) for f in folds if f["y"].sum() > 0
    ]
    return {
        "wape_score": round(max(0.0, 1 - wape(y, p)), 4),
        "level_bias": round(float(p.sum() / y.sum() - 1), 4),
        "median_level_error_pct": round(float(np.median(level_errors)) * 100, 1),
        "shape_wape_score": round(max(0.0, 1 - wape(y, rescaled)), 4),
    }


def bootstrap_days(folds: list[dict], config: str, rng: np.random.Generator) -> tuple[float, float]:
    """доверительный интервал WAPE-score бутстрепом по дням, как в factors/check_factors.py"""
    y = np.concatenate([f["y"] for f in folds])
    p = np.concatenate([f["preds"][config] for f in folds])
    days = np.concatenate([f["dates"] for f in folds])
    unique = np.unique(days)
    index = {d: np.flatnonzero(days == d) for d in unique}
    scores = []
    for _ in range(N_BOOTSTRAP):
        take = np.concatenate([index[d] for d in rng.choice(unique, len(unique), replace=True)])
        scores.append(max(0.0, 1 - wape(y[take], p[take])))
    return round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)


def permutation_test(geo: pl.DataFrame, folds: list[dict], rng: np.random.Generator) -> dict:
    """Перемешиваем географию между маршрутами: связь разрушена нарочно.

    Если поддельная модель набирает столько же, гео-признаки информации не несли. Форма
    суток от географии не зависит, поэтому пересчитывается только стадия уровня.
    """
    y = np.concatenate([f["y"] for f in folds])
    real = max(0.0, 1 - wape(y, np.concatenate([f["preds"]["geo_level"] for f in folds])))

    routes = np.array(sorted(geo["route"].to_list()))
    null = []
    for _ in range(N_PERMUTATIONS):
        shuffled = geo.with_columns(pl.Series("route", routes[rng.permutation(len(routes))])).sort("route")
        parts = []
        for fold in folds:
            level = predict_level(shuffled, fold["levels"], fold["route"])
            parts.append(level * fold["shape"][0.49])
        null.append(max(0.0, 1 - wape(y, np.concatenate(parts))))
    null = np.array(null)
    return {
        "real": round(real, 4),
        "null_median": round(float(np.median(null)), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "beaten_share": round(float((null < real).mean()), 3),
    }


def loo_errors(geo: pl.DataFrame, levels: dict[int, float], features=DECLARED) -> np.ndarray:
    """относительная ошибка уровня каждого маршрута, предсказанного по остальным"""
    routes = sorted(levels)
    return np.array([
        abs(predict_level(geo, {r: levels[r] for r in routes if r != held}, held, features)
            / levels[held] - 1)
        for held in routes
    ])


def level_loo_table(geo: pl.DataFrame, levels: dict[int, float]) -> list[dict]:
    """Насколько уровень маршрута предсказуем из географии, признак за признаком.

    LOO по девяти маршрутам: каждый предсказывается по восьми остальным. Печатаются все
    признаки, а не только объявленные заранее - чтобы отбор по результату был виден.
    """
    routes = sorted(levels)
    no_features = np.array([
        abs(np.exp(np.mean(np.log([levels[r] for r in routes if r != held]))) / levels[held] - 1)
        for held in routes
    ])
    candidates = {feature: (feature,) for feature in geo.columns if feature != "route"}
    candidates["ЗАЯВЛЕНО: " + " + ".join(DECLARED)] = DECLARED

    rows = [{"features": "без признаков (среднее по остальным)",
             "median_abs_error_pct": round(float(np.median(no_features)) * 100, 1),
             "max_abs_error_pct": round(float(np.max(no_features)) * 100, 1)}]
    for name, features in candidates.items():
        errors = loo_errors(geo, levels, features)
        rows.append({"features": name,
                     "median_abs_error_pct": round(float(np.median(errors)) * 100, 1),
                     "max_abs_error_pct": round(float(np.max(errors)) * 100, 1)})
    return sorted(rows, key=lambda r: r["median_abs_error_pct"])


def permutation_best_feature(geo: pl.DataFrame, levels: dict[int, float],
                             rng: np.random.Generator) -> dict:
    """Контроль на подгонку отбором: сравниваем ЛУЧШИЙ из 18 признаков с лучшим из 18 на
    перемешанной географии.

    Смотреть на таблицу и брать оттуда победителя - это восемнадцать попыток, а не одна.
    Корректное сравнение требует, чтобы и нулевое распределение строилось так же: внутри
    каждой перестановки тоже берётся минимум по всем признакам.
    """
    features = [c for c in geo.columns if c != "route"]
    routes = np.array(sorted(geo["route"].to_list()))

    def best(table: pl.DataFrame) -> tuple[str, float]:
        scores = {f: float(np.median(loo_errors(table, levels, (f,)))) for f in features}
        winner = min(scores, key=scores.get)
        return winner, scores[winner]

    winner, real = best(geo)
    null = np.array([
        best(geo.with_columns(pl.Series("route", routes[rng.permutation(len(routes))])).sort("route"))[1]
        for _ in range(N_PERMUTATIONS)
    ])
    errors = loo_errors(geo, levels, (winner,))
    return {
        "winner": winner,
        "real_median_error_pct": round(real * 100, 1),
        "real_mean_error_pct": round(float(np.mean(errors)) * 100, 1),
        "real_max_error_pct": round(float(np.max(errors)) * 100, 1),
        "null_median_pct": round(float(np.median(null)) * 100, 1),
        "null_p05_pct": round(float(np.percentile(null, 5)) * 100, 1),
        "beaten_share": round(float((null > real).mean()), 3),
        "monte_carlo_se": round(float(np.sqrt(0.25 / N_PERMUTATIONS)), 3),
    }


def level_sensitivity(folds: list[dict]) -> list[dict]:
    """Сколько стоит ошибка в объёме новой линии: настоящий уровень умножается на k.

    Отвечает на прикладной вопрос - насколько точной должна быть внешняя оценка объёма
    (обследование, аналогия, экспертная прикидка), чтобы прогноз имел смысл.
    """
    y = np.concatenate([f["y"] for f in folds])
    rows = []
    for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        p = np.concatenate([f["level_values"]["oracle_level"] * k * f["shape"][0.49] for f in folds])
        rows.append({"level_error": f"{k:+.0%}".replace("+", "x").replace("%", ""),
                     "k": k, "wape_score": round(max(0.0, 1 - wape(y, p)), 4)})
    return rows


def intervals_table(folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        y, lo, hi = fold["y"], fold["interval_lo"], fold["interval_hi"]
        rows.append({
            "route": fold["route"],
            "cell_coverage_pct": round(float(np.mean((y >= lo) & (y <= hi))) * 100, 1),
            "total_true": int(y.sum()),
            "total_lo": int(lo.sum()),
            "total_hi": int(hi.sum()),
            "total_inside": bool(lo.sum() <= y.sum() <= hi.sum()),
            "width_ratio": round(float(hi.sum() / lo.sum()), 2),
        })
    return rows


def table(rows: list[dict], title: str) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    head = " | ".join(c.ljust(widths[c]) for c in columns)
    body = "\n".join(" | ".join(str(r[c]).ljust(widths[c]) for c in columns) for r in rows)
    return f"\n{title}\n{head}\n{'-' * len(head)}\n{body}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    ART.mkdir(parents=True, exist_ok=True)
    panel, geo = load_panel(), load_geo()
    print(f"панель {panel.height} строк, маршрутов {len(ROUTES)}, "
          f"гео-признаков {len(geo.columns) - 1}")

    results: dict[str, dict] = {"config": {
        "routes": list(ROUTES), "eval": [str(d) for d in EVAL], "declared": list(DECLARED),
        "ridge_lambda": RIDGE_LAMBDA, "half_life_days": HALF_LIFE,
        "n_permutations": N_PERMUTATIONS, "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
    }}

    levels_full = mean_levels(panel)
    loo = level_loo_table(geo, levels_full)
    results["level_loo"] = loo
    print(table(loo, "Предсказуемость уровня маршрута из географии, LOO по девяти маршрутам"))
    results["permutation_best_feature"] = permutation_best_feature(geo, levels_full, rng)
    print(f"\nЛучший признак против перемешивания: {results['permutation_best_feature']}")

    per_route_rows, summary_rows = [], []
    for setup, cut in SETUPS.items():
        print(f"\n=== вариант {setup}, обучение до {cut} ===")
        run = run_setup(panel, geo, setup, cut)
        folds = run["folds"]

        for config in LEVEL_CONFIGS + RAW_CONFIGS:
            lo, hi = bootstrap_days(folds, config, rng)
            summary_rows.append({"setup": setup, "config": config, **pooled(folds, config),
                                 "ci95_lo": lo, "ci95_hi": hi})
            for fold in folds:
                per_route_rows.append({
                    "setup": setup, "config": config, "route": fold["route"],
                    "y_sum": int(fold["y"].sum()),
                    "level_true": round(fold["level_true"], 1),
                    "level_pred": (round(fold["level_values"][config], 1)
                                   if config in fold["level_values"] else None),
                    **score(fold["y"], fold["preds"][config]),
                })

        results.setdefault("permutation", {})[setup] = permutation_test(geo, folds, rng)
        results.setdefault("intervals", {})[setup] = intervals_table(folds)
        results.setdefault("sensitivity", {})[setup] = level_sensitivity(folds)

    print("\n=== разогрев: сколько своей истории нужно новой линии ===")
    results["warmup"] = warmup_curve(panel, geo)
    print(table(results["warmup"], "Оценка на 2-31 октября, обучение на восьми маршрутах "
                                   "плюс первые N дней выкинутого"))

    results["summary"] = summary_rows
    results["per_route"] = per_route_rows

    print(table(summary_rows, "Итог по конфигурациям: все девять выкинутых маршрутов вместе"))
    for setup in SETUPS:
        print(table(results["intervals"][setup],
                    f"Интервалы 80% на гео-модели, вариант {setup}"))
        print(table(results["sensitivity"][setup],
                    f"Чувствительность к ошибке в объёме линии, вариант {setup}"))
        print(f"\nПерестановочный контроль, вариант {setup}: {results['permutation'][setup]}")
    print(table([r for r in per_route_rows if r["config"] in ("geo_level", "warm_reference")],
                "По маршрутам: гео-модель против модели с историей"))

    (ART / "loro_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pl.DataFrame(summary_rows).write_csv(ART / "loro_summary.csv", separator=";")
    pl.DataFrame(per_route_rows).write_csv(ART / "loro_per_route.csv", separator=";")
    pl.DataFrame(loo).write_csv(ART / "level_loo.csv", separator=";")
    print(f"\nзаписано в {ART}")


if __name__ == "__main__":
    main()
