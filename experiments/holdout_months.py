"""Эксперимент: деградация точности при обеднении обучающей выборки по месяцам.

Вопрос: как растёт ошибка с горизонтом и с сокращением обучающей выборки, и что из
этого следует для прогноза на год вперёд, где проверить нечего.

Четыре схемы разбиения на данных январь-октябрь 2025 (10 месяцев):

expanding  обучение янв..k, прогноз k+1..окт          - горизонт при растущем обучении
sliding1   обучение ровно один месяц, прогноз все остальные  - чистое расстояние
sliding2   обучение два подряд идущих месяца, прогноз остальные
sliding3   обучение три подряд идущих месяца, прогноз остальные
lomo       обучение девять месяцев, прогноз один пропущенный - потолок интерполяции

Метрика считается для каждого целевого месяца отдельно на трёх уровнях агрегации:
почасовая сетка (как на лидерборде), суточные итоги по маршруту, месячный итог по
маршруту. Три конфигурации модели: catboost_cyclic, harmonic и продовая смесь
0.7 catboost_cyclic + 0.3 mlp.

Запуск:  PYTHONPATH=. python3 experiments/holdout_months.py
Выход:   artifacts/experiments/holdout_months_{months,routes,quantiles,preds}.parquet
"""

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from data_preparation import (
    load_calendar, load_labels, enrich_features,
    CYCLIC_INTRAWEEK, CALENDAR_BLOCK_FEATURES,
)
from metrics import wape

ART = Path("artifacts")
OUT = ART / "experiments"
MONTHS = list(range(1, 11))          # январь-октябрь 2025
MIX = {"catboost_cyclic": 0.7, "mlp": 0.3}

# Маршрут 5: ноль успешных валидаций во всей истории, в проде он заполняется отдельно
# (postprocess.fill_route5). В обучающей сетке оставляем как в проде, из замера исключаем -
# иначе метрика измеряла бы постобработку, а не модель.
ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
EVAL_ROUTES = tuple(r for r in ROUTES if r != 5)

BASE_FEATURES = ["hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]


# ---------------------------------------------------------------- модели

def cb_factory(half_life: float | None, alpha: float | None = None):
    """CatBoost как в проде, но с параметризованным периодом полураспада весов.

    half_life=None - равные веса. Нужен для схемы lomo: там обучение лежит по обе стороны
    от цели, и производственное затухание от края обучения (край - всегда октябрь) свело бы
    январь к весу 0.03, то есть потолок интерполяции мерился бы на почти пустых данных.
    """
    import catboost as cb

    meta = json.load(open(ART / "catboost" / "model_meta.json", encoding="utf-8"))
    params = dict(meta["params"])
    if alpha is not None:
        params["loss_function"] = f"Quantile:alpha={alpha}"
    feats = BASE_FEATURES + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES
    cats = [c for c in meta["cat_features"] if c in feats]

    def fit_fn(trn: pl.DataFrame, cut: date):
        m = cb.CatBoostRegressor(**params, cat_features=cats, verbose=0,
                                 allow_writing_files=False, random_seed=0)
        frame = trn.to_pandas()
        if half_life:
            age = (np.datetime64(cut) - trn["date"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]")
            w = 0.5 ** (age.astype(float) / half_life)
        else:
            w = np.ones(trn.height)
        m.fit(frame[feats], frame["target"], sample_weight=w)
        return m

    return fit_fn, (lambda m, df: np.clip(m.predict(df.to_pandas()[feats]), 0, None))


def harmonic_factory():
    from harmonic import HarmonicSpec, fit_harmonic, predict_harmonic

    spec = HarmonicSpec(**json.load(open(ART / "harmonic" / "model_meta.json", encoding="utf-8"))["spec"])
    return (lambda trn, cut: fit_harmonic(trn, spec, cut=cut),
            lambda m, df: predict_harmonic(m, df, spec))


def mlp_factory():
    from mlp import MLPSpec, fit_mlp, predict_mlp

    meta = json.load(open(ART / "mlp" / "model_meta.json", encoding="utf-8"))["spec"]
    spec = MLPSpec(**{k: v for k, v in meta.items() if k in MLPSpec.__dataclass_fields__})
    return (lambda trn, cut: fit_mlp(trn, spec, cut), predict_mlp)


# ---------------------------------------------------------------- сетка конфигураций

def configs() -> list[dict]:
    out: list[dict] = []

    for k in range(1, 10):                      # expanding: янв..k -> k+1..окт
        out.append({"scheme": "expanding", "train": list(range(1, k + 1)),
                    "eval": list(range(k + 1, 11)), "half_life": 60.0})

    for width in (1, 2, 3):                     # sliding: окно фиксированной ширины
        for start in range(1, 12 - width):
            train = list(range(start, start + width))
            out.append({"scheme": f"sliding{width}", "train": train,
                        "eval": [m for m in MONTHS if m not in train], "half_life": 60.0})

    for m in MONTHS:                            # lomo: девять месяцев -> пропущенный
        train = [x for x in MONTHS if x != m]
        out.append({"scheme": "lomo", "train": train, "eval": [m], "half_life": 60.0})
        out.append({"scheme": "lomo_flat", "train": train, "eval": [m], "half_life": None})

    return out


def geometry(train: list[int], target: int) -> dict:
    """положение целевого месяца относительно обучающего множества"""
    nearest = min(train, key=lambda s: abs(target - s))
    inside = min(train) < target < max(train)
    return {
        "n_train_months": len(train),
        "train_lo": min(train),
        "train_hi": max(train),
        "lead": target - max(train),                       # вперёд от края обучения
        "dist_signed": target - nearest,                   # до ближайшего обучающего месяца
        "dist_abs": abs(target - nearest),
        "direction": "inside" if inside else ("forward" if target > max(train) else "backward"),
    }


# ---------------------------------------------------------------- метрики

def month_metrics(df: pl.DataFrame, pred: np.ndarray) -> dict:
    """WAPE на трёх уровнях агрегации, сдвиг уровня и ошибка формы.

    Разделение уровня и формы - главная величина для годового горизонта. Прогноз,
    перемасштабированный на истинный месячный объём (сначала по городу, затем по каждому
    маршруту), теряет всю ошибку уровня и сохраняет только ошибку профиля. Разность между
    сырой и перемасштабированной метрикой - это та часть ошибки, которую в сервисе можно
    снять корректирующим коэффициентом, если уровень задан извне сценарием.
    """
    frame = df.select(["route", "date", "target"]).with_columns(pl.Series("p", pred))
    y, p = frame["target"].to_numpy().astype(float), frame["p"].to_numpy()

    daily = frame.group_by(["route", "date"]).agg(pl.col("target").sum(), pl.col("p").sum())
    monthly = frame.group_by("route").agg(pl.col("target").sum(), pl.col("p").sum())

    scaled_city = p * (y.sum() / p.sum()) if p.sum() > 0 else p
    routes = frame["route"].to_numpy()
    scaled_route = p.copy()
    for route in np.unique(routes):
        mask = routes == route
        if p[mask].sum() > 0:
            scaled_route[mask] = p[mask] * (y[mask].sum() / p[mask].sum())

    return {
        "wape_hour": wape(y, p),
        "wape_day": wape(daily["target"], daily["p"]),
        "wape_month_route": wape(monthly["target"], monthly["p"]),
        "wape_hour_level_city": wape(y, scaled_city),
        "wape_hour_level_route": wape(y, scaled_route),
        "bias_city": float(p.sum() / y.sum() - 1.0) if y.sum() > 0 else float("nan"),
        "y_sum": float(y.sum()),
        "p_sum": float(p.sum()),
    }


def route_metrics(df: pl.DataFrame, pred: np.ndarray) -> pl.DataFrame:
    return (
        df.select(["route", "target"]).with_columns(pl.Series("p", pred))
        .group_by("route")
        .agg(pl.col("target").sum().alias("y_sum"), pl.col("p").sum().alias("p_sum"),
             (pl.col("target") - pl.col("p")).abs().sum().alias("err_sum"))
        .with_columns((pl.col("err_sum") / pl.col("y_sum")).alias("wape_local"),
                      (pl.col("p_sum") / pl.col("y_sum") - 1).alias("bias"))
    )


# ---------------------------------------------------------------- прогон

def main() -> None:
    calendar = load_calendar("input/calendar/2025.xml")
    train_csv = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test_csv = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    all_df = pl.concat([train_csv, test_csv]).sort(["date", "hour", "route"]).with_columns(
        pl.col("date").dt.month().alias("m")
    )
    print(f"данные: {all_df.height} строк, {all_df['date'].min()} - {all_df['date'].max()}")

    harmonic = harmonic_factory()
    mlp = mlp_factory()

    month_rows: list[dict] = []
    pred_rows: list[pl.DataFrame] = []
    route_rows: list[pl.DataFrame] = []
    q_rows: list[dict] = []
    cfgs = configs()
    t_start = time.time()

    for i, cfg in enumerate(cfgs, 1):
        trn = all_df.filter(pl.col("m").is_in(cfg["train"]))
        cut = trn["date"].max()
        ev = all_df.filter(pl.col("m").is_in(cfg["eval"]) & pl.col("route").is_in(EVAL_ROUTES))

        cb_fit, cb_pred = cb_factory(cfg["half_life"])
        fitted = {
            "catboost_cyclic": (cb_pred, cb_fit(trn, cut)),
            "harmonic": (harmonic[1], harmonic[0](trn, cut)),
            "mlp": (mlp[1], mlp[0](trn, cut)),
        }
        preds = {name: fn(model, ev) for name, (fn, model) in fitted.items()}
        preds["mix_cb70_mlp30"] = sum(w * preds[n] for n, w in MIX.items())

        # Прогнозы сохраняем целиком: любой новый вопрос к эксперименту (форма суток,
        # покрытие, разбивка по часам) считается по ним без повторного обучения.
        cfg_id = f"{cfg['scheme']}:{','.join(map(str, cfg['train']))}"
        pred_rows.append(ev.select(["route", "date", "hour", "target", "m"]).with_columns(
            pl.lit(cfg_id).alias("config"),
            *[pl.Series(name, preds[name].astype(np.float32))
              for name in ("catboost_cyclic", "harmonic", "mlp", "mix_cb70_mlp30")]))

        for target in cfg["eval"]:
            mask = (ev["m"] == target).to_numpy()
            part = ev.filter(pl.col("m") == target)
            geo = geometry(cfg["train"], target)
            for name in ("catboost_cyclic", "harmonic", "mix_cb70_mlp30"):
                p = preds[name][mask]
                month_rows.append({"scheme": cfg["scheme"], "model": name,
                                   "train_months": ",".join(map(str, cfg["train"])),
                                   "target_month": target, **geo, **month_metrics(part, p)})
                route_rows.append(route_metrics(part, p).with_columns(
                    scheme=pl.lit(cfg["scheme"]), model=pl.lit(name),
                    target_month=pl.lit(target), lead=pl.lit(geo["lead"]),
                    dist_signed=pl.lit(geo["dist_signed"]),
                    n_train_months=pl.lit(geo["n_train_months"])))

        # Квантильные модели - только на расширяющемся окне: там горизонт определён
        # однозначно, и именно эта схема воспроизводит реальный сценарий.
        if cfg["scheme"] == "expanding":
            band = {}
            for alpha in (0.1, 0.9):
                f, p_fn = cb_factory(cfg["half_life"], alpha=alpha)
                band[alpha] = p_fn(f(trn, cut), ev)
            for target in cfg["eval"]:
                mask = (ev["m"] == target).to_numpy()
                part = ev.filter(pl.col("m") == target)
                y = part["target"].to_numpy().astype(float)
                lo, hi = band[0.1][mask], band[0.9][mask]
                daily = (part.select(["route", "date", "target"])
                         .with_columns(pl.Series("lo", lo), pl.Series("hi", hi))
                         .group_by(["route", "date"])
                         .agg(pl.col("target").sum(), pl.col("lo").sum(), pl.col("hi").sum()))
                q_rows.append({
                    "train_months": ",".join(map(str, cfg["train"])),
                    "target_month": target, **geometry(cfg["train"], target),
                    "width_rel": float((hi - lo).sum() / y.sum()),
                    "coverage_hour": float(((y >= lo) & (y <= hi)).mean()),
                    "coverage_day": float(((daily["target"] >= daily["lo"]) & (daily["target"] <= daily["hi"])).mean()),
                    "lo_sum": float(lo.sum()), "hi_sum": float(hi.sum()), "y_sum": float(y.sum()),
                })

        done = time.time() - t_start
        print(f"[{i:>2}/{len(cfgs)}] {cfg['scheme']:<10} обучение {cfg['train']} "
              f"-> {len(cfg['eval'])} мес, {done:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(month_rows).write_parquet(OUT / "holdout_months_months.parquet")
    pl.concat(route_rows).write_parquet(OUT / "holdout_months_routes.parquet")
    pl.DataFrame(q_rows).write_parquet(OUT / "holdout_months_quantiles.parquet")
    pl.concat(pred_rows).write_parquet(OUT / "holdout_months_preds.parquet", compression="zstd")
    print(f"готово за {time.time() - t_start:.0f}s, {len(month_rows)} строк замеров")


if __name__ == "__main__":
    main()
