"""Сборка финального прогноза: модели -> веса по фолдам -> калибровка -> постобработка.

Шаги:
1. Каждая модель обучается на фолдах 3 и 4 и прогнозирует их валидацию (out-of-fold).
2. Веса смеси подбираются на этих прогнозах минимизацией WAPE - на фолдах, не на цели.
3. Каждая модель переобучается на всей истории янв-окт и прогнозирует ноя-дек.
4. Поправка route x weekday снимается на октябре вне обучения (см. weekday_calibration).
5. Смесь с подобранными весами, затем постобработка (маршрут 5, новогодний блок).
"""

import json
from datetime import date, datetime
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

from backtest import make_folds, FOLD_BOUNDS_EXT, fit_group_calibration, apply_group_calibration
from data_preparation import (
    load_calendar, load_labels, enrich_features, build_submit, load_weather_features,
    CYCLIC_INTRAWEEK, CALENDAR_BLOCK_FEATURES, EXTERNAL_CAT_FEATURES,
    WEATHER_FEATURES, WEATHER_FORECAST_ONLY, TRAFFIC_FEATURES, SEASON_BRIDGE_FEATURES,
)

# Погодный и транспортный блок: по умолчанию выключен, на фолдах стоит 0.46 и 2.4 пункта
# соответственно (замеры в data_preparation.WEATHER_FEATURES). Включается флагом, чтобы
# оставаться воспроизводимой частью решения и питать корректирующие коэффициенты сервиса.
WEATHER_MODES: dict[str, list[str]] = {
    "off": [],
    "forecast": WEATHER_FORECAST_ONLY,
    "temp": ["temperature_c", "forecast_temperature_c", "temp_anom"],
    "all": WEATHER_FEATURES + ["temp_anom"],
    "all+traffic": WEATHER_FEATURES + ["temp_anom"] + TRAFFIC_FEATURES,
}
from metrics import wape
from postprocess import fill_route5, adjust_new_year, finalize

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)

# Сетка групповой поправки. По умолчанию route x weekday с усадкой 5: на реальной цели она
# стоит 0.38 пункта (0.87855 без поправки против 0.88234 с ней).
#
# Замер сеток на фолде 4 (единственном, где окно калибровки лежит в том же режиме, что цель):
#   без поправки                        0.0968
#   route x weekday, shrink 5           0.0993
#   route x hour, shrink 50             0.1007
#   route x weekday x hour, shrink 20   0.0963
#   route x weekday x hour, shrink 50   0.0964
# Фолд 4 слабое свидетельство: горизонт 31 день против 61 у задачи, а ценность поправки
# растёт с горизонтом. Единственную сетку, обогнавшую там «без поправки», проверили
# сабмитом: route x weekday x hour с усадкой 20 дала 0.88226 против 0.88268 у нашей.
# То есть route x weekday с усадкой 5 - оптимум, а фолд 4 в вопросе калибровки не судья.
#
# Оракул обещал по route x hour 1.99 пункта, но это недостижимый потолок: он знает истину
# валидационного периода, а оценка того же профиля по предыдущему окну не переносится -
# значит отклонение route x hour в основном шум периода, а не устойчивое смещение профиля.
KEYS = ("route", "weekday")
SHRINK = 5.0
CALIB_CUT = date(2025, 9, 30)
ART = Path("artifacts")


def build_models(weather: list[str] | None = None, bridge: bool = False) -> dict:
    import catboost as cb
    from harmonic import HarmonicSpec, fit_harmonic, predict_harmonic
    from mlp import MLPSpec, fit_mlp, predict_mlp

    models = {}

    cb_meta = json.load(open(ART / "catboost" / "model_meta.json", encoding="utf-8"))
    cb_params, cb_cats, cb_hl = dict(cb_meta["params"]), cb_meta["cat_features"], cb_meta.get("half_life_days")
    BASE = ["hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]

    def make_cb(feats: list[str], seed: int = 0):
        cat_f = [c for c in cb_cats + EXTERNAL_CAT_FEATURES if c in feats]

        def fit_fn(trn, cut):
            m = cb.CatBoostRegressor(**cb_params, cat_features=cat_f,
                                     verbose=0, allow_writing_files=False, random_seed=seed)
            frame = trn.to_pandas()
            if cb_hl:
                age = (np.datetime64(cut) - trn["date"].to_numpy().astype("datetime64[D]")).astype("timedelta64[D]")
                w = 0.5 ** (age.astype(float) / cb_hl)
            else:
                w = np.ones(trn.height)
            m.fit(frame[feats], frame["target"], sample_weight=w)
            return m
        return fit_fn, (lambda m, df: np.clip(m.predict(df.to_pandas()[feats]), 0, None))

    # Признаки праздничного блока из календаря: фолд 1 0.1997 -> 0.1704, фолд 3
    # 0.1595 -> 0.1519, фолд 4 без изменений. Лидерборд 0.88234 -> 0.88268.
    #
    # route_change здесь НЕ используется, хотя на фолдах давал 0.90 пункта. Причина в
    # data_preparation.EXTERNAL_FEATURES: на реальной цели он дал 0.88268 -> 0.88146.
    models["catboost_cyclic"] = make_cb(
        BASE + CYCLIC_INTRAWEEK + CALENDAR_BLOCK_FEATURES + list(weather or [])
        + (SEASON_BRIDGE_FEATURES if bridge else [])
    )
    models["catboost_base"] = make_cb(cb_meta["features"])

    h_spec = HarmonicSpec(**json.load(open(ART / "harmonic" / "model_meta.json", encoding="utf-8"))["spec"])
    models["harmonic"] = (lambda trn, cut: fit_harmonic(trn, h_spec, cut=cut),
                          lambda m, df: predict_harmonic(m, df, h_spec))

    m_meta = json.load(open(ART / "mlp" / "model_meta.json", encoding="utf-8"))["spec"]
    m_spec = MLPSpec(**{k: v for k, v in m_meta.items() if k in MLPSpec.__dataclass_fields__})
    m_spec = MLPSpec(**{**m_spec.as_dict(), "n_seeds": 5})
    models["mlp"] = (lambda trn, cut: fit_mlp(trn, m_spec, cut), predict_mlp)
    return models


def fit_weights(preds: dict[str, np.ndarray], y: np.ndarray, step: float = 0.1) -> dict[str, float]:
    """перебор весов на симплексе с шагом step; критерий - WAPE на фолдах"""
    names = list(preds)
    grid = [w for w in product(np.arange(0, 1 + step, step), repeat=len(names)) if abs(sum(w) - 1) < 1e-9]
    best, best_w = float("inf"), None
    for w in grid:
        p = sum(wi * preds[n] for wi, n in zip(w, names))
        s = wape(y, p)
        if s < best:
            best, best_w = s, w
    return dict(zip(names, best_w)), best


def main(keys: tuple[str, ...] = KEYS, shrink: float = SHRINK, tag: str = "",
         weather_mode: str = "off", bridge: bool = False) -> None:
    # Пайплайн сознательно не зависит от events/data: единственная фича оттуда, которая
    # что-то давала на фолдах, на реальной цели вредит (см. data_preparation).
    calendar = load_calendar("input/calendar/2025.xml")
    train = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    all_df = pl.concat([train, test]).sort(["date", "hour", "route"])
    submit_df = enrich_features(build_submit(routes=ROUTES), calendar).sort(["date", "hour", "route"])

    if WEATHER_MODES[weather_mode]:
        wx = load_weather_features(with_traffic=weather_mode == "all+traffic")
        keys_wx = ["route", "date", "hour"]
        all_df = all_df.join(wx, on=keys_wx, how="left", validate="1:1")
        submit_df = submit_df.join(wx, on=keys_wx, how="left", validate="1:1")
        for frame, label in ((all_df, "история"), (submit_df, "прогноз")):
            gaps = {c: frame[c].null_count() for c in WEATHER_MODES[weather_mode]
                    if frame[c].null_count()}
            if gaps:
                print(f"  внимание, пропуски погоды в {label}: {gaps}")

    folds = [make_folds(all_df, FOLD_BOUNDS_EXT)[i] for i in (2, 3)]
    models = build_models(WEATHER_MODES[weather_mode], bridge)

    # 1. out-of-fold прогнозы
    oof: dict[str, list[np.ndarray]] = {n: [] for n in models}
    ys = [f.valid["target"].to_numpy().astype(float) for f in folds]
    print("out-of-fold WAPE:")
    for name, (fit_fn, predict_fn) in models.items():
        row = []
        for f in folds:
            p = predict_fn(fit_fn(f.train, f.cut), f.valid)
            oof[name].append(p)
            row.append(wape(f.valid["target"], p))
        print(f"  {name:<16} фолд3 {row[0]:.4f} | фолд4 {row[1]:.4f}")

    y_all = np.concatenate(ys)
    pooled = {n: np.concatenate(v) for n, v in oof.items()}

    PRED = ART / "preds"
    PRED.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"target": y_all, **pooled}).write_parquet(PRED / "oof_pooled.parquet")
    equal = sum(pooled.values()) / len(pooled)
    print(f"\nравные веса: {wape(y_all, equal):.4f}")
    weights, score = fit_weights(pooled, y_all)
    print(f"подобранные: {wape(y_all, sum(w * pooled[n] for n, w in weights.items())):.4f}  {weights}")

    # 2. прогноз на ноя-дек + поправка по дням недели
    calib_train = all_df.filter(pl.col("date") <= CALIB_CUT)
    calib_hold = all_df.filter(pl.col("date") > CALIB_CUT)

    final: dict[str, np.ndarray] = {}
    for name, (fit_fn, predict_fn) in models.items():
        # считаем все модели: веса могут смениться после проверки устойчивости
        k = fit_group_calibration(calib_hold, predict_fn(fit_fn(calib_train, CALIB_CUT), calib_hold), keys, shrink=shrink)
        raw = predict_fn(fit_fn(all_df, date(2025, 10, 31)), submit_df)
        final[name] = {"raw": raw, "cal": apply_group_calibration(submit_df, raw, k, keys)}
        pl.DataFrame({"route": submit_df["route"], "date": submit_df["date_str"], "hour": submit_df["hour"],
                      "raw": final[name]["raw"], "cal": final[name]["cal"]}).write_parquet(ART / "preds" / f"{name}_submit.parquet")
        print(f"  {name}: прогноз готов, сумма {raw.sum():,.0f} -> с поправкой {final[name]['cal'].sum():,.0f}")

    # Новогодние множители теперь под вопросом: с признаком in_block модель сама
    # оценивает 31 декабря по январскому блоку (97 тыс. против 141 без признака), и
    # ручные 0.92 поверх этого - двойной счёт. Замерить это на фолдах нельзя, в их
    # валидации таких дней нет, поэтому пишем оба варианта и решаем сабмитом.
    stamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    for variant in ("raw", "cal"):
        mix = sum(weights[n] * final[n][variant] for n in final)
        for ny in (True, False):
            df = pl.DataFrame({
                "route": submit_df["route"], "date": submit_df["date_str"],
                "hour": submit_df["hour"], "prediction": mix,
            })
            df = fill_route5(df)
            if ny:
                df = adjust_new_year(df)
            df = finalize(df)
            name = f"ens-{variant}-{'ny' if ny else 'nony'}{tag}"
            path = ART / "submissions" / f"submission_{name}_{stamp}.csv"
            df.write_csv(path, separator=";")
            print(f"{path.name}: сумма {df['prediction'].sum():,}")

    json.dump({"weights": weights, "oof_wape": score, "calib_keys": list(keys), "shrink": shrink},
              open(ART / f"ensemble_weights_{stamp}.json", "w"), indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=",".join(KEYS),
                    help="ключи групповой поправки через запятую, например route,weekday,hour")
    ap.add_argument("--shrink", type=float, default=SHRINK)
    ap.add_argument("--tag", default="", help="суффикс в имени сабмита")
    ap.add_argument("--weather", choices=list(WEATHER_MODES), default="off",
                    help="погодный блок: off, forecast, temp, all, all+traffic")
    ap.add_argument("--bridge", action="store_true",
                    help="добавить длину светового дня - мост через невиданный сезон")
    a = ap.parse_args()
    main(tuple(a.keys.split(",")), a.shrink, a.tag, a.weather, a.bridge)
