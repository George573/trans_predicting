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
from data_preparation import load_calendar, load_labels, enrich_features, build_submit, CYCLIC_INTRAWEEK
from metrics import wape
from postprocess import fill_route5, adjust_new_year, finalize

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
KEYS = ("route", "weekday")
CALIB_CUT = date(2025, 9, 30)
ART = Path("artifacts")


def build_models() -> dict:
    import catboost as cb
    from harmonic import HarmonicSpec, fit_harmonic, predict_harmonic
    from mlp import MLPSpec, fit_mlp, predict_mlp

    models = {}

    cb_meta = json.load(open(ART / "catboost" / "model_meta.json", encoding="utf-8"))
    cb_params, cb_cats, cb_hl = dict(cb_meta["params"]), cb_meta["cat_features"], cb_meta.get("half_life_days")
    BASE = ["hour", "route", "is_holiday", "is_weekend", "is_short_working_day", "season", "weekday"]

    def make_cb(feats: list[str], seed: int = 0):
        def fit_fn(trn, cut):
            m = cb.CatBoostRegressor(**cb_params, cat_features=[c for c in cb_cats if c in feats],
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

    models["catboost_cyclic"] = make_cb(BASE + CYCLIC_INTRAWEEK)
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


def main() -> None:
    calendar = load_calendar("input/calendar/2025.xml")
    train = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    all_df = pl.concat([train, test]).sort(["date", "hour", "route"])
    submit_df = enrich_features(build_submit(routes=ROUTES), calendar).sort(["date", "hour", "route"])

    folds = [make_folds(all_df, FOLD_BOUNDS_EXT)[i] for i in (2, 3)]
    models = build_models()

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
        k = fit_group_calibration(calib_hold, predict_fn(fit_fn(calib_train, CALIB_CUT), calib_hold), KEYS, shrink=5.0)
        raw = predict_fn(fit_fn(all_df, date(2025, 10, 31)), submit_df)
        final[name] = {"raw": raw, "cal": apply_group_calibration(submit_df, raw, k, KEYS)}
        pl.DataFrame({"route": submit_df["route"], "date": submit_df["date_str"], "hour": submit_df["hour"],
                      "raw": final[name]["raw"], "cal": final[name]["cal"]}).write_parquet(ART / "preds" / f"{name}_submit.parquet")
        print(f"  {name}: прогноз готов, сумма {raw.sum():,.0f} -> с поправкой {final[name]['cal'].sum():,.0f}")

    stamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    for variant in ("raw", "cal"):
        mix = sum(weights[n] * final[n][variant] for n in final)
        df = pl.DataFrame({
            "route": submit_df["route"], "date": submit_df["date_str"],
            "hour": submit_df["hour"], "prediction": mix,
        })
        df = finalize(adjust_new_year(fill_route5(df)))
        path = ART / "submissions" / f"submission_ens-{variant}_{stamp}.csv"
        df.write_csv(path, separator=";")
        print(f"{path.name}: сумма {df['prediction'].sum():,}")

    json.dump({"weights": weights, "oof_wape": score},
              open(ART / f"ensemble_weights_{stamp}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
