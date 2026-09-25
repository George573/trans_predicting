"""Поправка уровня по дням недели, снятая на последнем однорежимном окне.

Зачем. Отношение выходных к будням дрейфует по сезону: январь 0.492, август 0.640,
сентябрь и октябрь по 0.524. Модель учится на всём годе с равными весами и усредняет этот
дрейф, из-за чего систематически завышает выходные осенью и зимой. Разложение ошибки
показало, что на уровне route x weekday сидит 2.3 пункта WAPE.

Как. Модель переобучается на истории до 30 сентября, прогнозирует октябрь, и по
расхождению с фактом октября считаются коэффициенты route x weekday с усадкой. Октябрь
для модели вне обучения, а по режиму совпадает с ноябрём - это и делает поправку честной.

Замер на фолде 4 (октябрь по янв-сен): 0.1051 -> 0.0997.
Ограничение: окно и цель должны быть в одном режиме. Через границу лето/учебный год
поправка вредит (фолд 3: 0.1096 -> 0.13-0.18), поэтому применять её к прогнозу, который
пересекает такую границу, нельзя.
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from backtest import fit_group_calibration
from data_preparation import load_calendar, load_labels, enrich_features, build_submit
from metrics import wape

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
KEYS = ("route", "weekday")
SHRINK = 5.0
CALIB_CUT = date(2025, 9, 30)      # край обучения для оценки поправки
ART = Path("artifacts")


def load_all() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    calendar = load_calendar("input/calendar/2025.xml")
    train = enrich_features(
        load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test = enrich_features(
        load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    return pl.concat([train, test]).sort(["date", "hour", "route"]), calendar, enrich_features(build_submit(routes=ROUTES), calendar)


def model_fns(name: str):
    """(fit_fn, predict_fn) по сохранённой в artifacts конфигурации модели"""
    meta = json.load(open(ART / name / "model_meta.json", encoding="utf-8"))

    if name == "harmonic":
        from harmonic import HarmonicSpec, fit_harmonic, predict_harmonic
        spec = HarmonicSpec(**meta["spec"])
        return (lambda trn, cut: fit_harmonic(trn, spec, cut=cut),
                lambda m, df: predict_harmonic(m, df, spec))

    if name == "mlp":
        from mlp import MLPSpec, fit_mlp, predict_mlp
        spec = MLPSpec(**{k: v for k, v in meta["spec"].items() if k in MLPSpec.__dataclass_fields__})
        return (lambda trn, cut: fit_mlp(trn, spec, cut), predict_mlp)

    if name == "catboost":
        import catboost as cb
        params = dict(meta["params"])
        feats, hl = meta["features"], meta.get("half_life_days")
        cats = meta["cat_features"]

        def weights(dates, cut):
            if not hl:
                return np.ones(len(dates))
            age = (np.datetime64(cut) - dates.to_numpy().astype("datetime64[D]")).astype("timedelta64[D]")
            return 0.5 ** (age.astype(float) / hl)

        def fit_fn(trn, cut):
            m = cb.CatBoostRegressor(**params, cat_features=cats, verbose=0,
                                     allow_writing_files=False, random_seed=0)
            frame = trn.to_pandas()
            m.fit(frame[feats], frame["target"], sample_weight=weights(trn["date"], cut))
            return m

        return fit_fn, (lambda m, df: np.clip(m.predict(df.to_pandas()[feats]), 0, None))

    raise ValueError(name)


def calibration_for(name: str, all_df: pl.DataFrame) -> pl.DataFrame:
    fit_fn, predict_fn = model_fns(name)
    train = all_df.filter(pl.col("date") <= CALIB_CUT)
    hold = all_df.filter(pl.col("date") > CALIB_CUT)

    model = fit_fn(train, CALIB_CUT)
    pred = predict_fn(model, hold)
    k = fit_group_calibration(hold, pred, KEYS, shrink=SHRINK)

    print(f"  {name}: WAPE октября вне обучения {wape(hold['target'], pred):.4f}, "
          f"коэффициенты {k['k'].min():.3f}-{k['k'].max():.3f}")
    return k


def apply_to_submission(path: Path, calib: pl.DataFrame, submit_df: pl.DataFrame) -> pl.DataFrame:
    sub = pl.read_csv(path, separator=";").with_columns(pl.col("prediction").cast(pl.Float64))
    keyed = (
        sub.join(submit_df.select(["date_str", "route", "hour", "weekday"]).unique(),
                 left_on=["date", "route", "hour"], right_on=["date_str", "route", "hour"], how="left")
        .join(calib, on=list(KEYS), how="left")
        .with_columns((pl.col("prediction") * pl.col("k").fill_null(1.0)).round(0).cast(pl.Int64).alias("prediction"))
        .select(["route", "date", "hour", "prediction"])
        .sort(["route", "date", "hour"])
    )
    assert keyed.height == sub.height
    return keyed


def main() -> None:
    all_df, _, submit_df = load_all()
    print(f"поправка {KEYS}, окно {CALIB_CUT.isoformat()}+, усадка {SHRINK}")

    subs = {p.stem.split("_")[1]: p for p in sorted((ART / "submissions").glob("submission_*.csv"))}
    stamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    for name in ("catboost", "harmonic", "mlp"):
        if name not in subs or not (ART / name / "model_meta.json").exists():
            print(f"  {name}: пропущен (нет сабмита или артефакта)")
            continue
        calib = calibration_for(name, all_df)
        out = apply_to_submission(subs[name], calib, submit_df)
        path = ART / "submissions" / f"submission_{name}-wdcal_{stamp}.csv"
        out.write_csv(path, separator=";")
        print(f"     -> {path.name}, сумма {int(out['prediction'].sum()):,}")


if __name__ == "__main__":
    main()
