"""SHAP-вклады сервисной CatBoost и вклад датированных правил об изменениях маршрутов.

1. SHAP: сервисная модель из бандла, сетка ноября-декабря 2025, вклады в посадках в час.
2. Правила об изменениях маршрутов (events/data/route_changes.csv), два замера:
   - фолд 3 (обучение по 31.08, проверка сентябрь-октябрь): изменения начинаются внутри
     горизонта, прогноз в затронутых ячейках умножается на эффект вида изменения из каталога
     условий сервиса (отмена -96.7%, укорачивание -15.2%, объезд -17.0%);
   - фолд 4 (обучение по 30.09, проверка октябрь): режим уже шёл до отсечки, прогноз в
     затронутых ячейках заменяется профилем режима - медианой наблюдённых посадок того же
     маршрута, дня недели и часа с начала изменения до отсечки.

Выход: artifacts/experiments/dated_rules_shap.json и графики docs/img/shap_*.svg,
docs/img/dated_rules.svg. Запуск из корня: PYTHONPATH=. python3 experiments/dated_rules_shap.py
"""

import json
from datetime import date
from pathlib import Path

import catboost as cb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from backtest import FOLD_BOUNDS_EXT, make_folds
from data_preparation import build_submit, enrich_features, load_calendar, load_labels
from export_bundle import FEATURES, cat_features, fit, predict
from final_ensemble import ROUTES
from metrics import wape

OUT = Path("artifacts/experiments/dated_rules_shap.json")
IMG = Path("docs/img")
BLUE, GOLD, GREEN, RED, GREY = "#2b6cb0", "#b7791f", "#276749", "#c53030", "#718096"
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.unicode_minus": False})

GROUPS = {
    "профиль суток": ["hour", "hour_sin", "hour_cos"],
    "маршрут": ["route"],
    "день недели": ["weekday", "weekday_sin", "weekday_cos"],
    "выходной и праздник": ["is_weekend", "is_holiday", "is_short_working_day"],
    "праздничные блоки": ["days_before_block", "days_after_block", "in_block"],
    "сезон": ["season"],
    "день месяца": ["day_of_month_sin", "day_of_month_cos"],
}
CALENDAR = GROUPS["день недели"] + GROUPS["выходной и праздник"] + GROUPS["праздничные блоки"]

# изменения, действующие в октябре 2025 и опубликованные до отсечки 30.09 (events/data/route_changes.csv)
RULES = [
    {"id": "r21", "name": "50: по выходным не работает", "routes": [50], "start": date(2025, 9, 6), "weekend": True, "hours": (5, 23)},
    {"id": "r20", "name": "7: по выходным укорочен", "routes": [7], "start": date(2025, 9, 6), "weekend": True, "hours": (5, 23)},
    {"id": "r12", "name": "12: по выходным объезд", "routes": [12], "start": date(2025, 6, 7), "weekend": True, "hours": (5, 23)},
    {"id": "r19", "name": "7 и 50: вечером укорочены", "routes": [7, 50], "start": date(2025, 8, 15), "weekend": None, "hours": (23, 23)},
]


EFFECT = {"cancel": 1 - 0.967, "short": 1 - 0.152, "reroute": 1 - 0.170}
KIND_RU = {"cancel": "отмена", "short": "укорачивание", "reroute": "объезд"}


def change_cells(valid: pl.DataFrame, lo: date, hi: date) -> list[dict]:
    """Изменения из route_changes.csv, пересекающие окно проверки, с маской ячеек."""
    ch = pl.read_csv("events/data/route_changes.csv", try_parse_dates=True)
    out = []
    for r in ch.iter_rows(named=True):
        start, end = max(r["date_from"], lo), min(r["date_to"], hi)
        if start > end:
            continue
        routes = [int(x) for x in str(r["routes"]).split()]
        h0, h1 = int(r["hour_from"]), min(int(r["hour_to"]), 24) - 1
        if r["mode"] == "span":
            continue
        expr = (pl.col("route").is_in(routes) & pl.col("date").is_between(start, end)
                & (pl.col("hour").is_between(h0, h1) if h0 <= h1 else (pl.col("hour") >= h0) | (pl.col("hour") <= h1)))
        if r["days"] == "weekend":
            expr &= pl.col("date").dt.weekday() >= 6
        idx = valid.with_row_index("i").filter(expr)["i"].to_numpy()
        if len(idx):
            out.append({"id": r["change_id"], "kind": r["kind"], "routes": routes, "idx": idx, "start": r["date_from"],
                        "name": f"{' и '.join(map(str, routes))}: {KIND_RU[r['kind']]}, {start:%d.%m}-{end:%d.%m}"})
    return out


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def shap_values(model: cb.CatBoostRegressor, df: pl.DataFrame, cats: list[str]) -> np.ndarray:
    frame = df.to_pandas()[FEATURES]
    values = model.get_feature_importance(cb.Pool(frame, cat_features=cats), type="ShapValues")
    return values[:, :-1], values[:, -1]


def rule_mask(df: pl.DataFrame, rule: dict) -> pl.Expr:
    weekend = pl.col("date").dt.weekday() >= 6
    expr = pl.col("route").is_in(rule["routes"]) & pl.col("hour").is_between(*rule["hours"])
    if rule["weekend"] is True:
        expr &= weekend
    elif rule["weekend"] is False:
        expr &= ~weekend
    return expr


def apply_rule(valid: pl.DataFrame, pred: np.ndarray, train: pl.DataFrame, rule: dict) -> np.ndarray:
    hist = (train.filter(rule_mask(train, rule) & (pl.col("date") >= rule["start"]))
            .with_columns(pl.col("date").dt.weekday().alias("wd"))
            .group_by(["route", "wd", "hour"]).agg(pl.col("target").median().alias("regime")))
    cells = (valid.with_row_index("i").with_columns(pl.col("date").dt.weekday().alias("wd"))
             .filter(rule_mask(valid, rule)).join(hist, on=["route", "wd", "hour"], how="inner"))
    out = pred.copy()
    out[cells["i"].to_numpy()] = cells["regime"].to_numpy()
    return out, cells["i"].to_numpy()


def main() -> None:
    meta = json.loads(Path("artifacts/catboost/model_meta.json").read_text(encoding="utf-8"))
    cats = cat_features(meta)
    calendar = load_calendar("input/calendar/2025.xml")
    train = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=ROUTES), calendar)
    test = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=ROUTES), calendar)
    all_df = pl.concat([train, test]).sort(["date", "hour", "route"])
    submit = enrich_features(build_submit(routes=ROUTES), calendar).filter(pl.col("route") != 5).sort(["date", "hour", "route"])

    # 1. SHAP на сервисной модели
    service = cb.CatBoostRegressor()
    service.load_model("artifacts/bundle/catboost/model.cbm")
    shap, base = shap_values(service, submit, cats)
    mean_abs = {f: float(np.abs(shap[:, i]).mean()) for i, f in enumerate(FEATURES)}
    grouped = {g: float(np.abs(shap[:, [FEATURES.index(f) for f in fs]].sum(axis=1)).mean()) for g, fs in GROUPS.items()}
    pred = shap.sum(axis=1) + base
    cal_idx = [FEATURES.index(f) for f in CALENDAR]
    daily = (submit.select("date").with_columns(pl.Series("cal", shap[:, cal_idx].sum(axis=1)), pl.Series("pred", pred))
             .group_by("date").agg(pl.col("cal").sum(), pl.col("pred").sum()).sort("date")
             .with_columns((100 * pl.col("cal") / (pl.col("pred") - pl.col("cal"))).alias("cal_pct")))

    # 2. датированные правила на фолде 4
    fold = make_folds(all_df, FOLD_BOUNDS_EXT)[3]
    model = fit(meta, fold.train, fold.cut)
    base_pred = np.clip(predict(model, fold.valid), 0, None)
    y = fold.valid["target"].to_numpy()
    base_score = 1 - wape(y, base_pred)
    rules, combined = [], base_pred.copy()
    for rule in RULES:
        one, idx = apply_rule(fold.valid, base_pred, fold.train, rule)
        combined, _ = apply_rule(fold.valid, combined, fold.train, rule)
        rules.append({"id": rule["id"], "name": rule["name"], "cells": int(len(idx)),
                      "delta_points": round(100 * ((1 - wape(y, one)) - base_score), 3),
                      "cells_wape_model": round(float(np.abs(y[idx] - base_pred[idx]).sum() / max(y[idx].sum(), 1)), 3),
                      "cells_wape_rule": round(float(np.abs(y[idx] - one[idx]).sum() / max(y[idx].sum(), 1)), 3)})
    all_score = 1 - wape(y, combined)

    f3 = make_folds(all_df, FOLD_BOUNDS_EXT)[2]
    m3 = fit(meta, f3.train, f3.cut)
    p3 = np.clip(predict(m3, f3.valid), 0, None)
    y3 = f3.valid["target"].to_numpy()
    s3 = 1 - wape(y3, p3)
    changes = change_cells(f3.valid, date(2025, 9, 1), date(2025, 10, 31))
    both, rules3 = p3.copy(), []
    for c in changes:
        one = p3.copy()
        one[c["idx"]] *= EFFECT[c["kind"]]
        both[c["idx"]] *= EFFECT[c["kind"]]
        i = c["idx"]
        rules3.append({"id": c["id"], "name": c["name"], "cells": int(len(i)),
                       "delta_points": round(100 * ((1 - wape(y3, one)) - s3), 3),
                       "cells_wape_model": round(float(np.abs(y3[i] - p3[i]).sum() / max(y3[i].sum(), 1)), 3),
                       "cells_wape_rule": round(float(np.abs(y3[i] - one[i]).sum() / max(y3[i].sum(), 1)), 3)})
    s3_all = 1 - wape(y3, both)
    fresh = p3.copy()
    for c in changes:
        if c["start"] > f3.cut:
            fresh[c["idx"]] *= EFFECT[c["kind"]]
    s3_fresh = 1 - wape(y3, fresh)
    for r, c in zip(rules3, changes):
        r["started_after_cut"] = bool(c["start"] > f3.cut)

    result = {
        "shap": {"rows": int(submit.height), "base_value": float(np.mean(base)), "mean_abs_by_feature": mean_abs,
                 "mean_abs_by_group": grouped, "check_max_abs_diff": float(np.abs(pred - predict(service, submit)).max())},
        "rules_fold4": {"cut": fold.cut.isoformat(), "valid": "2025-10-01..2025-10-31",
                        "score_model": round(base_score, 5), "score_all_rules": round(all_score, 5),
                        "delta_all_points": round(100 * (all_score - base_score), 3), "rules": rules},
        "rules_fold3": {"cut": f3.cut.isoformat(), "valid": "2025-09-01..2025-10-31",
                        "effects": EFFECT, "score_model": round(s3, 5), "score_all_rules": round(s3_all, 5),
                        "delta_all_points": round(100 * (s3_all - s3), 3),
                        "score_new_rules": round(s3_fresh, 5), "delta_new_points": round(100 * (s3_fresh - s3), 3),
                        "rules": rules3},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # графики
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    items = sorted(grouped.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, alpha=0.85)
    for i, (_, v) in enumerate(items):
        ax.text(v + 2, i, f"{v:.0f}", va="center", fontsize=8)
    style(ax, "средний |SHAP|, посадок в час", title="Что определяет прогноз CatBoost, ноябрь-декабрь 2025")
    fig.savefig(IMG / "shap_groups.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    items = sorted(mean_abs.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=GREY)
    style(ax, "средний |SHAP|, посадок в час", title="SHAP по признакам")
    fig.savefig(IMG / "shap_features.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    d = daily.to_pandas()
    ax.bar(d["date"], d["cal_pct"], color=[RED if v < 0 else GREEN for v in d["cal_pct"]], width=0.8)
    ax.axhline(0, color="#333", lw=0.8)
    for day, label in [(date(2025, 11, 1), "рабочая суббота"), (date(2025, 11, 2), "2-4 ноября"), (date(2025, 12, 31), "31 декабря")]:
        v = float(d.loc[d["date"].dt.date == day, "cal_pct"].iloc[0])
        ax.annotate(label, (day, v), xytext=(0, -14 if v < 0 else 6), textcoords="offset points", ha="center", fontsize=8)
    style(ax, "", "вклад календаря, % от прогноза без него",
          "Календарные признаки по дням: выходные, праздники, праздничные блоки")
    fig.savefig(IMG / "shap_calendar_days.svg")
    plt.close(fig)

    top = sorted(rules3, key=lambda r: -abs(r["delta_points"]))[:8]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, rows, total, title in [
        (axes[0], top, result["rules_fold3"]["delta_new_points"],
         "Изменение начинается внутри горизонта\nфолд 3: обучение по 31.08, сентябрь-октябрь"),
        (axes[1], rules, result["rules_fold4"]["delta_all_points"],
         "Режим уже шёл до отсечки\nфолд 4: обучение по 30.09, октябрь"),
    ]:
        names = [r["name"] + ("" if r.get("started_after_cut", True) else " (до отсечки)") for r in rows] + [
            "все новые изменения" if rows is top else "все вместе"]
        vals = [r["delta_points"] for r in rows] + [total]
        ax.barh(names[::-1], vals[::-1], color=[GREEN if v > 0 else RED for v in vals[::-1]])
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlim(min(vals + [0]) - 0.3, max(vals + [0]) + 0.3)
        ax.tick_params(axis="y", labelsize=8)
        style(ax, "изменение WAPE-score, пункты", title=title)
    fig.savefig(IMG / "dated_rules.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
