"""Графики к отчёту docs/RESEARCH-horizon.md по данным эксперимента holdout_months.

Выход: docs/img/horizon_*.svg - векторные, чтобы читались в отчёте и в вебе.
Запуск: PYTHONPATH=. python3 experiments/horizon_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path("artifacts/experiments")
IMG = Path("docs/img")
SUMMER = (6, 7, 8)
COLORS = {"catboost_cyclic": "#2b6cb0", "harmonic": "#b7791f", "mix_cb70_mlp30": "#276749"}
TITLES = {"catboost_cyclic": "catboost_cyclic", "harmonic": "harmonic", "mix_cb70_mlp30": "смесь 0.7/0.3"}


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_lead(m: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (subset, title) in zip(axes, [
        (m.filter(pl.col("scheme") == "expanding"), "расширяющееся окно: обучение янв..k"),
        (m.filter(pl.col("scheme") == "expanding", ~pl.col("target_month").is_in(SUMMER)),
         "то же, цель только вне лета"),
    ]):
        for model in COLORS:
            d = (subset.filter(pl.col("model") == model)
                 .group_by("lead").agg(pl.col("wape_hour").mean()).sort("lead"))
            ax.plot(d["lead"], d["wape_hour"], "o-", color=COLORS[model], label=TITLES[model], lw=1.8, ms=4)
        style(ax, "лаг, месяцев от края обучения", "WAPE по часам", title)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(IMG / "horizon_lead.svg")
    plt.close(fig)


def fig_distance(m: pl.DataFrame, floor: pl.DataFrame) -> None:
    slide = (m.filter(pl.col("scheme") == "sliding1")
             .join(floor, on=["model", "target_month"], how="left")
             .with_columns((pl.col("wape_hour") - pl.col("floor_hour")).alias("excess")))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for model in COLORS:
        d = (slide.filter(pl.col("model") == model)
             .group_by("dist_signed").agg(pl.col("excess").mean()).sort("dist_signed"))
        ax.plot(d["dist_signed"], d["excess"], "o-", color=COLORS[model], label=TITLES[model], lw=1.8, ms=4)
    ax.axvline(0, color="#444", lw=0.8, ls="--")
    ax.axhline(0, color="#444", lw=0.8)
    style(ax, "расстояние до обучающего месяца (< 0 - прогноз назад)",
          "превышение над потолком интерполяции", "обучение на одном месяце")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(IMG / "horizon_distance.svg")
    plt.close(fig)


def fig_aggregation(m: pl.DataFrame) -> None:
    d = (m.filter(pl.col("scheme") == "expanding", pl.col("model") == "mix_cb70_mlp30")
         .group_by("lead").agg(pl.col("wape_hour").mean(), pl.col("wape_day").mean(),
                               pl.col("wape_month_route").mean(),
                               pl.col("wape_hour_level_route").mean()).sort("lead"))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for col, label, color in (("wape_hour", "час x маршрут", "#2b6cb0"),
                              ("wape_day", "день x маршрут", "#276749"),
                              ("wape_month_route", "месяц x маршрут", "#b7791f"),
                              ("wape_hour_level_route", "час, уровень задан извне", "#9b2c2c")):
        ax.plot(d["lead"], d[col], "o-", label=label, color=color, lw=1.8, ms=4)
    style(ax, "лаг, месяцев", "WAPE", "уровень агрегации и цена незнания уровня")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(IMG / "horizon_aggregation.svg")
    plt.close(fig)


def fig_intervals(q: pl.DataFrame) -> None:
    d = q.group_by("lead").agg(pl.col("width_rel").mean(), pl.col("coverage_hour").mean(),
                               pl.col("coverage_day").mean()).sort("lead")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(d["lead"], d["width_rel"], "o-", color="#2b6cb0", lw=1.8, ms=4, label="ширина интервала 10-90")
    style(ax, "лаг, месяцев", "ширина, доля фактического объёма", "квантильный интервал: ширина против покрытия")
    ax2 = ax.twinx()
    ax2.plot(d["lead"], d["coverage_hour"], "s--", color="#9b2c2c", lw=1.5, ms=4, label="покрытие, часы")
    ax2.plot(d["lead"], d["coverage_day"], "^--", color="#b7791f", lw=1.5, ms=4, label="покрытие, сутки")
    ax2.axhline(0.8, color="#444", lw=0.8, ls=":")
    ax2.set_ylabel("доля фактов внутри интервала"); ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()[:2]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=9, loc="center right")
    fig.savefig(IMG / "horizon_intervals.svg")
    plt.close(fig)


def _level_index() -> dict[int, float]:
    from datetime import date
    from data_preparation import load_calendar, load_labels, enrich_features
    routes = (1, 7, 11, 12, 17, 25, 26, 28, 50)
    cal = load_calendar("input/calendar/2025.xml")
    a = enrich_features(load_labels("dataset/labels/labels_day_train.csv", date(2025, 1, 1), date(2025, 8, 31), routes=routes), cal)
    b = enrich_features(load_labels("dataset/labels/labels_day_test.csv", date(2025, 9, 1), date(2025, 10, 31), routes=routes), cal)
    df = pl.concat([a, b]).with_columns(pl.col("date").dt.month().alias("m"))
    daily = df.group_by(["m", "date"]).agg(pl.col("target").sum().alias("v"),
                                           (pl.col("is_weekend") | pl.col("is_holiday")).any().alias("off"))
    lvl = daily.filter(~pl.col("off")).group_by("m").agg(pl.col("v").mean().alias("level")).sort("m")
    mean = lvl["level"].mean()
    return {int(k): float(v) / float(mean) for k, v in zip(lvl["m"], lvl["level"])}


def fig_level_gap(m: pl.DataFrame, floor: pl.DataFrame) -> None:
    """главный график: ошибку определяет разрыв уровня, а не расстояние"""
    lvl = _level_index()

    def lr_best(row) -> float:
        train = [int(x) for x in row["train_months"].split(",")]
        return float(min(abs(np.log(lvl[row["target_month"]] / lvl[s])) for s in train))

    sl = (m.filter(pl.col("scheme").str.starts_with("sliding"))
          .with_columns(pl.struct(["train_months", "target_month"])
                        .map_elements(lr_best, return_dtype=pl.Float64).alias("lr"))
          .join(floor, on=["model", "target_month"], how="left")
          .with_columns((pl.col("wape_hour") - pl.col("floor_hour")).alias("excess"),
                        pl.col("dist_signed").abs().alias("dist_abs")))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    d = sl.filter(pl.col("model") == "mix_cb70_mlp30")

    axes[0].scatter(d["dist_abs"], d["excess"], s=22, color="#2b6cb0", alpha=0.55, edgecolor="none")
    x = d["dist_abs"].to_numpy().astype(float); y = d["excess"].to_numpy()
    coef = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(coef, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    axes[0].plot(np.sort(x), np.polyval(coef, np.sort(x)), color="#9b2c2c", lw=1.8)
    style(axes[0], "расстояние до обучающего месяца, месяцев", "превышение над потолком",
          f"расстояние объясняет почти ничего: R2 = {r2:.2f}")

    axes[1].scatter(d["lr"] * 100, d["excess"], s=22, color="#276749", alpha=0.55, edgecolor="none")
    x2 = d["lr"].to_numpy() * 100
    coef2 = np.polyfit(x2, y, 1)
    r2b = 1 - ((y - np.polyval(coef2, x2)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    axes[1].plot(np.sort(x2), np.polyval(coef2, np.sort(x2)), color="#9b2c2c", lw=1.8)
    style(axes[1], "разрыв уровня до ближайшего по уровню месяца, %", "превышение над потолком",
          f"разрыв уровня объясняет две трети: R2 = {r2b:.2f}")
    fig.savefig(IMG / "horizon_level_gap.svg")
    plt.close(fig)

    # разбивка: расстояние при совпадающем и при расходящемся уровне
    fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)
    for cond, label, color in ((pl.col("lr") < 0.05, "уровень совпадает (< 5%)", "#276749"),
                               (pl.col("lr") > 0.12, "уровень расходится (> 12%)", "#9b2c2c")):
        g = (sl.filter(pl.col("model") == "mix_cb70_mlp30", cond)
             .group_by("dist_abs").agg(pl.col("excess").mean()).sort("dist_abs"))
        ax.plot(g["dist_abs"], g["excess"], "o-", color=color, label=label, lw=1.8, ms=5)
    ax.axhline(0, color="#444", lw=0.8)
    style(ax, "расстояние до обучающего месяца, месяцев", "превышение над потолком",
          "одно и то же расстояние, разная цена")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(IMG / "horizon_distance_split.svg")
    plt.close(fig)


def fig_2026(m: pl.DataFrame) -> None:
    """проекция по месяцам 2026 года и чувствительность к годовому тренду"""
    f = m.filter(pl.col("scheme") == "lomo_flat", pl.col("model") == "mix_cb70_mlp30")
    floor = dict(zip(f["target_month"].to_list(), f["wape_hour"].to_list()))
    PEN = 0.0255
    NOV_DEC = 0.109
    months = list(range(1, 13))
    base = [floor.get(mo, NOV_DEC) for mo in months]
    proj = [b + PEN for b in base]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.bar(months, base, color="#2b6cb0", label="потолок: месяц окружён обучением")
    ax.bar(months, [p - b for p, b in zip(proj, base)], bottom=base, color="#90cdf4",
           label="плата за горизонт при совпадающем уровне")
    for mo in (11, 12):
        ax.patches[mo - 1].set_hatch("//")
    ax.set_xticks(months)
    ax.set_xticklabels(["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"], fontsize=8)
    style(ax, "", "WAPE по часам", "ожидаемая точность по месяцам 2026 года")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(11.5, 0.02, "штриховка -\nноя-дек взяты\nпо факту 2025", fontsize=7.5, color="#4a5568", ha="center")

    ax = axes[1]
    weighted = 0.145
    trends = np.arange(0, 0.11, 0.005)
    ax.plot(trends * 100, [1 - (weighted + g) for g in trends], color="#9b2c2c", lw=2)
    for thr, lbl in ((0.88, "порог 5 баллов"), (0.80, "порог 4 балла"), (0.70, "порог 3 балла")):
        ax.axhline(thr, color="#718096", lw=0.8, ls=":")
        ax.text(10.2, thr + 0.004, lbl, fontsize=8, color="#4a5568", ha="right")
    style(ax, "неизвестный годовой тренд спроса, %", "WAPE-score за год",
          "чем стоит незнание тренда: сдвиг уровня входит в метрику целиком")
    fig.savefig(IMG / "horizon_2026.svg")
    plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    m = pl.read_parquet(OUT / "holdout_months_months.parquet")
    q = pl.read_parquet(OUT / "holdout_months_quantiles.parquet")
    floor = (m.filter(pl.col("scheme") == "lomo_flat")
             .select(["model", "target_month", "wape_hour"]).rename({"wape_hour": "floor_hour"}))
    fig_lead(m); fig_distance(m, floor); fig_aggregation(m); fig_intervals(q)
    fig_level_gap(m, floor); fig_2026(m)
    print("графики записаны в", IMG)


if __name__ == "__main__":
    main()
