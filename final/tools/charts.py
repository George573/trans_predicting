"""Графики для документов final/docs по числам, зафиксированным в отчётах.

Выход: final/docs/img/*.svg. Запуск из корня репозитория: python3 final/tools/charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
BLUE, GOLD, GREEN, RED, GREY = "#2b6cb0", "#b7791f", "#276749", "#c53030", "#718096"
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.unicode_minus": False})


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.savefig(IMG / f"{name}.svg")
    plt.close(fig)


def score_progress():
    ml = [
        ("гармоники", 0.86711), ("CatBoost", 0.87087), ("MLP", 0.87163),
        ("CatBoost +\nгармоники", 0.87411), ("0.7 CatBoost\n+ 0.3 MLP", 0.87855),
        ("+ поправка\nмаршрут x\nдень недели", 0.88234), ("+ праздничные\nблоки,\nНовый год", 0.88268),
        ("+ погода", 0.88466),
    ]
    cnn = [("+ флаги\nкалендаря", 0.88158), ("+ официальный\nкалендарь", 0.88409),
           ("+ события", 0.88499),
           ("финальная\nверсия", 0.88646)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [8, 4.6]}, sharey=True)
    for ax, rows, color, title in [
        (axes[0], ml, BLUE, "Classical ML: путь к 0.88466"),
        (axes[1], cnn, GREEN, "CNN: путь к 0.88646"),
    ]:
        x = np.arange(len(rows))
        vals = [v for _, v in rows]
        ax.bar(x, vals, color=color, alpha=0.85, width=0.65)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.0006, f"{v:.5f}", ha="center", fontsize=8)
        ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
        ax.axhline(0.88, color=RED, lw=1, ls="--")
        ax.text(-0.45, 0.8803, "порог 0.88", color=RED, fontsize=8, ha="left")
        style(ax, title=title)
    axes[0].set_ylim(0.862, 0.888)
    axes[0].set_ylabel("WAPE-score, ноябрь-декабрь 2025")
    save(fig, "score_progress")


def cnn_training():
    val = [0.2934, 0.3762, 0.2073, 0.1823, 0.1773, 0.1876, 0.1474, 0.1395, 0.1470, 0.1559,
           0.1409, 0.1240, 0.1310, 0.1327, 0.1374, 0.1308, 0.1263, 0.1186, 0.1322, 0.1274,
           0.1240, 0.1264, 0.1313, 0.1186, 0.1251, 0.1267, 0.1215, 0.1255, 0.1397, 0.1297]
    fin = [0.2555, 0.2845, 0.2061, 0.2086, 0.1587, 0.1552, 0.1318, 0.1252, 0.1278, 0.1369,
           0.1636, 0.1199, 0.1464, 0.1142, 0.1364, 0.1169, 0.1084, 0.1076, 0.1169, 0.1097,
           0.1124, 0.1079, 0.1016, 0.1173, 0.1099, 0.1106, 0.1058, 0.1066, 0.1084]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    ax.plot(range(1, len(val) + 1), val, "o-", ms=3, color=BLUE,
            label="валидация: обучение до 1 сен, окна 14 дней сен-окт")
    ax.plot(range(1, len(fin) + 1), fin, "o-", ms=3, color=GREEN,
            label="финальное обучение: до 1 окт, остановка по октябрю")
    ax.axhline(0.1398, color=GOLD, ls="--", lw=1)
    ax.text(30, 0.142, "один контекст на 1 сен, 61 день: WAPE 0.1398", color=GOLD,
            fontsize=8, ha="right")
    ax.set_ylim(0.09, 0.30)
    style(ax, "эпоха", "WAPE на валидации", "Обучение CNN")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "cnn_training")


def backtest_vs_target():
    names = ["изменения маршрутов", "погодный блок"]
    folds = [+0.81, -0.39]
    target = [-0.122, +0.198]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.bar(x - 0.18, folds, 0.34, color=GREY, label="бэктест, фолды 3-4")
    ax.bar(x + 0.18, target, 0.34, color=BLUE, label="реальная цель, ноябрь-декабрь")
    for xi, f, t in zip(x, folds, target):
        ax.text(xi - 0.18, f + (0.03 if f > 0 else -0.06), f"{f:+.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.18, t + (0.03 if t > 0 else -0.08), f"{t:+.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylim(-0.5, 0.95)
    style(ax, ylabel="выигрыш, пункты WAPE-score",
          title="Бэктест против реальности: знак эффекта меняется")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "backtest_vs_target")


def ml_steps():
    steps = [("ансамбль\nбез поправок", 0.87855), ("групповая\nпоправка", 0.88234),
             ("праздничные блоки\nи Новый год", 0.88268), ("погода", 0.88466)]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    prev = steps[0][1]
    ax.bar(0, prev, color=GREY, width=0.6)
    ax.text(0, prev + 0.0004, f"{prev:.5f}", ha="center", fontsize=8)
    for i, (_, v) in enumerate(steps[1:], 1):
        ax.bar(i, v - prev, bottom=prev, color=GREEN, width=0.6)
        ax.text(i, v + 0.0004, f"+{(v - prev) * 100:.3f} п.\n{v:.5f}", ha="center", fontsize=8)
        prev = v
    ax.set_xticks(range(len(steps)), [n for n, _ in steps], fontsize=8)
    ax.set_ylim(0.876, 0.887)
    style(ax, ylabel="WAPE-score", title="Вклад ступеней обвязки ансамбля")
    save(fig, "ml_steps")


def geo_levels():
    routes = ["12", "26", "11", "7", "50", "1", "28", "17", "25"]
    fact = [1355, 745, 1303, 923, 791, 776, 411, 2092, 317]
    geo = [1534, 861, 1518, 1071, 611, 537, 658, 629, 784]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    ax.plot([0, 2300], [0, 2300], color=GREY, lw=1)
    ax.fill_between([0, 2300], [0, 2300 * 0.75], [0, 2300 * 1.25], color=GREY, alpha=0.12,
                    label="±25%")
    ax.scatter(fact, geo, color=BLUE, zorder=3)
    for r, f, g in zip(routes, fact, geo):
        ax.annotate(r, (f, g), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 2300)
    ax.set_ylim(0, 2300)
    style(ax, "факт, посадок в час", "прогноз по географии",
          "Объём новой линии по географии")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "geo_levels")


def geo_configs():
    rows = [("CatBoost с\nгео-признаками", 0.4920, GREY), ("средний\nуровень", 0.5051, GREY),
            ("гео-уровень", 0.5841, BLUE), ("оракульный\nуровень", 0.8527, GOLD),
            ("модель с\nисторией", 0.9231, GREEN)]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, [v for _, v, _ in rows], color=[c for *_, c in rows], width=0.6)
    for xi, (_, v, _) in zip(x, rows):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, [n for n, *_ in rows], fontsize=8)
    ax.set_ylim(0, 1)
    style(ax, ylabel="WAPE-score, сен-окт",
          title="Новая линия: форма переносится, объём нет")
    save(fig, "geo_configs")


def geo_warmup():
    days = [0, 7, 14, 31]
    score = [0.522, 0.864, 0.892, 0.903]
    worst = [0.000, 0.616, 0.817, 0.819]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.plot(days, score, "o-", color=GREEN, label="все девять маршрутов")
    ax.plot(days, worst, "o--", color=GOLD, label="худший маршрут")
    for d, s in zip(days, score):
        ax.text(d + 0.6, s - 0.05, f"{s:.3f}", fontsize=9, color=GREEN)
    ax.axhline(0.88, color=RED, lw=1, ls=":")
    ax.text(31, 0.885, "0.88", color=RED, fontsize=8, ha="right")
    ax.set_xticks(days)
    ax.set_ylim(-0.02, 1)
    style(ax, "дней собственных валидаций новой линии", "WAPE-score, октябрь",
          "Сколько истории нужно новой линии")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "geo_warmup")


def calendar_effects():
    rows = [("праздник в будний день\nк тому же дню недели", -52.4, -58.3, -47.9),
            ("новогодние каникулы\nк воскресенью", -14.6, -26.5, -5.3),
            ("выходные внутри\nдлинных праздников", -15.0, -20.4, -8.6),
            ("сокращённый день,\n17-19 ч", -9.7, -13.2, -3.1),
            ("сокращённый день,\n15-16 ч", 13.5, 11.0, 19.0),
            ("первая неделя\nсентября, будни", 8.7, 7.4, 10.2),
            ("продлённая ночная\nработа, 0-3 ч", 78.4, 25.6, 147.2)]
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, (_, e, lo, hi) in zip(y, rows):
        color = RED if e < 0 else GREEN
        ax.barh(yi, e, color=color, alpha=0.8, height=0.6)
        ax.plot([lo, hi], [yi, yi], color="#222", lw=1.2)
        ax.text(max(e, hi) + 3 if e > 0 else min(e, lo) - 3, yi, f"{e:+.1f}%",
                va="center", ha="left" if e > 0 else "right", fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y, [n for n, *_ in rows], fontsize=8)
    ax.set_xlim(-75, 165)
    style(ax, "посадки против нормы, %, с 95% ДИ", title="Календарь: измеренные эффекты")
    save(fig, "calendar_effects")


def weather_traffic_effects():
    rows = [("морось 0.05-0.3 мм/ч", -2.2, -3.5, -0.9), ("дождь 0.3-1 мм/ч", -6.4, -10.3, -2.9),
            ("дождь 1-3 мм/ч", -7.6, -12.9, -3.2), ("ливень > 3 мм/ч", -9.7, -14.0, -3.9),
            ("снегопад, зима", 0.6, -0.2, 1.7),
            ("Яндекс Пробки 3-4 балла", 5.9, -0.3, 13.6), ("балл ЦОДД 8-9", 0.9, -1.8, 3.5),
            ("парковки у линии\nполнее на 10+ п.п.", 3.0, 1.2, 5.0),
            ("пост о задержке трамвая,\nтот же час", -29.6, -37.9, -22.8),
            ("отмена маршрута", -96.7, -96.7, -96.7), ("объезд", -17.0, -17.0, -17.0)]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, (_, e, lo, hi) in zip(y, rows):
        significant = lo > 0 or hi < 0
        ax.barh(yi, e, color=(RED if e < 0 else GREEN) if significant else GREY,
                alpha=0.85, height=0.6)
        if hi != lo:
            ax.plot([lo, hi], [yi, yi], color="#222", lw=1.2)
        ax.text(min(e, lo) - 2 if e < 0 else max(e, hi) + 2, yi, f"{e:+.1f}%", va="center",
                ha="right" if e < 0 else "left", fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y, [n for n, *_ in rows], fontsize=8)
    ax.set_xlim(-115, 25)
    style(ax, "посадки против нормы, %; серым - ДИ включает 0",
          title="Погода, трафик и изменения маршрутов")
    save(fig, "weather_traffic_effects")


def rain_by_route():
    routes = ["1", "7", "11", "12", "17", "25", "26", "28", "50"]
    e = [-7.0, -5.4, -6.5, -6.4, -9.6, -10.0, -6.0, -7.2, -3.1]
    lo = [-11, -10, -11, -10, -15, -16, -10, -13, -8]
    hi = [-3, 0, -3, -3, -5, -3, -2, -3, 1]
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    x = np.arange(len(routes))
    ax.bar(x, e, color=BLUE, alpha=0.85, width=0.6)
    ax.errorbar(x, e, yerr=[np.array(e) - np.array(lo), np.array(hi) - np.array(e)],
                fmt="none", ecolor="#222", lw=1, capsize=3)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x, routes)
    style(ax, "маршрут", "посадки против нормы, %",
          "Дождь от 0.3 мм/ч, дневные часы, по маршрутам")
    save(fig, "rain_by_route")


def perf_rps():
    rows = [("Python ONNX\n+ Go", 3700, 5.74, 150), ("два процесса\nраннера", 4650, 5.39, 205),
            ("голова CNN\nв Go", 23200, 1.24, 11), ("+ блоки\nпо 4 строки", 27500, 1.02, 11),
            ("+ softplus\nмногочленом", 30000, 0.91, 9)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    x = np.arange(len(rows))
    colors = [GREY, GREY, GREEN, GREEN, GREEN]
    axes[0].bar(x, [r[1] for r in rows], color=colors, width=0.6)
    for xi, r in zip(x, rows):
        axes[0].text(xi, r[1] + 400, f"{r[1]:,}".replace(",", " "), ha="center", fontsize=8)
    axes[0].set_xticks(x, [r[0] for r in rows], fontsize=8)
    style(axes[0], ylabel="максимальный RPS", title="Пропускная способность, 2 ядра")
    axes[1].bar(x, [r[2] for r in rows], color=colors, width=0.6)
    for xi, r in zip(x, rows):
        axes[1].text(xi, r[2] + 0.1, f"{r[2]:.2f} мс\n{r[3]} МБ", ha="center", fontsize=8)
    axes[1].set_xticks(x, [r[0] for r in rows], fontsize=8)
    axes[1].set_ylim(0, 7)
    style(axes[1], ylabel="p95 при 8 потоках, мс", title="Задержка и память")
    save(fig, "perf_rps")


def perf_catboost():
    rows = [("1 vCPU,\n1000 RPS", 1000, 1.8, 32, 57), ("1 vCPU,\nмаксимум", 1747, 8.0, 23, 32),
            ("2 vCPU,\nмаксимум", 2432, 4.6, 20, 32)]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    x = np.arange(len(rows))
    for off, idx, name, color in [(-0.25, 2, "p50", GREEN), (0, 3, "p95", GOLD),
                                  (0.25, 4, "p99", RED)]:
        ax.bar(x + off, [r[idx] for r in rows], 0.24, color=color, label=name)
    for xi, r in zip(x, rows):
        ax.text(xi, max(r[2:]) + 3, f"{r[1]} RPS", ha="center", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 70)
    style(ax, ylabel="задержка, мс", title="Резерв: CatBoost в Go, микс запросов диспетчера")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "perf_catboost")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for fn in (score_progress, cnn_training, backtest_vs_target, ml_steps, geo_levels,
               geo_configs, geo_warmup, calendar_effects, weather_traffic_effects,
               rain_by_route, perf_rps, perf_catboost):
        fn()
    print("\n".join(sorted(p.name for p in IMG.glob("*.svg"))))
