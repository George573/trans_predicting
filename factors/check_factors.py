import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).parent / "data"
FORECAST_START = pd.Timestamp("2025-11-01")
SEASONS = {1: "зима", 2: "зима", 3: "зима", 4: "весна", 5: "весна", 6: "лето", 7: "лето", 8: "лето", 9: "осень", 10: "осень"}
NONE = "нет"
RAIN_BINS = ([-1, 0.05, 0.3, 1, 3, 99], [NONE, "0.05-0.3 мм/ч", "0.3-1 мм/ч", "1-3 мм/ч", "больше 3 мм/ч"])
BOOTSTRAP = 300
FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]
RNG = np.random.default_rng(0)


def norm(table, column, holiday):
    wide = table.assign(value=table[column].where(~holiday)).pivot(index="time", columns="route", values="value")
    base = pd.concat([wide.shift(days * 24) for days in (7, 14, -7, -14)]).groupby(level=0).median()
    return base.stack().reindex(pd.MultiIndex.from_arrays([table.time, table.route])).values


def load():
    table = pd.read_csv(DATA / "factors_hourly.csv")
    table["time"] = pd.to_datetime(table.date) + pd.to_timedelta(table.hour, unit="h")
    table = table[(table.route != 5) & (table.time < FORECAST_START)].sort_values(["route", "time"]).reset_index(drop=True)
    table["day"] = table.time.dt.normalize()
    table["month"] = table.time.dt.month
    table["season"] = table.month.map(SEASONS)
    holiday = ((table.day_off == 1) & (table.time.dt.dayofweek < 5)) | (table.day < "2025-01-09")
    table["norm"] = norm(table, "boardings", holiday)
    table["usable"] = ~holiday & (table.norm > 0)
    table["daytime"] = table.hour.between(6, 22)
    for column in ["temperature_c", "forecast_temperature_c", "parking_pct", "parking_pct_route", "accidents_moscow"]:
        table[f"{column}_anomaly"] = table[column] - norm(table, column, holiday)
    table["mo_vehicles_anomaly"] = table.mo_vehicles / norm(table, "mo_vehicles", holiday) - 1
    recent = table.groupby("route").accidents_near_route.transform(lambda s: s.rolling(3, min_periods=1).sum())
    table["accident_recent"] = recent > 0
    table["precipitation_day"] = table.groupby(["route", "day"]).precipitation_mm.transform("sum")
    return table


def labelled(values, edges, labels):
    return pd.cut(values, edges, labels=labels).astype(object).fillna("nan")


def effect(table, bins, mask):
    frame = table.loc[mask, ["day", "boardings", "norm"]].assign(bin=bins[mask])
    frame = frame[frame.bin != "nan"]
    daily = frame.groupby(["day", "bin"])[["boardings", "norm"]].sum().unstack("bin", fill_value=0)

    def change(days):
        sums = days.sum()
        share = sums["boardings"] / sums["norm"]
        return share / share[NONE] - 1

    samples = pd.DataFrame([change(daily.iloc[RNG.integers(0, len(daily), len(daily))]) for _ in range(BOOTSTRAP)])
    result = pd.DataFrame({
        "hours": frame.bin.value_counts(),
        "effect": change(daily),
        "low": samples.quantile(0.025),
        "high": samples.quantile(0.975),
    })
    return result.drop(index=NONE)


def show(title, result):
    print(f"\n{title}")
    for name, row in result.iterrows():
        print(f"  {name:32s} {row.hours:7.0f} ч  {100 * row.effect:+6.1f}%  [{100 * row.low:+.1f}; {100 * row.high:+.1f}]")


def weather_effects(table, rain, cold):
    day = table.usable & table.daytime
    show("Осадки в этот час, дневные часы 6-22", effect(table, rain, day))
    for season in ["зима", "весна", "лето", "осень"]:
        for day_off, kind in [(0, "будни"), (1, "выходные")]:
            mask = day & (table.season == season) & (table.day_off == day_off)
            show(f"Осадки: {season}, {kind}", effect(table, rain, mask))
    forecast = labelled(table.forecast_precipitation_mm, *RAIN_BINS)
    show("Осадки по прогнозу ECMWF за сутки вперёд, дневные часы", effect(table, forecast, day))
    for season in ["зима", "весна", "лето", "осень"]:
        show(f"Отклонение температуры от обычной для этого часа: {season}", effect(table, cold, day & (table.season == season)))
    heat = pd.Series(np.where(table.temperature_c > 28, "жара выше 28", NONE), table.index)
    show("Жара, летние дневные часы", effect(table, heat, day & (table.season == "лето")))
    snow = pd.Series(np.where((table.precipitation_mm > 0.3) & (table.temperature_c <= 0), "снег больше 0.3 мм/ч", NONE), table.index)
    show("Снегопад, зимние дневные часы", effect(table, snow, day & (table.season == "зима")))


def traffic_effects(table, delay):
    day = table.usable & table.daytime
    yandex = labelled(table.yandex_score, [-1, 2, 4, 10], [NONE, "3-4 балла", "5-6 баллов"])
    show("Балл Яндекс Пробок в этот час (0-2 = нет)", effect(table, yandex, day))
    codd = labelled(table.codd_score, [-1, 5, 7, 10], [NONE, "6-7 баллов", "8-9 баллов"])
    show("Максимальный балл ЦОДД за день (0-5 = нет)", effect(table, codd, day))
    vehicles = labelled(table.mo_vehicles_anomaly, [-9, -0.03, 0.03, 9], ["машин в МО меньше на 3%+", NONE, "машин в МО больше на 3%+"])
    show("Машин на дорогах МО утром против обычного", effect(table, vehicles, day))
    accidents = labelled(table.accidents_moscow_anomaly, [-99, -6, 6, 99], ["ДТП в Москве на 6+ меньше", NONE, "ДТП в Москве на 6+ больше"])
    show("ДТП с пострадавшими в Москве за день против обычного", effect(table, accidents, day))
    near = pd.Series(np.where(table.accident_recent, "ДТП в 100 м от линии за 3 ч", NONE), table.index)
    show("ДТП у линии маршрута", effect(table, near, day))
    show("Пост Дептранса «задерживаются трамваи №» с этим маршрутом", effect(table, delay, day))
    parking = labelled(table.parking_pct_route_anomaly, [-999, -10, 10, 999], ["парковки у линии свободнее на 10+ п.п.", NONE, "парковки у линии полнее на 10+ п.п."])
    show("Загрузка парковок в 1 км от линии против обычной", effect(table, parking, day))


def route_effects(table, rain, cold, delay):
    lines = json.loads((DATA / "route_lines.geojson").read_text())["features"]
    names = {line["properties"]["route"]: line["properties"]["names"][0].split(": ")[-1] for line in lines}
    rainy = rain.map({NONE: NONE, "0.3-1 мм/ч": "дождь", "1-3 мм/ч": "дождь", "больше 3 мм/ч": "дождь"}).fillna("nan")
    frozen = cold.map({NONE: NONE, "холоднее на 4-8": "холоднее", "холоднее на 8+": "холоднее"}).fillna("nan")
    delayed = delay.map({NONE: NONE, "в час поста": "задержка", "через 1 ч": "задержка", "через 2 ч": "задержка"}).fillna("nan")
    near = pd.Series(np.where(table.accident_recent, "ДТП", NONE), table.index)
    print("\nПо маршрутам: изменение посадок, % [95% ДИ]; дождь от 0.3 мм/ч, холоднее обычного на 4+ градуса, 0-2 ч после поста о задержке, ДТП у линии за 3 ч")
    for route in sorted(table.route.unique()):
        mask = table.usable & table.daytime & (table.route == route)
        cells = []
        for bins in (rainy, frozen, delayed, near):
            row = effect(table, bins, mask).iloc[0] if (~bins[mask].isin([NONE, "nan"])).any() else None
            cells.append("-" if row is None else f"{100 * row.effect:+.1f} [{100 * row.low:+.0f}; {100 * row.high:+.0f}]")
        posts = (table.delay_post_age_h[table.route == route] == 0).sum()
        print(f"  {route:>2} {names[route][:42]:42s} дождь {cells[0]:20s} холод {cells[1]:20s} задержка ({posts} постов) {cells[2]:20s} ДТП {cells[3]}")


def weather_and_traffic(table):
    city = table[table.route == table.route.iloc[0]]
    days = city.groupby("day").agg(
        precipitation=("precipitation_mm", "sum"),
        temperature=("temperature_c", "mean"),
        codd=("codd_score", "max"),
        accidents=("accidents_moscow", "first"),
        vehicles=("mo_vehicles_anomaly", "first"),
        day_off=("day_off", "first"),
    )
    wet = pd.cut(days.precipitation, [-1, 0.5, 5, 99], labels=["сухо", "0.5-5 мм", "больше 5 мм"])
    snow = pd.Series(np.where((days.temperature <= 0) & (days.precipitation > 2), "снег", ""), days.index)
    print("\nПогода и пробки по дням (будни): балл ЦОДД, ДТП в Москве, машин в МО против обычного")
    working = days.day_off == 0
    summary = days[working].groupby([wet[working], snow[working]], observed=True).agg(
        days=("codd", "size"), codd=("codd", "mean"), accidents=("accidents", "mean"), vehicles=("vehicles", "mean")
    )
    print(summary.round(3).to_string())


def joint_model(table):
    rows = table[table.usable & table.daytime & (table.month >= 4)].copy()
    features = {
        "осадки, на 1 мм/ч": rows.precipitation_mm.clip(0, 3),
        "теплее обычного, на 10 градусов": rows.temperature_c_anomaly / 10,
        "0-2 ч после поста о задержке": rows.delay_post_age_h.le(2).astype(float),
        "ДТП у линии за 3 ч": rows.accident_recent.astype(float),
        "ДТП в Москве за день, на 10 больше": rows.accidents_moscow_anomaly / 10,
        "парковки у линии, на 10 п.п. полнее": rows.parking_pct_route_anomaly.fillna(0) / 10,
        "выходной день (контроль)": rows.day_off.astype(float),
    }
    matrix = pd.DataFrame(features).assign(intercept=1.0)
    keep = matrix.notna().all(axis=1)
    matrix, rows = matrix[keep], rows[keep]
    target = np.log((rows.boardings + 1) / (rows.norm + 1))
    weights = np.sqrt(rows.norm.values)

    def fit(index):
        solution, *_ = np.linalg.lstsq(matrix.values[index] * weights[index, None], target.values[index] * weights[index], rcond=None)
        return solution

    days = rows.day.values
    unique_days = np.unique(days)
    by_day = {day: np.flatnonzero(days == day) for day in unique_days}
    point = fit(np.arange(len(rows)))
    samples = np.array([
        fit(np.concatenate([by_day[day] for day in RNG.choice(unique_days, len(unique_days))])) for _ in range(200)
    ])
    print(f"\nВсё вместе: совместная регрессия отклонения посадок, апрель-октябрь, дневные часы, {len(rows)} маршруто-часов")
    for index, name in enumerate(matrix.columns[:-1]):
        low, high = np.percentile(samples[:, index], [2.5, 97.5])
        print(f"  {name:40s} {100 * np.expm1(point[index]):+6.1f}%  [{100 * np.expm1(low):+.1f}; {100 * np.expm1(high):+.1f}]")


def corrections(table, bins, keys):
    factor = pd.Series(1.0, table.index)
    for month in range(1, 11):
        fit = table.usable & (table.month != month)
        sums = table[fit].assign(bin=bins[fit]).groupby(keys + ["bin"])[["boardings", "norm"]].sum()
        share = (sums.boardings / sums.norm).unstack("bin")
        share = share.div(share[NONE], axis=0).stack()
        current = table.month == month
        wanted = pd.MultiIndex.from_frame(table.loc[current, keys].assign(bin=bins[current]))
        factor[current] = share.reindex(wanted).fillna(1.0).values
    return factor


def wape_gain(table, factor):
    rows = table[table.usable]
    base = (rows.boardings - rows.norm).abs()
    corrected = (rows.boardings - rows.norm * factor[rows.index]).abs()
    daily = pd.DataFrame({"y": rows.boardings, "base": base, "corrected": corrected, "day": rows.day}).groupby("day").sum()
    gain = lambda days: (days.base.sum() - days.corrected.sum()) / days.y.sum()
    samples = [gain(daily.iloc[RNG.integers(0, len(daily), len(daily))]) for _ in range(BOOTSTRAP)]
    return 1 - base.sum() / rows.boardings.sum(), gain(daily), *np.percentile(samples, [2.5, 97.5])


def wape_checks(table, rain, cold, delay):
    table = table.assign(everything="все")
    rain = rain.where(table.daytime, NONE).replace("nan", NONE)
    cold = cold.replace("nan", NONE)
    factors = {
        "дождь": corrections(table, rain, ["everything"]),
        "дождь по сезону и типу дня": corrections(table, rain, ["season", "day_off"]),
        "температура по сезону": corrections(table, cold, ["season"]),
        "посты о задержках": corrections(table, delay, ["everything"]),
    }
    factors["погода"] = factors["дождь по сезону и типу дня"] * factors["температура по сезону"]
    factors["погода и посты"] = factors["погода"] * factors["посты о задержках"]
    print("\nWAPE-score нормы (медиана того же часа ±1-2 недели) с поправкой и без, коэффициенты считаются без месяца, на котором проверяются")
    for name, factor in factors.items():
        base, gain, low, high = wape_gain(table, factor)
        print(f"  {name:28s} {base:.4f} -> {base + gain:.4f}  {100 * gain:+.3f} п.п.  [{100 * low:+.3f}; {100 * high:+.3f}]")


def boosting_checks(table):
    table = table.assign(weekday=table.time.dt.dayofweek, delay=table.delay_post_age_h.fillna(99), route=table.route.astype("category"))
    calendar = ["route", "hour", "weekday", "day_off"]
    weather = ["temperature_c", "precipitation_mm", "precipitation_day"]
    traffic = ["delay", "accidents_moscow", "accidents_near_route"]
    sets = {"календарь": calendar, "+погода": calendar + weather, "+трафик": calendar + traffic, "+всё": calendar + weather + traffic}
    print("\nБустинг на календаре и факторах, WAPE-score; обучение на всём до начала периода")
    print(f"  {'':12s}" + "".join(f"{start[:7]:>12s}" for start, _ in FOLDS))
    for name, columns in sets.items():
        scores = []
        for start, end in FOLDS:
            train, test = table.time < start, (table.time >= start) & (table.time < end)
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, categorical_features=["route"], random_state=0)
            model.fit(table.loc[train, columns], np.log1p(table.boardings[train]))
            prediction = np.expm1(model.predict(table.loc[test, columns])).clip(0)
            scores.append(1 - np.abs(table.boardings[test] - prediction).sum() / table.boardings[test].sum())
        print(f"  {name:12s}" + "".join(f"{score:12.4f}" for score in scores))


if __name__ == "__main__":
    table = load()
    rain = labelled(table.precipitation_mm, *RAIN_BINS)
    cold = labelled(table.temperature_c_anomaly, [-99, -8, -4, 4, 8, 99], ["холоднее на 8+", "холоднее на 4-8", NONE, "теплее на 4-8", "теплее на 8+"])
    delay = labelled(table.delay_post_age_h, [-1, 0, 1, 2, 9], ["в час поста", "через 1 ч", "через 2 ч", "через 3-5 ч"]).replace("nan", NONE)
    print("Изменение посадок против нормы (медиана того же маршрута, дня недели и часа за ±1-2 недели), % [95% ДИ по дням]")
    weather_effects(table, rain, cold)
    traffic_effects(table, delay)
    route_effects(table, rain, cold, delay)
    weather_and_traffic(table)
    joint_model(table)
    wape_checks(table, rain, cold, delay)
    boosting_checks(table)
