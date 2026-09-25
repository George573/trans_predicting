import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
ROUTES = {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
HOLIDAYS = pd.to_datetime(
    [f"2025-01-0{day}" for day in range(1, 9)]
    + ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11"]
    + ["2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15"]
)
DELAYED_ROUTES_RE = re.compile(r"задерживаются трамва\w*((?:\s*№?\s*\d{1,2}\s*(?:,|и)?)+)")


def naive(moments):
    return moments.dt.tz_localize(None) if moments.dt.tz is not None else moments


def residual(series, step, log=True):
    series = series.where(pd.Series(~series.index.normalize().isin(HOLIDAYS), series.index), axis=0)
    base = pd.concat([series.shift(k * step) for k in (7, 14, -7, -14)]).groupby(level=0).median()
    return np.log((series + 1) / (base + 1)) if log else series - base


def report(name, x, y):
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    pearson = joined.x.corr(joined.y)
    spearman = joined.x.rank().corr(joined.y.rank())
    print(f"  {name:55s} n={len(joined):5d}  pearson={pearson:+.3f}  spearman={spearman:+.3f}")


labels = pd.concat(pd.read_csv(ROOT / f"dataset/labels/labels_day_{part}.csv", sep=";") for part in ("train", "test"))
labels["time"] = pd.to_datetime(labels.date) + pd.to_timedelta(labels.hour, unit="h")
by_route_hour = labels.pivot_table(index="time", columns="route", values="boardings", aggfunc="sum").asfreq("h").fillna(0)
daily = residual(by_route_hour.sum(axis=1).resample("D").sum(), 1)
hourly = residual(by_route_hour.sum(axis=1), 24)
hourly = hourly[(hourly.index.hour >= 6) & (hourly.index.hour <= 22)]

print("Отклонение пассажиропотока от обычного для этого дня недели и часа против отклонения трафика")
telegram = pd.read_csv(DATA / "congestion_telegram.csv", parse_dates=["time_msk"])
telegram["time"] = naive(telegram.time_msk)
oblast = telegram[telegram.region == "Московская область"]
morning = oblast[oblast.time.dt.hour < 12].groupby(oblast.time.dt.normalize())
report("МО: машин на дорогах в 08:55 (ЦБДД)", residual(morning.vehicles_on_roads.first().asfreq("D"), 1), daily)
report("МО: утренний балл (ЦБДД)", residual(morning.score.first().asfreq("D"), 1, log=False), daily)
moscow = telegram[telegram.region == "Москва"].dropna(subset=["score"])
report("Москва: максимальный балл за день (ЦОДД)", moscow.groupby(moscow.time.dt.normalize()).score.max(), daily)

yandex = pd.read_csv(DATA / "congestion_yandex_wayback.csv", parse_dates=["time_msk"])
report("Москва: балл Яндекс Пробок (архив)", yandex.groupby(naive(yandex.time_msk).dt.floor("h")).score.mean(), hourly)

accidents = pd.read_csv(DATA / "accidents_gibdd.csv", parse_dates=["time_msk"])
accidents = accidents[accidents.region == "Москва"]
report("Москва: ДТП за день (ГИБДД)", residual(accidents.groupby(accidents.time_msk.dt.normalize()).size().asfreq("D").fillna(0), 1, log=False), daily)

parking = pd.read_csv(DATA / "parking_occupancy_hourly.csv", parse_dates=["time_msk"])
parking = parking.groupby("time_msk").occupancy_pct.mean().asfreq("h")
report("Москва: средняя загрузка парковок, по часам", residual(parking, 24), hourly)

print("\nМаршрут в ближайшие 3 часа после поста Дептранса «задерживаются трамваи №...»")
route_hourly = residual(by_route_hour, 24)
events = pd.read_csv(DATA / "events_deptrans_telegram.csv", parse_dates=["time_msk"], keep_default_na=False)
drops = []
for post in events.itertuples():
    start = naive(pd.Series([post.time_msk])).iloc[0].floor("h")
    for match in DELAYED_ROUTES_RE.finditer(post.text.lower()):
        for route in {int(n) for n in re.findall(r"\d{1,2}", match.group(1))} & ROUTES:
            if 6 <= start.hour <= 21 and start in route_hourly.index:
                drops.append(route_hourly.loc[start:start + pd.Timedelta(hours=2), route].mean())
drops = pd.Series(drops).dropna()
baseline = route_hourly[(route_hourly.index.hour >= 6) & (route_hourly.index.hour <= 21)].stack().dropna()
print(f"  случаев: {len(drops)}; пассажиропоток от обычного: медиана {np.exp(drops.median()):.2f}, среднее {np.exp(drops.mean()):.2f}")
print(f"  все часы всех маршрутов для сравнения: медиана {np.exp(baseline.median()):.2f}, среднее {np.exp(baseline.mean()):.2f}")
