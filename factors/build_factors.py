import json
import re
import time
import urllib.parse
import urllib.request
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "data"
WEATHER = ROOT / "weather" / "data"
TRAFFIC = ROOT / "traffic" / "data"
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
HOURS = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
FORECAST_START = pd.Timestamp("2025-11-01")
HOLIDAYS = pd.to_datetime(
    [f"2025-01-0{day}" for day in range(1, 9)]
    + ["2025-05-01", "2025-05-02", "2025-05-09", "2025-06-12", "2025-11-03", "2025-11-04", "2025-12-31"]
)
WORKING_WEEKENDS = pd.to_datetime(["2025-11-01"])

OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OSM_QUERY = (
    '[out:json][timeout:170];relation["route"="tram"]["ref"~"^(1|5|7|11|12|17|25|26|28|50)$"]'
    "(55.49,37.3,55.95,37.95);out geom;"
)
KM_PER_DEGREE = np.array([111.32 * np.cos(np.radians(55.75)), 110.57])
WEATHER_POINTS = {"Москва (Балчуг)": (37.63, 55.745), "Москва (ВДНХ)": (37.617, 55.833)}
WEATHER_COLUMNS = ["temperature_c", "precipitation_mm", "wind_speed_ms", "cloud_cover_pct", "humidity_pct"]
ACCIDENT_RADIUS_KM = 0.1
PARKING_RADIUS_KM = 1.0
DELAY_HOURS = 6
DELAYED_RE = re.compile(r"задерживаются трамва\w*([^.\n]*)", re.I)
ROUTE_NUMBER_RE = re.compile(r"(?<![\w-])\d{1,2}(?![\w-])")


def msk_hour(moments):
    return pd.to_datetime(moments.str[:16]).dt.floor("h")


def route_lines():
    path = OUT / "route_lines.geojson"
    if not path.exists():
        relations = overpass(OSM_QUERY)
        features = []
        for route in ROUTES:
            ways, members = {}, [r for r in relations if r["tags"]["ref"] == str(route)]
            for relation in members:
                for member in relation["members"]:
                    if member["type"] == "way" and member["role"] == "":
                        ways[member["ref"]] = [[point["lon"], point["lat"]] for point in member["geometry"]]
            features.append({
                "type": "Feature",
                "properties": {
                    "route": route,
                    "names": [r["tags"]["name"] for r in members],
                    "osm_relations": [r["id"] for r in members],
                },
                "geometry": {"type": "MultiLineString", "coordinates": list(ways.values())},
            })
        OUT.mkdir(exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False))
    return {f["properties"]["route"]: f["geometry"]["coordinates"] for f in json.loads(path.read_text())["features"]}


def track_points(lines, step_km=0.05):
    points = []
    for line in lines:
        line = np.array(line) * KM_PER_DEGREE
        for start, end in zip(line[:-1], line[1:]):
            count = max(1, int(np.hypot(*(end - start)) / step_km))
            points.extend(start + (end - start) * share for share in np.linspace(0, 1, count, endpoint=False))
    return np.array(points)


def near_track(lat, lon, points, radius_km):
    places = np.c_[lon, lat] * KM_PER_DEGREE
    distance = np.full(len(places), np.inf)
    for chunk in np.array_split(points, max(1, len(points) // 500)):
        gaps = np.sqrt(((places[:, None, :] - chunk[None]) ** 2).sum(-1)).min(1)
        distance = np.minimum(distance, gaps)
    return distance <= radius_km


def delayed_routes(text):
    return {int(number) for part in DELAYED_RE.findall(text) for number in ROUTE_NUMBER_RE.findall(part)}


def day_off(days):
    weekend = days.dayofweek >= 5
    return ((weekend & ~days.isin(WORKING_WEEKENDS)) | days.isin(HOLIDAYS)).astype(int)


def city_factors():
    hourly = pd.DataFrame(index=HOURS)
    yandex = traffic("congestion_yandex_wayback.csv")
    hourly["yandex_score"] = yandex.groupby(msk_hour(yandex.time_msk)).score.mean()
    parking = traffic("parking_occupancy_hourly.csv")
    hourly["parking_pct"] = parking.groupby(pd.to_datetime(parking.time_msk)).occupancy_pct.mean()

    daily = pd.DataFrame(index=HOURS.normalize().unique())
    telegram = traffic("congestion_telegram.csv")
    hour = msk_hour(telegram.time_msk)
    moscow = telegram[telegram.region == "Москва"]
    daily["codd_score"] = moscow.groupby(hour.dt.normalize()).score.max()
    oblast = telegram[(telegram.region == "Московская область") & (hour.dt.hour < 12)]
    daily["mo_vehicles"] = oblast.groupby(hour.dt.normalize()).vehicles_on_roads.first()
    accidents = traffic("accidents_gibdd.csv")
    accidents = accidents[accidents.region == "Москва"]
    daily["accidents_moscow"] = accidents.groupby(msk_hour(accidents.time_msk).dt.normalize()).size()
    daily["accidents_moscow"] = daily.accidents_moscow.fillna(0)
    return hourly, daily


def route_factors(route, points):
    frame = pd.DataFrame(index=HOURS)
    accidents = traffic("accidents_gibdd.csv").dropna(subset=["lat", "lon"])
    accidents = accidents[near_track(accidents.lat.values, accidents.lon.values, points, ACCIDENT_RADIUS_KM)]
    frame["accidents_near_route"] = accidents.groupby(msk_hour(accidents.time_msk)).size()
    frame["accidents_near_route"] = frame.accidents_near_route.fillna(0)

    posts = traffic("events_deptrans_telegram.csv")
    frame["delay_post_age_h"] = np.nan
    for hour, text in zip(msk_hour(posts.time_msk), posts.text.fillna("")):
        if route in delayed_routes(text):
            window = pd.date_range(hour, periods=DELAY_HOURS, freq="h").intersection(HOURS)
            ages = pd.Series(np.arange(len(window)), window)
            frame.loc[window, "delay_post_age_h"] = np.fmin(frame.loc[window, "delay_post_age_h"], ages)

    spots = traffic("parking_spots.csv")
    nearby = spots.parking_id[near_track(spots.lat.values, spots.lon.values, points, PARKING_RADIUS_KM)]
    parking = traffic("parking_occupancy_hourly.csv")
    parking = parking[parking.parking_id.isin(nearby)]
    frame["parking_pct_route"] = parking.groupby(pd.to_datetime(parking.time_msk)).occupancy_pct.mean()
    return frame


def weather(path, columns, prefix=""):
    table = pd.read_csv(path, parse_dates=["time_msk"])
    table = table[table.location.isin(WEATHER_POINTS)].set_index(["location", "time_msk"])[columns]
    return table.add_prefix(prefix)


def build():
    labels = pd.concat(
        pd.read_csv(ROOT / f"dataset/labels/labels_day_{part}.csv", sep=";") for part in ("train", "test")
    )
    labels["time"] = pd.to_datetime(labels.date) + pd.to_timedelta(labels.hour, unit="h")
    boardings = labels.set_index(["route", "time"]).boardings

    fact = weather(WEATHER / "fact_open_meteo_era5.csv.gz", WEATHER_COLUMNS)
    forecast = weather(WEATHER / "forecast_ecmwf_day1.csv.gz", ["temperature_c", "precipitation_mm"], "forecast_")
    hourly, daily = city_factors()
    lines = route_lines()

    frames = []
    for route in ROUTES:
        points = track_points(lines[route])
        centre = points.mean(0)
        point = min(WEATHER_POINTS, key=lambda name: np.hypot(*(np.array(WEATHER_POINTS[name]) * KM_PER_DEGREE - centre)))
        frame = pd.DataFrame(index=pd.Index(HOURS, name="time"))
        frame["route"] = route
        frame["boardings"] = boardings.get(route, pd.Series(dtype=float)).reindex(HOURS)
        frame.loc[frame.index < FORECAST_START, "boardings"] = frame.boardings.fillna(0)
        frame["day_off"] = day_off(HOURS.normalize())
        frame = frame.join(fact.loc[point]).join(forecast.loc[point])
        frame = frame.join(hourly).join(daily.reindex(HOURS.normalize()).set_axis(HOURS))
        frames.append(frame.join(route_factors(route, points)))
        print(f"  маршрут {route}: погода по точке {point}, {len(points)} точек линии")

    table = pd.concat(frames).reset_index()
    table.insert(1, "date", table.time.dt.strftime("%Y-%m-%d"))
    table.insert(2, "hour", table.time.dt.hour)
    table = table.drop(columns="time")
    table = table[["route"] + [column for column in table.columns if column != "route"]]
    table.round(2).convert_dtypes().to_csv(OUT / "factors_hourly.csv", index=False)
    print(f"  factors_hourly.csv: {len(table)} строк")


@cache
def traffic(name):
    return pd.read_csv(TRAFFIC / name)


def overpass(query):
    body = urllib.parse.urlencode({"data": query}).encode()
    for attempt, server in enumerate(OVERPASS * 2):
        try:
            request = urllib.request.Request(server, body, {"User-Agent": "tram-factors/1.0"})
            with urllib.request.urlopen(request, timeout=200) as response:
                return json.load(response)["elements"]
        except OSError as error:
            print(f"  {server}: {error}")
            time.sleep(30 * attempt)
    raise RuntimeError("Overpass недоступен")


if __name__ == "__main__":
    build()
