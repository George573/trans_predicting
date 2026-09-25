import csv
import gzip
import http.client
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path

YEAR = 2025
MSK = timedelta(hours=3)
OUT = Path(__file__).parent / "data"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
COLUMNS = [
    "location", "lat", "lon", "time_msk",
    "temperature_c", "humidity_pct", "precipitation_mm", "pressure_hpa",
    "wind_speed_ms", "wind_dir_deg", "cloud_cover_pct",
]
OPEN_METEO_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover",
]
FORECAST_MODELS = {"ecmwf_ifs025": "ecmwf", "icon_seamless": "icon", "gfs_seamless": "gfs"}
METEOSTAT_MODEL_SOURCES = {"dwd_mosmix", "metno_forecast"}

STATIONS = [
    ("Москва (ВДНХ)", 55.833, 37.617, "27612", "27612"),
    ("Москва (Балчуг)", 55.745, 37.630, "27605", None),
    ("Клин", 56.350, 36.750, "27417", "27417"),
    ("Дмитров", 56.333, 37.517, "27419", "27419"),
    ("Волоколамск", 56.017, 35.933, "27502", "27502"),
    ("Можайск", 55.517, 36.000, "27509", "27509"),
    ("Ново-Иерусалим", 55.900, 36.817, "27511", "27511"),
    ("Павловский Посад", 55.767, 38.683, "27523", "27523"),
    ("Наро-Фоминск", 55.383, 36.700, "27611", "27611"),
    ("Серпухов", 54.933, 37.467, "27618", "27618"),
    ("Коломна", 55.133, 38.733, "27625", "27625"),
    ("Кашира", 54.833, 38.150, "27627", "27627"),
    ("Шереметьево", 55.967, 37.400, None, "UUEE0"),
    ("Внуково", 55.583, 37.267, None, "UUWW0"),
    ("Домодедово", 55.400, 37.900, None, "UUDD0"),
    ("Жуковский", 55.550, 38.150, None, "UUBW0"),
]

GISMETEO_CITIES = {
    "Москва": "moscow-4368",
    "Клин": "klin-4329",
    "Дмитров": "dmitrov-4330",
    "Волоколамск": "volokolamsk-4341",
    "Можайск": "mozhaysk-4343",
    "Истра": "istra-4344",
    "Павловский Посад": "pavlovsky-posad-4346",
    "Наро-Фоминск": "naro-fominsk-4367",
    "Серпухов": "serpukhov-4370",
    "Коломна": "kolomna-4372",
    "Кашира": "kashira-4373",
    "Домодедово": "domodedovo-4369",
    "Жуковский": "zhukovsky-11329",
}

RP5_WIND = {
    "севера": 0, "северо-северо-востока": 22.5, "северо-востока": 45, "востоко-северо-востока": 67.5,
    "востока": 90, "востоко-юго-востока": 112.5, "юго-востока": 135, "юго-юго-востока": 157.5,
    "юга": 180, "юго-юго-запада": 202.5, "юго-запада": 225, "западо-юго-запада": 247.5,
    "запада": 270, "западо-северо-запада": 292.5, "северо-запада": 315, "северо-северо-запада": 337.5,
}

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))


def fetch(url, data=None, headers=None):
    body = urllib.parse.urlencode(data).encode() if data else None
    request = urllib.request.Request(url, body, {"User-Agent": UA, **(headers or {})})
    for attempt in range(5):
        try:
            with opener.open(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            print("  429, жду минуту")
            time.sleep(61)
        except (OSError, http.client.HTTPException) as error:
            if attempt == 4:
                raise
            print(f"  {error}, повтор")
            time.sleep(15 * 2**attempt)


def fetch_json(url, **params):
    return json.loads(fetch(f"{url}?{urllib.parse.urlencode(params)}"))


def number(value):
    return None if value in ("", None, -999) else value


def write_csv(name, rows, columns=COLUMNS):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"  {name}: {len(rows)} строк")


def in_year(moment):
    return moment.year == YEAR


def open_meteo_era5():
    rows = []
    for name, lat, lon, *_ in STATIONS:
        hourly = fetch_json(
            "https://archive-api.open-meteo.com/v1/archive",
            latitude=lat, longitude=lon, start_date=f"{YEAR}-01-01", end_date=f"{YEAR}-12-31",
            hourly=",".join(OPEN_METEO_VARS), models="era5", timezone="Europe/Moscow", wind_speed_unit="ms",
        )["hourly"]
        for i, moment in enumerate(hourly["time"]):
            rows.append([name, lat, lon, moment, *(hourly[v][i] for v in OPEN_METEO_VARS)])
    write_csv("fact_open_meteo_era5.csv", rows)


def open_meteo_forecasts():
    rows = {model: [] for model in FORECAST_MODELS}
    for name, lat, lon, *_ in STATIONS:
        hourly = fetch_json(
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            latitude=lat, longitude=lon, start_date=f"{YEAR}-01-01", end_date=f"{YEAR}-12-31",
            hourly=",".join(f"{v}_previous_day1" for v in OPEN_METEO_VARS), models=",".join(FORECAST_MODELS),
            timezone="Europe/Moscow", wind_speed_unit="ms",
        )["hourly"]
        for model in FORECAST_MODELS:
            for i, moment in enumerate(hourly["time"]):
                values = [hourly[f"{v}_previous_day1_{model}"][i] for v in OPEN_METEO_VARS]
                rows[model].append([name, lat, lon, moment, *values])
    for model, short in FORECAST_MODELS.items():
        write_csv(f"forecast_{short}_day1.csv", rows[model])


def meteostat():
    rows = []
    for name, lat, lon, _, station in STATIONS:
        if not station:
            continue
        for year in (YEAR - 1, YEAR):
            try:
                raw = gzip.decompress(fetch(f"https://data.meteostat.net/hourly/{year}/{station}.csv.gz"))
            except urllib.error.HTTPError as error:
                print(f"  {name} {year}: HTTP {error.code}")
                continue
            for r in csv.DictReader(io.StringIO(raw.decode())):
                moment = datetime(int(r["year"]), int(r["month"]), int(r["day"]), int(r["hour"])) + MSK
                if not in_year(moment):
                    continue
                temp, rhum, prcp, pres, wspd, wdir, cldc, snwd = (
                    meteostat_observed(r, key) for key in ("temp", "rhum", "prcp", "pres", "wspd", "wdir", "cldc", "snwd")
                )
                values = [
                    temp, rhum, prcp, pres,
                    round(float(wspd) / 3.6, 2) if wspd else None, wdir, float(cldc) * 12.5 if cldc else None, snwd,
                ]
                if any(value is not None for value in values):
                    rows.append([name, lat, lon, moment.isoformat(timespec="minutes"), *values, r.get("temp_source") if temp is not None else None])
    write_csv("fact_meteostat.csv", rows, COLUMNS + ["snow_depth_cm", "temp_source"])


def meteostat_observed(row, key):
    return None if row.get(f"{key}_source") in METEOSTAT_MODEL_SOURCES else number(row.get(key))


def nasa_power():
    parameters = ["T2M", "RH2M", "PRECTOTCORR", "WS10M", "WD10M", "PS"]
    rows = []
    for name, lat, lon, *_ in STATIONS:
        data = fetch_json(
            "https://power.larc.nasa.gov/api/temporal/hourly/point",
            parameters=",".join(parameters), community="RE", latitude=lat, longitude=lon,
            start=f"{YEAR - 1}1231", end=f"{YEAR}1231", format="JSON", **{"time-standard": "UTC"},
        )["properties"]["parameter"]
        for key in data["T2M"]:
            moment = datetime.strptime(key, "%Y%m%d%H") + MSK
            if not in_year(moment):
                continue
            t, rh, prcp, ws, wd, ps = (number(data[p][key]) for p in parameters)
            rows.append([
                name, lat, lon, moment.isoformat(timespec="minutes"),
                t, rh, prcp, None, ws, wd, None, round(ps * 10, 1) if ps is not None else None,
            ])
    write_csv("fact_nasa_power.csv", rows, COLUMNS + ["surface_pressure_hpa"])


def rp5_value(text):
    text = text.strip()
    if text in ("Осадков нет", "Следы осадков", "Облаков нет.", "Облаков нет"):
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def rp5():
    fetch("https://rp5.ru/")
    rows = []
    for name, lat, lon, wmo, _ in STATIONS:
        if not wmo:
            continue
        print(f"  {name}")
        time.sleep(10)
        answer = fetch(
            "https://rp5.ru/responses/reFileSynop.php",
            data={
                "wmo_id": wmo, "a_date1": f"01.01.{YEAR}", "a_date2": f"31.12.{YEAR}",
                "f_ed3": 1, "f_ed4": 1, "f_ed5": 1, "f_pe": 1, "f_pe1": 2, "lng_id": 2, "type": "csv",
            },
            headers={"Referer": "https://rp5.ru/", "X-Requested-With": "XMLHttpRequest"},
        ).decode()
        link = re.search(r"href=(\S+\.csv\.gz)", answer)
        if not link:
            print(f"  {name}: rp5 не отдал файл: {answer[:200]}")
            continue
        text = gzip.decompress(fetch(link.group(1))).decode("utf-8")
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        for r in csv.DictReader(lines, delimiter=";"):
            moment = datetime.strptime(next(iter(r.values())), "%d.%m.%Y %H:%M")
            wind_from = r["DD"].removeprefix("Ветер, дующий с ").strip()
            pressure = rp5_value(r["P"])
            rows.append([
                name, lat, lon, moment.isoformat(timespec="minutes"),
                rp5_value(r["T"]), rp5_value(r["U"]), rp5_value(r["RRR"]),
                round(pressure * 1.33322, 1) if pressure is not None else None,
                rp5_value(r["Ff"]), 0 if wind_from.startswith("Штиль") else RP5_WIND.get(wind_from), rp5_value(r["N"]),
                rp5_value(r["tR"]), rp5_value(r["sss"]), r["WW"].strip(),
            ])
    rows.sort(key=lambda row: (row[0], row[3]))
    write_csv("fact_rp5_synop.csv", rows, COLUMNS + ["precipitation_period_h", "snow_depth_cm", "weather"])


def gismeteo_rows(name, slug):
    html = fetch(f"https://www.gismeteo.ru/weather-{slug}/archive/").decode()
    sections = {part.split('"', 1)[0]: part for part in html.split('data-row="')[1:]}
    years = [int(y) for y in re.findall(r">\s*(20\d\d)\s*<", sections["year-sticky"])]
    series = {
        "temp_max_c": re.findall(r"class='maxt'><temperature-value value=\"(-?[\d.]+)\"", sections["temperature-air"]),
        "temp_min_c": re.findall(r"class='mint'><temperature-value value=\"(-?[\d.]+)\"", sections["temperature-air"]),
        "wind_speed_ms": re.findall(r'wind-value wind-speed.*?value="([\d.]+)"', sections["wind"], re.S),
        "wind_gust_ms": re.findall(r'wind-value wind-gust.*?value="([\d.]+)"', sections["wind"], re.S),
        "snow_depth_cm": re.findall(r'<snow-value class="value[^"]*"[^>]*value="([\d.]+)"', sections["snow-height"]),
        "uv_index": re.findall(r'<div class="row-item[^"]*"[^>]*>\s*(\d+)\s*</div>', sections["radiation"]),
    }
    months = 12 * len(years)
    broken = {key: len(values) for key, values in series.items() if len(values) != months}
    if broken or YEAR not in years:
        raise ValueError(f"Gismeteo {name}: годы {years}, ожидалось {months} значений, получено {broken}")
    start = 12 * years.index(YEAR)
    return [
        [name, f"{YEAR}-{month + 1:02d}", *(values[start + month] for values in series.values())]
        for month in range(12)
    ]


def gismeteo():
    rows = []
    for name, slug in GISMETEO_CITIES.items():
        rows += gismeteo_rows(name, slug)
    columns = ["location", "month", "temp_max_c", "temp_min_c", "wind_speed_ms", "wind_gust_ms", "snow_depth_cm", "uv_index"]
    write_csv("fact_gismeteo_monthly.csv", rows, columns)


SOURCES = {
    "era5": open_meteo_era5,
    "forecasts": open_meteo_forecasts,
    "meteostat": meteostat,
    "nasa": nasa_power,
    "rp5": rp5,
    "gismeteo": gismeteo,
}

if __name__ == "__main__":
    for source in sys.argv[1:] or SOURCES:
        print(source)
        SOURCES[source]()
