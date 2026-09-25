import csv
import gzip
import html
import http.client
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

YEAR = 2025
MSK = timezone(timedelta(hours=3))
OUT = Path(__file__).parent / "data"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"

TELEGRAM_CHANNELS = {
    "DtOperativno": ("Москва", 24600),
    "cbdd50": ("Московская область", 19600),
}
POST_RE = re.compile(r'data-post="[^"/]+/(\d+)"(.*?)(?=data-post=|\Z)', re.S)
TEXT_RE = re.compile(r'class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>', re.S)
TIME_RE = re.compile(r'datetime="([^"]+)"')
SCORE_RE = re.compile(r"(?<![\d,.])(10|\d)\s*-?\s*(?:ти|ми|х)?\s*балл")
SPEED_RE = re.compile(r"скорость[^0-9]{0,40}(\d{1,3})\s*км/ч")
VEHICLES_RE = re.compile(r"зафиксировано\s+(\d[\d\s]*\d)\s*авт")
VEHICLES_DELTA_RE = re.compile(r"на\s+(\d+(?:,\d+)?)%\s+(больше|меньше)")
FORECAST_RE = re.compile(r"ожида|прогноз|будет|предполага|продлится")
NOT_TRAFFIC_RE = re.compile(r"бофорт|землетряс|егэ|призов|кэшбэк")
POST_TAGS = {
    "congestion": ("балл",),
    "closure": ("закрыт", "перекр", "недоступн"),
    "reopen": ("открыт", "восстановлен"),
    "tram": ("трамва",),
    "accident": ("дтп", "столкнул", "опрокинул"),
    "roadworks": ("ремонт",),
}

GIBDD_REGIONS = {"45": "Москва", "46": "Московская область"}
DATA_MOS = "https://data.mos.ru/api/v2"
MONTHS_RU = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август",
             "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
PARKING = "https://huggingface.co/datasets/matrosovdani/moscow-parking-occupancy/resolve/main/data"


def fetch(url, body=None, headers=None):
    request = urllib.request.Request(url, body, {"User-Agent": UA, **(headers or {})})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
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


def fetch_json(url, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    return json.loads(fetch(url, body, {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}))


def write_csv(name, rows, columns):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"  {name}: {len(rows)} строк")


def ru_date(text):
    return datetime.strptime(text.strip()[:10], "%d.%m.%Y").date() if text and text.strip() else None


def telegram_page(page):
    for post_id, body in POST_RE.findall(page):
        moment = TIME_RE.search(body)
        if not moment:
            continue
        text = TEXT_RE.search(body)
        text = re.sub(r"<br\s*/?>", "\n", text.group(1)) if text else ""
        yield int(post_id), datetime.fromisoformat(moment.group(1)).astimezone(MSK), html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def telegram_posts(channel, before):
    posts = {}
    while True:
        batch = list(telegram_page(fetch(f"https://t.me/s/{channel}?before={before}").decode()))
        if not batch:
            break
        posts |= {post_id: (moment, text) for post_id, moment, text in batch if moment.year == YEAR}
        oldest = min(batch)
        if oldest[1].year < YEAR or oldest[0] >= before:
            break
        if before == TELEGRAM_CHANNELS[channel][1] and max(batch)[1].year == YEAR:
            raise ValueError(f"{channel}: стартовый пост {before} уже в {YEAR} году, увеличьте его")
        before = oldest[0]
        time.sleep(1.5)
    print(f"  {channel}: {len(posts)} постов")
    return sorted(posts.items())


def post_tags(text):
    low = text.lower()
    return "|".join(tag for tag, words in POST_TAGS.items() if any(word in low for word in words))


def congestion(text):
    low = text.lower()
    scores = {}
    for sentence in re.split(r"(?<=[.!?\n])\s+", low):
        if NOT_TRAFFIC_RE.search(sentence):
            continue
        for match in SCORE_RE.finditer(sentence):
            scores.setdefault("forecast" if FORECAST_RE.search(sentence) else "actual", int(match.group(1)))
    if not scores:
        return None
    speed = SPEED_RE.search(low)
    vehicles = VEHICLES_RE.search(low)
    delta = VEHICLES_DELTA_RE.search(low)
    return [
        scores.get("actual"),
        scores.get("forecast"),
        int(speed.group(1)) if speed else None,
        int(re.sub(r"\s", "", vehicles.group(1))) if vehicles else None,
        float(("-" if delta.group(2) == "меньше" else "") + delta.group(1).replace(",", ".")) if delta else None,
    ]


def telegram():
    scores, events = [], []
    for channel, (region, before) in TELEGRAM_CHANNELS.items():
        for post_id, (moment, text) in telegram_posts(channel, before):
            url = f"https://t.me/{channel}/{post_id}"
            time_msk = moment.isoformat(timespec="minutes")
            values = congestion(text)
            if values:
                scores.append([region, time_msk, *values, url])
            if channel == "DtOperativno":
                events.append([time_msk, post_tags(text), url, text])
    write_csv("congestion_telegram.csv", scores, [
        "region", "time_msk", "score", "forecast_score", "avg_speed_kmh", "vehicles_on_roads", "vehicles_vs_prev_month_pct", "url",
    ])
    write_csv("events_deptrans_telegram.csv", events, ["time_msk", "tags", "url", "text"])


def gibdd_cards(region, month):
    start, size = 1, 1000
    while True:
        query = {
            "date": [f"MONTHS:{month}.{YEAR}"], "ParReg": "877", "reg": region, "ind": "1",
            "order": {"type": "1", "fieldName": "dat"}, "st": str(start), "en": str(start + size - 1),
        }
        answer = fetch_json("http://stat.gibdd.ru/map/getDTPCardData", {"data": json.dumps(query, ensure_ascii=False)})["data"]
        cards = json.loads(answer)["tab"] if answer else []
        yield from cards
        time.sleep(2)
        if len(cards) < size:
            return
        start += size


def gibdd_row(card, region):
    info = card.get("infoDtp") or {}
    vehicles = json.dumps(info.get("ts_info") or [], ensure_ascii=False).lower()
    return [
        region, datetime.strptime(f"{card['date']} {card['Time']}", "%d.%m.%Y %H:%M").isoformat(timespec="minutes"),
        card.get("District"), card.get("DTP_V"), card.get("POG"), card.get("RAN"), card.get("K_TS"), card.get("K_UCH"),
        info.get("COORD_W"), info.get("COORD_L"), info.get("street"), info.get("house"), info.get("dor"), info.get("km"),
        info.get("k_ul"), "|".join(info.get("s_pog") or []), info.get("s_pch"), info.get("osv"),
        info.get("change_org_motion"), "|".join(info.get("sdor") or []), "|".join(info.get("ndu") or []),
        "|".join(info.get("OBJ_DTP") or []), int("трамва" in vehicles or "трамва" in (card.get("DTP_V") or "").lower()),
        card["KartId"],
    ]


def gibdd():
    rows = []
    for code, region in GIBDD_REGIONS.items():
        for month in range(1, 13):
            rows += [gibdd_row(card, region) for card in gibdd_cards(code, month)]
        print(f"  {region}: {sum(row[0] == region for row in rows)} ДТП")
    write_csv("accidents_gibdd.csv", rows, [
        "region", "time_msk", "district", "accident_type", "dead", "injured", "vehicles", "participants", "lat", "lon",
        "street", "house", "road", "km", "road_category", "weather", "road_surface", "light", "traffic_change", "place",
        "road_defects", "objects_nearby", "tram_involved", "card_id",
    ])


def data_mos(dataset_id):
    for _ in range(100):
        files = [f for f in fetch_json(f"{DATA_MOS}/odataExports/status", {"datasetId": dataset_id}) if f["format"] == "JSON"]
        if files and files[0]["status"] == "finished":
            archive = zipfile.ZipFile(io.BytesIO(fetch(f"{DATA_MOS}/odata/MEDIA/getFile?id={files[0]['fileId']}")))
            raw = archive.read(archive.namelist()[0])
            try:
                return json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                return json.loads(raw.decode("cp1251"))
        if not files or files[0]["status"] in (None, "error"):
            fetch_json(f"{DATA_MOS}/odataExports/export", {"datasetId": dataset_id, "format": ["JSON"]})
        time.sleep(3)
    raise TimeoutError(f"data.mos.ru не подготовил выгрузку набора {dataset_id}")


def data_mos_congestion():
    rows = sorted([f"{r['Period'][3:]}-{r['Period'][:2]}", r["CapacityRating"], r["CapacityPercentage"]] for r in data_mos(62525))
    write_csv("congestion_monthly_datamos.csv", rows, ["month", "score", "congestion_pct"])


def data_mos_ridership():
    rows = sorted(
        [f"{r['Year']}-{MONTHS_RU.index(r['Month']) + 1:02d}", r["TransportType"], r["PassengerTraffic"]]
        for r in data_mos(62521)
    )
    write_csv("ridership_monthly_datamos.csv", rows, ["month", "transport", "passengers"])


def data_mos_road_repairs():
    rows = []
    for r in data_mos(62101):
        begin = ru_date(r.get("ActualBeginDate")) or ru_date(r.get("WorksBeginDate"))
        end = ru_date(r.get("ActualEndDate")) or ru_date(r.get("PlannedEndDate"))
        if r.get("WorkYear") != YEAR and not ((begin or date.min) <= date(YEAR, 12, 31) and (end or date.max) >= date(YEAR, 1, 1)):
            continue
        lon, lat = (r.get("geodata_center") or {}).get("coordinates") or (None, None)
        rows.append([
            r["WorksPlace"], r["AdmArea"], r["District"], r["WorkYear"], r["WorksType"], r["WorksStatus"],
            r.get("WorksBeginDate"), r.get("PlannedEndDate"), r.get("ActualBeginDate"), r.get("ActualEndDate"),
            lat, lon, json.dumps(r.get("geoData"), ensure_ascii=False), r["global_id"],
        ])
    write_csv("events_road_repairs_datamos.csv", rows, [
        "place", "adm_area", "district", "work_year", "works_type", "status", "planned_begin", "planned_end",
        "actual_begin", "actual_end", "lat", "lon", "geometry", "global_id",
    ])


def data_mos_all():
    data_mos_congestion()
    data_mos_ridership()
    data_mos_road_repairs()


def parking():
    import pandas

    spots = pandas.read_parquet(f"{PARKING}/parking_spots.parquet")
    spots = spots[~spots.feed_silent]
    frames = [pandas.read_parquet(f"{PARKING}/occupancy_{YEAR}-{month:02d}.parquet") for month in range(3, 13)]
    occupancy = pandas.concat(frames)
    occupancy = occupancy[occupancy.parking_id.isin(spots.id)]
    occupancy["time_msk"] = occupancy.time.dt.tz_convert("Europe/Moscow").dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M")
    occupancy = occupancy[occupancy.time_msk.str.startswith(str(YEAR))]
    occupancy["occupancy_rate"] = occupancy.occupancy_rate.clip(0, 100)
    hourly = occupancy.groupby(["parking_id", "time_msk"]).occupancy_rate.mean().round(1).reset_index()
    write_csv("parking_occupancy_hourly.csv", hourly.values.tolist(), ["parking_id", "time_msk", "occupancy_pct"])
    columns = ["id", "name_ru", "address_street_ru", "subway_ru", "latitude", "longitude", "common_spaces"]
    write_csv("parking_spots.csv", spots[columns].values.tolist(), ["parking_id", "name", "address", "metro", "lat", "lon", "spaces"])


def yandex_wayback():
    index = fetch(
        "http://web.archive.org/cdx/search/cdx?url=core-jams-info.maps.yandex.net/info*"
        f"&from={YEAR - 1}1231&to={YEAR + 1}0101&filter=statuscode:200&output=json&fl=timestamp,original"
    )
    readings = {}
    for stamp, original in json.loads(index)[1:]:
        try:
            body = fetch(f"http://web.archive.org/web/{stamp}id_/{original}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            continue
        finally:
            time.sleep(2)
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        record = re.search(r'\{"?regionId"?:\s*"?213"?,[^}]*\}', body.decode("utf-8", "replace"))
        fields = dict(re.findall(r'"?(\w+)"?:\s*"?([^",}]*)"?', record.group(0))) if record else {}
        if "level" not in fields:
            continue
        moment = datetime.strptime(fields["isotime"], "%Y-%m-%dT%H:%M:%S%z").astimezone(MSK)
        if moment.year == YEAR:
            readings[moment] = [int(fields["level"]), round(float(fields["length"]) / 1000, 1), f"https://web.archive.org/web/{stamp}/{original}"]
    rows = [[moment.isoformat(timespec="minutes"), *values] for moment, values in sorted(readings.items())]
    write_csv("congestion_yandex_wayback.csv", rows, ["time_msk", "score", "jams_length_km", "url"])


SOURCES = {
    "yandex": yandex_wayback,
    "telegram": telegram,
    "gibdd": gibdd,
    "datamos": data_mos_all,
    "parking": parking,
}

if __name__ == "__main__":
    for source in sys.argv[1:] or SOURCES:
        print(source)
        SOURCES[source]()
