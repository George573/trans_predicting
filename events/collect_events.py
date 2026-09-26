import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from build_events import ROUTES, distance_km, route_lines

DATA = Path(__file__).parent / "data"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
KUDAGO = (
    "https://kudago.com/public-api/v1.4/events/?location=msk&page_size=100&expand=place"
    "&fields=id,title,dates,place,categories,tags,is_free,price,age_restriction,favorites_count,comments_count,site_url"
)
PERIOD = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
MSK_OFFSET = pd.Timedelta(hours=3)
SESSION_HOURS = 2
RUNNING_HOURS = 24
NEAR_KM = (0.5, 1.0, 3.0)
HOURS = pd.date_range(PERIOD[0], PERIOD[1], freq="h", inclusive="left")


def kudago_events():
    since, until = ((moment - MSK_OFFSET).timestamp() for moment in PERIOD)
    url, events = f"{KUDAGO}&actual_since={int(since)}&actual_until={int(until)}", []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as response:
            page = json.load(response)
        events += page["results"]
        url = page["next"]
        print(f"  KudaGo: {len(events)} из {page['count']}")
        time.sleep(0.3)
    return events


def kudago_sessions(events):
    rows = []
    for event in events:
        place = event["place"] or {}
        coords = place.get("coords") or {}
        for dates in event["dates"]:
            start = pd.Timestamp(dates["start"], unit="s") + MSK_OFFSET if -1e10 < dates["start"] < 1e10 else None
            end = pd.Timestamp(dates["end"], unit="s") + MSK_OFFSET if -1e10 < dates["end"] < 1e10 else None
            if start is None or end is None or end < PERIOD[0] or start >= PERIOD[1]:
                continue
            date_only = start == start.normalize() and (end - start) % pd.Timedelta(days=1) == pd.Timedelta(0)
            rows.append({
                "kudago_id": event["id"],
                "title": event["title"],
                "categories": " ".join(event["categories"]),
                "start_msk": start,
                "end_msk": end if end > start else start + pd.Timedelta(hours=SESSION_HOURS),
                "time_known": int(not date_only),
                "place": place.get("title"),
                "address": place.get("address"),
                "subway": place.get("subway"),
                "lat": coords.get("lat"),
                "lon": coords.get("lon"),
                "is_free": int(event["is_free"]),
                "price": event["price"],
                "favorites": event["favorites_count"],
                "comments": event["comments_count"],
                "url": event["site_url"],
            })
    sessions = pd.DataFrame(rows).sort_values(["start_msk", "kudago_id"]).reset_index(drop=True)
    sessions["hours"] = ((sessions.end_msk - sessions.start_msk).dt.total_seconds() / 3600).round(2)
    sessions["kind"] = np.where(sessions.hours > RUNNING_HOURS, "running", np.where(sessions.time_known == 1, "session", "day"))
    return sessions


def add_distances(table):
    lines = route_lines()
    places = table[["lat", "lon"]].dropna().drop_duplicates()
    for route in ROUTES:
        distance = {(lat, lon): round(distance_km(lat, lon, lines[route]), 2) for lat, lon in places.itertuples(index=False)}
        table[f"dist_km_{route}"] = [distance.get((lat, lon)) for lat, lon in zip(table.lat, table.lon)]
    return table


def afisha_hourly(sessions):
    frames = []
    timed = sessions[sessions.kind == "session"]
    for route in ROUTES:
        frame = pd.DataFrame(index=HOURS)
        for radius in NEAR_KM:
            near = timed[timed[f"dist_km_{route}"] <= radius]
            starts = near.start_msk.dt.floor("h").value_counts()
            ends = near.end_msk.dt.floor("h").value_counts()
            label = f"{radius:g}km".replace(".", "")
            frame[f"afisha_starts_{label}"] = starts.reindex(HOURS, fill_value=0).values
            frame[f"afisha_ends_{label}"] = ends.reindex(HOURS, fill_value=0).values
            frame[f"afisha_fav_starts_{label}"] = near.groupby(near.start_msk.dt.floor("h")).favorites.sum().reindex(HOURS, fill_value=0).values
        frame.insert(0, "route", route)
        frames.append(frame)
    table = pd.concat(frames).rename_axis("moment").reset_index()
    table.insert(1, "date", table.moment.dt.strftime("%Y-%m-%d"))
    table.insert(2, "hour", table.moment.dt.hour)
    return table.drop(columns="moment")


def main():
    raw = DATA / "kudago_events_raw.json.gz"
    if not raw.exists() or "--refresh" in sys.argv:
        raw.write_bytes(gzip.compress(json.dumps(kudago_events(), ensure_ascii=False).encode()))
    sessions = add_distances(kudago_sessions(json.loads(gzip.decompress(raw.read_bytes()))))
    sessions.to_csv(DATA / "kudago_sessions.csv", index=False)
    print(f"  kudago_sessions.csv: {len(sessions)} сеансов, {sessions.kudago_id.nunique()} мероприятий")
    hourly = afisha_hourly(sessions)
    hourly.to_csv(DATA / "afisha_hourly.csv", index=False)
    print(f"  afisha_hourly.csv: {len(hourly)} строк")


if __name__ == "__main__":
    main()
