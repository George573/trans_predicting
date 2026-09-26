"""Calendar, events and route changes for Moscow trams, 2025-2026.

Output (events/data):
    calendar_daily.csv   production calendar, school holidays, church and city holidays, one row per day
    football_moscow.csv  home matches of Moscow clubs in the Russian Premier League
    events.csv           all events with venue coordinates and distance to each tram route
    events_hourly.csv    route x hour features for 2025-2026, ready to join on (route, date, hour)

Inputs: events_manual.csv and route_changes.csv (curated by hand from Deptrans posts and venue sites),
football_rpl_raw.csv (football-data.co.uk), ../factors/data/route_lines.geojson (OpenStreetMap).
Standard library and pandas only.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")
NEAR_KM = 1.5
PRE_HOURS, POST_HOURS = 2, 2
KM_PER_DEGREE = np.array([111.32 * np.cos(np.radians(55.75)), 110.57])

# Coordinates of the main entrance or the stadium bowl, checked against Yandex Maps, about 100 m accuracy.
VENUES = {
    "luzhniki": ("БСА Лужники", 55.7158, 37.5537),
    "vdnh": ("ВДНХ, главный вход", 55.8263, 37.6377),
    "sokolniki": ("Парк Сокольники, главный вход", 55.7925, 37.6773),
    "rzd_arena": ("РЖД Арена (Локомотив)", 55.8033, 37.7417),
    "vtb_arena": ("ВТБ Арена (Динамо)", 55.7916, 37.5596),
    "veb_arena": ("ВЭБ Арена (ЦСКА)", 55.7911, 37.5343),
    "lukoil_arena": ("Лукойл Арена (Спартак)", 55.8178, 37.4403),
    "center": ("центр Москвы", 55.7539, 37.6208),
    "citywide": ("вся Москва", None, None),
}
HOME_VENUE = {
    "Lokomotiv Moscow": "rzd_arena",
    "Dynamo Moscow": "vtb_arena",
    "CSKA Moscow": "veb_arena",
    "Spartak Moscow": "lukoil_arena",
}

# Production calendar: Labour Code art. 112, decrees No. 1335 of 04.10.2024 (2025) and No. 1466 of 24.09.2025 (2026).
HOLIDAY_NAMES = {
    (1, 1): "Новогодние каникулы", (1, 2): "Новогодние каникулы", (1, 3): "Новогодние каникулы",
    (1, 4): "Новогодние каникулы", (1, 5): "Новогодние каникулы", (1, 6): "Новогодние каникулы",
    (1, 7): "Рождество Христово", (1, 8): "Новогодние каникулы", (2, 23): "День защитника Отечества",
    (3, 8): "Международный женский день", (5, 1): "Праздник Весны и Труда", (5, 9): "День Победы",
    (6, 12): "День России", (11, 4): "День народного единства",
}
TRANSFERRED_OFF = {
    2025: ["2025-05-02", "2025-05-08", "2025-06-13", "2025-11-03", "2025-12-31"],
    2026: ["2026-01-09", "2026-03-09", "2026-05-11", "2026-12-31"],
}
SHORT_DAYS = {
    2025: ["2025-03-07", "2025-04-30", "2025-06-11", "2025-11-01"],
    2026: ["2026-04-30", "2026-05-08", "2026-06-11", "2026-11-03"],
}
WORKING_WEEKENDS = ["2025-11-01"]

# Moscow schools, quarter system, recommendations of the Moscow Department of Education.
SCHOOL_HOLIDAYS = [
    ("2024-12-29", "2025-01-08", "зимние"),
    ("2025-02-15", "2025-02-23", "дополнительные для 1 классов"),
    ("2025-03-22", "2025-03-30", "весенние"),
    ("2025-05-27", "2025-08-31", "летние"),
    ("2025-10-25", "2025-11-04", "осенние"),
    ("2025-12-31", "2026-01-11", "зимние"),
    ("2026-02-21", "2026-03-01", "дополнительные для 1 классов"),
    ("2026-03-21", "2026-03-29", "весенние"),
    ("2026-05-27", "2026-08-31", "летние"),
    ("2026-10-24", "2026-11-04", "осенние"),
    ("2026-12-31", "2026-12-31", "зимние"),
]
ORTHODOX_EASTER = {2025: date(2025, 4, 20), 2026: date(2026, 4, 12)}
CITY_DAY = {2025: ["2025-09-13", "2025-09-14"], 2026: ["2026-09-05", "2026-09-06"]}


def calendar_daily():
    days = pd.date_range(START, END, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["dow"] = cal.date.dt.dayofweek
    transferred = pd.to_datetime(sum(TRANSFERRED_OFF.values(), []))
    short = pd.to_datetime(sum(SHORT_DAYS.values(), []))
    working_weekend = pd.to_datetime(WORKING_WEEKENDS)
    names = [HOLIDAY_NAMES.get((d.month, d.day), "") for d in cal.date]
    cal["holiday_name"] = names
    is_holiday = cal.holiday_name != ""
    is_transfer = cal.date.isin(transferred)
    is_weekend = (cal.dow >= 5) & ~cal.date.isin(working_weekend)
    cal["day_type"] = np.select(
        [is_holiday, is_transfer, is_weekend, cal.date.isin(short)],
        ["holiday", "transferred_day_off", "weekend", "short_workday"],
        "workday",
    )
    cal.loc[cal.date.isin(working_weekend), "day_type"] = "working_saturday_short"
    cal.loc[is_transfer & (cal.holiday_name == ""), "holiday_name"] = "перенесённый выходной"
    cal["is_day_off"] = cal.day_type.isin(["holiday", "transferred_day_off", "weekend"]).astype(int)
    cal["is_holiday_or_transfer"] = cal.day_type.isin(["holiday", "transferred_day_off"]).astype(int)
    cal["is_short_workday"] = cal.day_type.isin(["short_workday", "working_saturday_short"]).astype(int)

    # Runs of consecutive days off: length and position inside the run.
    run_id = (cal.is_day_off != cal.is_day_off.shift()).cumsum()
    run_len = cal.groupby(run_id).is_day_off.transform("size") * cal.is_day_off
    cal["off_run_len"] = run_len
    cal["off_run_pos"] = (cal.groupby(run_id).cumcount() + 1) * cal.is_day_off
    long_run = run_len >= 3
    # Days until the next long run of days off and since the end of the previous one.
    idx = np.arange(len(cal))
    starts = idx[long_run & ~long_run.shift(fill_value=False)]
    ends = idx[long_run & ~long_run.shift(-1, fill_value=False)]
    cal["days_to_long_weekend"] = [int(min((s - i for s in starts if s > i), default=99)) for i in idx]
    cal["days_after_long_weekend"] = [int(min((i - e for e in ends if e < i), default=99)) for i in idx]
    cal.loc[long_run, ["days_to_long_weekend", "days_after_long_weekend"]] = 0
    cal["is_workday_before_long_weekend"] = ((cal.days_to_long_weekend == 1) & (cal.is_day_off == 0)).astype(int)

    cal["new_year_period"] = 0
    cal["first_workweek_after_new_year"] = 0
    for year in (2025, 2026):
        jan = cal.date.dt.year == year
        ny_end = cal[jan & (cal.date.dt.month == 1) & (cal.is_day_off == 1)].date
        ny_end = ny_end[ny_end.diff().dt.days.fillna(1).eq(1).cumprod().astype(bool)].max()
        cal.loc[(cal.date >= f"{year}-01-01") & (cal.date <= ny_end), "new_year_period"] = 1
        workdays = cal[(cal.date > ny_end) & (cal.is_day_off == 0)].date.head(5)
        cal.loc[cal.date.isin(workdays), "first_workweek_after_new_year"] = 1
    cal.loc[cal.date.dt.strftime("%m-%d") == "12-31", "new_year_period"] = 1
    cal["may_holidays"] = ((cal.date.dt.month == 5) & (cal.date.dt.day <= 11)).astype(int)

    cal["school_holiday"] = ""
    for first, last, name in SCHOOL_HOLIDAYS:
        cal.loc[(cal.date >= first) & (cal.date <= last), "school_holiday"] = name
    cal["is_school_holiday"] = (cal.school_holiday != "").astype(int)
    cal["is_school_summer"] = (cal.school_holiday == "летние").astype(int)
    cal["school_year_first_week"] = cal.date.dt.strftime("%m-%d").between("09-01", "09-07").astype(int)

    church = {}
    for year, easter in ORTHODOX_EASTER.items():
        church[easter - timedelta(days=49)] = "Прощёное воскресенье (конец Масленицы)"
        church[easter - timedelta(days=7)] = "Вербное воскресенье"
        church[easter - timedelta(days=1)] = "Великая суббота, пасхальная служба ночью"
        church[easter] = "Пасха"
        church[easter + timedelta(days=7)] = "Красная горка"
        church[easter + timedelta(days=9)] = "Радоница"
        church[easter + timedelta(days=49)] = "Троица"
        church[date(year, 1, 6)] = "Сочельник, рождественская служба ночью"
        church[date(year, 1, 18)] = "Крещенский сочельник, купания ночью"
    cal["church_day"] = [church.get(d.date(), "") for d in cal.date]
    cal["is_cemetery_day"] = cal.church_day.isin(["Вербное воскресенье", "Пасха", "Красная горка", "Радоница", "Троица"]).astype(int)
    cal["is_city_day"] = cal.date.isin(pd.to_datetime(sum(CITY_DAY.values(), []))).astype(int)
    cal["date"] = cal.date.dt.date
    return cal


def extended_night_dates():
    """Nights with public transport running past the usual 1:00. Value is the date of the early morning hours."""
    nights = []
    for year, easter in ORTHODOX_EASTER.items():
        nights += [date(year, 1, 1), date(year, 1, 7), date(year, 1, 19), easter]
    nights += [date(2025, 9, 14), date(2026, 9, 6)]
    return set(nights)


def football():
    raw = pd.read_csv(DATA / "football_rpl_raw.csv")
    moments = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M")
    # football-data.co.uk gives UK time: MSK = UK + 2 in British summer time, + 3 in winter.
    bst = pd.Series([_is_bst(t) for t in moments])
    raw["kickoff_msk"] = moments + pd.to_timedelta(np.where(bst, 2, 3), unit="h")
    games = raw[raw.Home.isin(HOME_VENUE) & (raw.kickoff_msk >= START)].copy()
    games = games.rename(columns={"Season": "season", "Home": "home", "Away": "away"})
    games["venue"] = games.home.map(HOME_VENUE)
    return games[["season", "kickoff_msk", "home", "away", "venue"]].sort_values("kickoff_msk")


def _is_bst(moment):
    year = moment.year
    last_sunday = lambda month: max(pd.date_range(f"{year}-{month:02d}-24", periods=8, freq="D")
                                    .to_series().loc[lambda s: (s.dt.month == month) & (s.dt.dayofweek == 6)])
    return last_sunday(3) + pd.Timedelta(hours=1) <= moment < last_sunday(10) + pd.Timedelta(hours=1)


def route_lines():
    geo = json.loads((ROOT / "factors" / "data" / "route_lines.geojson").read_text())
    return {f["properties"]["route"]: [np.array(line) for line in f["geometry"]["coordinates"]] for f in geo["features"]}


def distance_km(lat, lon, lines):
    point = np.array([lon, lat]) * KM_PER_DEGREE
    best = np.inf
    for line in lines:
        xy = line[:, :2] * KM_PER_DEGREE
        a, b = xy[:-1], xy[1:]
        ab = b - a
        t = np.clip(((point - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        best = min(best, np.hypot(*(a + ab * t[:, None] - point).T).min())
    return best


def events_table(games):
    manual = pd.read_csv(DATA / "events_manual.csv", parse_dates=["start_msk", "end_msk"])
    matches = pd.DataFrame({
        "event_id": [f"f{i:03d}" for i in range(1, len(games) + 1)],
        "name": games.home.str.replace(" Moscow", "") + " - " + games.away + ", РПЛ " + games.season,
        "category": "football",
        "start_msk": games.kickoff_msk.values,
        "end_msk": (games.kickoff_msk + pd.Timedelta(minutes=110)).values,
        "venue": games.venue.values,
        "routes": "",
        "source": "https://www.football-data.co.uk/new/RUS.csv",
    })
    events = pd.concat([manual, matches], ignore_index=True)
    lines = route_lines()
    events["venue_name"] = events.venue.map(lambda v: VENUES[v][0])
    events["lat"] = events.venue.map(lambda v: VENUES[v][1])
    events["lon"] = events.venue.map(lambda v: VENUES[v][2])
    for route in ROUTES:
        events[f"dist_km_{route}"] = [
            round(distance_km(lat, lon, lines[route]), 2) if pd.notna(lat) else np.nan
            for lat, lon in zip(events.lat, events.lon)
        ]
    dist_cols = [f"dist_km_{r}" for r in ROUTES]
    events["routes_near"] = events[dist_cols].apply(
        lambda row: " ".join(str(r) for r, d in zip(ROUTES, row) if d <= NEAR_KM), axis=1)
    return events.sort_values("start_msk").reset_index(drop=True)


def route_change_hours():
    """(route, date, hour) -> change kind, expanded from route_changes.csv."""
    changes = pd.read_csv(DATA / "route_changes.csv", dtype=str)
    out = {}
    for c in changes.itertuples():
        routes = [int(r) for r in c.routes.split()]
        first, last = pd.Timestamp(c.date_from), pd.Timestamp(c.date_to)
        h0, h1 = int(c.hour_from), int(c.hour_to)
        if c.mode == "span":
            moments = pd.date_range(first + pd.Timedelta(hours=h0), last + pd.Timedelta(hours=h1), freq="h", inclusive="left")
        else:
            moments = []
            for day in pd.date_range(first, last, freq="D"):
                if c.days == "weekend" and day.dayofweek < 5:
                    continue
                moments += list(pd.date_range(day + pd.Timedelta(hours=h0), day + pd.Timedelta(hours=h1), freq="h", inclusive="left"))
        for moment in moments:
            for route in routes:
                out[(route, moment.normalize(), moment.hour)] = c.kind
    return out


def hourly(cal, events):
    hours = pd.date_range(START, END + pd.Timedelta(hours=23), freq="h")
    base = pd.DataFrame({"moment": hours})
    base["date"] = base.moment.dt.date
    base["hour"] = base.moment.dt.hour
    cal_cols = ["date", "day_type", "is_day_off", "is_holiday_or_transfer", "is_short_workday", "off_run_len",
                "off_run_pos", "days_to_long_weekend", "days_after_long_weekend", "is_workday_before_long_weekend",
                "new_year_period", "first_workweek_after_new_year", "may_holidays", "is_school_holiday",
                "is_school_summer", "school_year_first_week", "is_cemetery_day", "is_city_day"]
    base = base.merge(cal[cal_cols], on="date", how="left")
    nights = extended_night_dates()
    base["extended_night_service"] = (base.date.isin(nights) & (base.hour <= 3)).astype(int)

    frames = []
    changes = route_change_hours()
    citywide = events[events.venue.isin(["citywide", "center"])]
    for route in ROUTES:
        frame = base.copy()
        frame.insert(0, "route", route)
        dist = events[f"dist_km_{route}"]
        local = events[dist.notna()].assign(dist=dist[dist.notna()])
        frame["event_dist_km"] = np.nan
        frame["event_phase"] = ""
        frame["event_category"] = ""
        frame["event_id"] = ""
        for e in local.itertuples():
            window = (frame.moment >= e.start_msk.floor("h") - pd.Timedelta(hours=PRE_HOURS)) & (
                frame.moment <= e.end_msk + pd.Timedelta(hours=POST_HOURS))
            closer = window & ~(frame.event_dist_km <= e.dist)
            if not closer.any():
                continue
            phase = np.where(frame.moment[closer] < e.start_msk.floor("h"), "pre",
                             np.where(frame.moment[closer] <= e.end_msk, "during", "post"))
            frame.loc[closer, "event_dist_km"] = e.dist
            frame.loc[closer, "event_phase"] = phase
            frame.loc[closer, "event_category"] = e.category
            frame.loc[closer, "event_id"] = e.event_id
        frame["event_near_route"] = (frame.event_dist_km <= NEAR_KM).astype(int)
        frame["citywide_event"] = ""
        for e in citywide.itertuples():
            window = (frame.moment >= e.start_msk.floor("h")) & (frame.moment <= e.end_msk)
            frame.loc[window, "citywide_event"] = e.event_id
        keys = list(zip(frame.route, pd.to_datetime(frame.date), frame.hour))
        frame["route_change"] = [changes.get(k, "") for k in keys]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).drop(columns="moment")
    return out


def main():
    DATA.mkdir(exist_ok=True)
    cal = calendar_daily()
    cal.to_csv(DATA / "calendar_daily.csv", index=False)
    games = football()
    games.to_csv(DATA / "football_moscow.csv", index=False)
    events = events_table(games)
    events.to_csv(DATA / "events.csv", index=False)
    features = hourly(cal, events)
    features.to_csv(DATA / "events_hourly.csv", index=False)
    print(f"calendar {len(cal)} days, football {len(games)} matches, events {len(events)}, hourly {len(features)} rows")
    print(events.groupby("category").size().to_string())
    near = events[events.routes_near != ""]
    print(f"events within {NEAR_KM} km of a route: {len(near)}")
    print(near.groupby(["venue", "routes_near"]).size().to_string())


if __name__ == "__main__":
    main()
