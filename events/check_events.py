"""Effect of holidays, events and route changes on tram boardings, January-October 2025.

Norm: median boardings of the same route and hour on the same weekday 1, 2 and 3 weeks before and after.
Days off by the production calendar, short days, hours with a route change and hours near an event
are excluded from the norm. Effect = sum(boardings) / sum(norm) - 1 over the hours of the group,
95% CI by bootstrap over days. Run after build_events.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent / "data"
RNG = np.random.default_rng(0)
SHIFTS = (7, 14, 21, -7, -14, -21)


def load():
    labels = pd.concat(
        [pd.read_csv(ROOT / "dataset" / "labels" / f"labels_day_{part}.csv", sep=";") for part in ("train", "test")])
    labels["date"] = pd.to_datetime(labels.date)
    feats = pd.read_csv(DATA / "events_hourly.csv", parse_dates=["date"], keep_default_na=False,
                        na_values={"event_dist_km": [""]})
    feats = feats[feats.date <= "2025-10-31"]
    df = feats.merge(labels, on=["route", "date", "hour"], how="left")
    df = df[df.route != 5].copy()
    df["boardings"] = df.boardings.fillna(0)
    return df


def add_norm(df, column="norm", same_dow=True, sunday=False):
    clean = df.copy()
    bad = (clean.is_holiday_or_transfer == 1) | (clean.is_short_workday == 1) | (clean.route_change != "") | (
        clean.event_near_route == 1) | (clean.extended_night_service == 1) | (clean.new_year_period == 1)
    clean.loc[bad, "boardings"] = np.nan
    table = clean.pivot_table(index=["date", "hour"], columns="route", values="boardings", dropna=False)
    shifted = []
    for k in SHIFTS:
        s = table.copy()
        s.index = pd.MultiIndex.from_arrays([s.index.get_level_values(0) + pd.Timedelta(days=k), s.index.get_level_values(1)])
        shifted.append(s.reindex(table.index).values)
    norm = pd.DataFrame(np.nanmedian(np.stack(shifted), axis=0), index=table.index, columns=table.columns)
    if sunday:
        # For days off: compare with the nearest regular Sundays instead of the same weekday.
        sundays = table[table.index.get_level_values(0).dayofweek == 6]
        dates = table.index.get_level_values(0)
        rows = []
        for d in dates.unique():
            near = sundays.loc[(sundays.index.get_level_values(0) - d).map(abs).map(lambda x: x.days <= 21)]
            rows.append(near.groupby(level=1).median().assign(date=d))
        sun = pd.concat(rows).reset_index().set_index(["date", "hour"])
        norm = sun.reindex(table.index)
    long = norm.stack(future_stack=True).rename(column).reset_index()
    return df.merge(long, on=["date", "hour", "route"], how="left")


def effect(df, mask, norm="norm", n_boot=400):
    part = df[mask & df[norm].notna() & (df[norm] > 0)]
    if part.empty:
        return None
    by_day = part.groupby("date")[["boardings", norm]].sum()
    point = by_day.boardings.sum() / by_day[norm].sum() - 1
    idx = RNG.integers(0, len(by_day), (n_boot, len(by_day)))
    obs, nrm = by_day.boardings.values[idx].sum(1), by_day[norm].values[idx].sum(1)
    low, high = np.percentile(obs / nrm - 1, [2.5, 97.5])
    return len(by_day), len(part), point, low, high


def fmt(res):
    if res is None:
        return "нет данных"
    days, hours, point, low, high = res
    return f"{point * 100:+.1f}% [{low * 100:+.1f}; {high * 100:+.1f}], дней {days}, маршруто-часов {hours}"


def main():
    df = load()
    df = add_norm(df)
    df = add_norm(df, "norm_sunday", sunday=True)
    day = (df.hour >= 6) & (df.hour <= 22)
    plain = (df.route_change == "") & (df.event_near_route == 0)
    rows = []

    def add(section, name, mask, norm="norm"):
        res = effect(df, mask & plain, norm)
        rows.append((section, name, fmt(res)))
        print(f"{section:12} {name:60} {fmt(res)}")

    cal = "календарь"
    for label, dates in {
        "Новогодние каникулы 1-8 января": ("2025-01-01", "2025-01-08"),
        "Майские праздники 1-4 и 8-11 мая": None,
        "12-15 июня": ("2025-06-12", "2025-06-15"),
    }.items():
        if dates:
            m = df.date.between(*dates)
        else:
            m = df.date.between("2025-05-01", "2025-05-04") | df.date.between("2025-05-08", "2025-05-11")
        add(cal, f"{label}: против обычного воскресенья", m, "norm_sunday")
    add(cal, "Праздник или перенесённый выходной в будний день: против воскресенья",
        (df.is_holiday_or_transfer == 1) & (df.date.dt.dayofweek < 5) & (df.new_year_period == 0), "norm_sunday")
    add(cal, "Праздник или перенос в будний день: против того же дня недели",
        (df.is_holiday_or_transfer == 1) & (df.date.dt.dayofweek < 5) & (df.new_year_period == 0))
    add(cal, "Выходные внутри длинных праздников (сб, вс): против той же субботы/воскресенья",
        (df.off_run_len >= 3) & (df.date.dt.dayofweek >= 5) & (df.new_year_period == 0) & (df.is_holiday_or_transfer == 0))
    add(cal, "Первая рабочая неделя после Нового года", df.first_workweek_after_new_year == 1)
    add(cal, "Сокращённый предпраздничный день, весь день", df.is_short_workday == 1)
    add(cal, "Сокращённый день, 15-16 ч", (df.is_short_workday == 1) & df.hour.between(15, 16))
    add(cal, "Сокращённый день, 17-19 ч", (df.is_short_workday == 1) & df.hour.between(17, 19))
    add(cal, "Рабочий день перед длинными выходными (не сокращённый)",
        (df.is_workday_before_long_weekend == 1) & (df.is_short_workday == 0))
    add(cal, "Рабочий день сразу после длинных выходных",
        (df.days_after_long_weekend == 1) & (df.is_day_off == 0) & (df.first_workweek_after_new_year == 0))
    for name, (a, b) in {"весенние школьные каникулы 24-28 марта": ("2025-03-24", "2025-03-28"),
                         "каникулы 1 классов 17-21 февраля": ("2025-02-17", "2025-02-21"),
                         "осенние школьные каникулы 27-31 октября": ("2025-10-27", "2025-10-31")}.items():
        add(cal, f"Будни, {name}", df.date.between(a, b) & day)
    add(cal, "Первая неделя сентября (начало учебного года), будни", (df.school_year_first_week == 1) & (df.date.dt.dayofweek < 5) & day)
    add(cal, "Дни посещения кладбищ (Вербное, Пасха, Красная горка, Радоница, Троица), 8-18 ч",
        (df.is_cemetery_day == 1) & df.hour.between(8, 18))
    add(cal, "День города 13-14 сентября, 10-23 ч", (df.is_city_day == 1) & df.hour.between(10, 23))
    add(cal, "Ночи с продлённой работой транспорта, 0-3 ч (Рождество, Крещение, Пасха, День города)",
        (df.extended_night_service == 1) & (df.new_year_period == 0) | (df.extended_night_service == 1) & (df.date == "2025-01-07"))

    ev = "события"
    near = df.event_near_route == 1
    for phase, name in (("pre", "за 2 часа до начала"), ("during", "во время"), ("post", "2 часа после окончания")):
        m = (df.event_category == "football") & (df.event_phase == phase)
        res = effect(df, m & near & (df.route_change == ""))
        rows.append((ev, f"Матч РПЛ в 1.5 км от линии (РЖД Арена - маршрут 7), {name}", fmt(res)))
        print(f"{ev:12} football near {phase:8} {fmt(res)}")
    for route, venue_note in ((28, "Лукойл Арена в 1.55 км"), (26, "Лужники в 2.8 км"), (7, "РЖД Арена в 1.3 км"),
                              (11, "РЖД Арена в 1.8 км"), (12, "РЖД Арена в 1.9 км")):
        for phase in ("pre", "post"):
            m = (df.route == route) & (df.event_phase == phase) & (df.event_dist_km <= 3) & (df.route_change == "")
            for cat in ("football", "concert"):
                res = effect(df, m & (df.event_category == cat))
                if res:
                    rows.append((ev, f"Маршрут {route}, {cat}, {venue_note}, {phase}", fmt(res)))
                    print(f"{ev:12} route {route} {cat} {phase}: {fmt(res)}")
    m = (df.event_id.isin(["e008"])) & (df.route == 26) & (df.event_phase == "post")
    rows.append((ev, "Финал Кубка России 1 июня, маршрут 26, 2 часа после матча", fmt(effect(df, m))))
    vdnh = df.route.isin([11, 17, 25]) & (df.event_dist_km < 0.5) & (df.route_change == "")
    for cat in ("city_festival", "sport_run", "forum"):
        for phase in ("pre", "during", "post"):
            res = effect(df, vdnh & (df.event_category == cat) & (df.event_phase == phase))
            if res:
                rows.append((ev, f"ВДНХ, {cat}, маршруты 11/17/25, {phase}", fmt(res)))
                print(f"{ev:12} vdnh {cat} {phase}: {fmt(res)}")

    rc = "перекрытия"
    for kind in ("cancel", "short", "reroute"):
        m = df.route_change == kind
        part = df[m & df.norm.notna()]
        if part.empty:
            continue
        by = part.groupby("date")[["boardings", "norm"]].sum()
        point = by.boardings.sum() / by.norm.sum() - 1
        rows.append((rc, f"Изменение маршрута: {kind}", f"{point * 100:+.1f}%, дней {len(by)}, маршруто-часов {len(part)}"))
        print(f"{rc:12} {kind:8} {point * 100:+.1f}% days {len(by)}")

    out = pd.DataFrame(rows, columns=["раздел", "условие", "посадки против нормы [95% ДИ]"])
    out.to_csv(DATA / "effects_report.csv", index=False)

    # Hourly profile of a short pre-holiday day relative to the norm, all routes.
    prof = df[(df.is_short_workday == 1) & df.norm.notna()].groupby("hour")[["boardings", "norm"]].sum()
    prof["ratio"] = (prof.boardings / prof.norm).round(3)
    prof.to_csv(DATA / "short_day_profile.csv")
    print(prof.loc[12:21, "ratio"].to_string())


if __name__ == "__main__":
    main()
