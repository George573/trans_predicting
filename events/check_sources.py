from pathlib import Path

import pandas as pd

from check_events import add_norm, effect, fmt, load

DATA = Path(__file__).parent / "data"
DAYTIME = (8, 23)


def with_afisha():
    df = add_norm(load())
    afisha = pd.read_csv(DATA / "afisha_hourly.csv", parse_dates=["date"])
    df = df.merge(afisha, on=["route", "date", "hour"], how="left")
    df = df.sort_values(["route", "date", "hour"]).reset_index(drop=True)
    by_route = df.groupby("route")
    for radius in ("05km", "1km", "3km"):
        df[f"arrive_{radius}"] = by_route[f"afisha_starts_{radius}"].shift(-1, fill_value=0) + df[f"afisha_starts_{radius}"]
        df[f"leave_{radius}"] = by_route[f"afisha_ends_{radius}"].shift(1, fill_value=0) + df[f"afisha_ends_{radius}"]
        df[f"fav_{radius}"] = by_route[f"afisha_fav_starts_{radius}"].shift(-1, fill_value=0) + df[f"afisha_fav_starts_{radius}"]
    return calibrated(df)


def afisha_effects(df):
    base = (df.event_near_route == 0) & (df.route_change == "") & df.hour.between(*DAYTIME)
    print("Мероприятия KudaGo рядом с линией, 8-23 ч, против нормы")
    for radius, title in (("05km", "до 500 м"), ("1km", "до 1 км"), ("3km", "до 3 км")):
        print(f"  {title}")
        for column, label in ((f"arrive_{radius}", "начало в этот или следующий час"), (f"leave_{radius}", "конец в этот или прошлый час")):
            for low, high in ((1, 2), (3, 9), (10, 10**6)):
                mask = base & df[column].between(low, high)
                print(f"    {label}, {low}-{min(high, 99)} сеансов: {fmt(effect(df, mask))}")
        for low, high in ((50, 199), (200, 10**6)):
            mask = base & df[f"fav_{radius}"].between(low, high)
            print(f"    сумма избранного {low}+ у начинающихся: {fmt(effect(df, mask))}")


def calibrated(df):
    quiet = (df.event_near_route == 0) & (df.route_change == "") & (df.arrive_3km == 0) & (df.leave_3km == 0) & df.norm.notna()
    sums = df[quiet].groupby(["route", "hour"])[["boardings", "norm"]].sum()
    df["norm_raw"] = df.norm
    df["norm"] = df.norm * df.set_index(["route", "hour"]).index.map(sums.boardings / sums.norm).values
    return df


def main():
    afisha_effects(with_afisha())


if __name__ == "__main__":
    main()
