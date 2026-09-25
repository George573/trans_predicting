import polars as pl
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

WEEKEND = 1
SHORT_DAY = 2


def _parse_day_overrides(xml_file: Path, year: int) -> dict[date, dict[str, str | int]]:
    root = ET.parse(xml_file).getroot()

    year_from_xml = root.get("year")
    if year_from_xml is None or int(year_from_xml) != year:
        raise ValueError(f"Calendar year mismatch in {xml_file}")

    day_overrides: dict[date, dict[str, str | int]] = {}

    for day in root.findall("days/day"):
        d = day.get("d")
        t = day.get("t")
        if d is None or t is None:
            continue

        month_str, day_str = d.split(".")
        day_date = date(year, int(month_str), int(day_str))

        day_overrides[day_date] = {
            "t": int(t),
            "h": day.get("h") or "",
        }

    return day_overrides


def _build_year_rows(year: int, day_overrides: dict[date, dict[str, str | int]]) -> list[tuple[date, bool, bool, bool]]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)

    rows: list[tuple[date, bool, bool, bool]] = []
    current = start

    while current <= end:
        override = day_overrides.get(current)

        if override:
            day_type = int(override["t"])
            is_weekend = day_type == WEEKEND
            is_short_working_day = day_type == SHORT_DAY
            is_holiday = day_type == WEEKEND and bool(override["h"])
        else:
            is_weekend = current.weekday() in (5, 6)
            is_short_working_day = False
            is_holiday = False

        rows.append((current, is_holiday, is_weekend, is_short_working_day))
        current += timedelta(days=1)

    return rows


def extract_holidays_from_local_file(calendar_path: str) -> pl.DataFrame:
    rows: list[tuple[date, bool, bool, bool]] = []

    day_overrides = _parse_day_overrides(xml_file=calendar_path, year=2025)
    rows.extend(_build_year_rows(year=2025, day_overrides=day_overrides))

    return pl.DataFrame(
        rows,
        orient="row",
        schema=["date", "is_holiday", "is_weekend", "is_short_working_day"],
    )
