"""Russian production-calendar flags from bundled yearly XML files."""

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib.resources import files
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class DayFlags:
    is_holiday: bool
    is_day_off: bool
    is_short_working_day: bool


def parse_calendar(xml_file, year: int) -> dict[date, DayFlags]:
    """Read overrides: 1 = day off, 2 = short working day, 3 = working day.

    Both holiday IDs (h) and transfers (f) contribute to is_holiday.
    Incomplete or ambiguous records raise an error.
    """
    root = ET.parse(xml_file).getroot()
    if root.tag != "calendar" or root.get("year") != str(year):
        raise ValueError(f"Calendar year mismatch: expected {year}")
    if root.find("days") is None:
        raise ValueError("Calendar is missing its days section")
    overrides = {}
    for record in root.findall("days/day"):
        raw_day, raw_type = record.get("d"), record.get("t")
        if raw_day is None or raw_type is None:
            raise ValueError("Calendar day must include d and t attributes")
        try:
            month, day = map(int, raw_day.split("."))
            day_date = date(year, month, day)
            day_type = int(raw_type)
            if record.get("f"):
                source_month, source_day = map(int, record.get("f").split("."))
                date(year, source_month, source_day)
        except ValueError as error:
            raise ValueError(f"Invalid calendar day: {raw_day}") from error
        if day_type not in (1, 2, 3):
            raise ValueError(f"Unknown calendar day type {day_type} for {day_date}")
        if day_date in overrides:
            raise ValueError(f"Duplicate calendar day: {day_date}")
        holiday = bool(record.get("h") or record.get("f"))
        if holiday and day_type != 1:
            raise ValueError(f"Holiday or transfer must be a day off: {day_date}")
        overrides[day_date] = DayFlags(holiday, day_type == 1, day_type == 2)
    return overrides


@lru_cache(maxsize=None)
def _year_overrides(year: int) -> dict[date, DayFlags]:
    resource = files("tram_forecast").joinpath("calendars", f"{year}.xml")
    if not resource.is_file():
        raise ValueError(
            f"No Russian production calendar for {year}; add calendars/{year}.xml"
        )
    with resource.open("rb") as handle:
        return parse_calendar(handle, year)


def day_flags(day: date) -> DayFlags:
    """Apply explicit overrides before the usual Saturday/Sunday rule."""
    overrides = _year_overrides(day.year)
    return overrides.get(day, DayFlags(False, day.weekday() >= 5, False))


def is_holiday(day: date) -> bool:
    """Public holiday or transferred day off; ordinary weekends are excluded."""
    return day_flags(day).is_holiday
