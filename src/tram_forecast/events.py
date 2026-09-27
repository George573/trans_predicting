"""Load only scheduled, advance-known event flags from the hourly CSV."""

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

EVENT_FEATURES = (
    "extended_night_service", "event_near_route", "is_citywide_event",
)


@dataclass(frozen=True)
class ScheduledEvents:
    values: np.ndarray  # [routes, features, hours]; NaN means a missing row
    routes: tuple
    start: date

    def window(self, route, start, hours):
        if route not in self.routes:
            raise ValueError(f"Missing scheduled events for route {route}")
        offset = (start - self.start).days * 24
        if offset < 0 or offset + hours > self.values.shape[2]:
            raise ValueError(f"Scheduled events do not cover route {route}, {start}, {hours} hours")
        values = self.values[self.routes.index(route), :, offset:offset + hours]
        missing = np.flatnonzero(~np.isfinite(values).all(axis=0))
        if missing.size:
            hour = int(missing[0])
            raise ValueError(
                f"Missing scheduled events for route {route}, "
                f"{start + timedelta(days=hour // 24)}, hour {hour % 24}"
            )
        return values.copy()


def load_events(path):
    """Cache the full schedule independently of label splits; refresh edited files."""
    path = Path(path).resolve()
    stat = path.stat()
    return _load_events(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_events(path, modified, size):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {"route", "date", "hour", *EVENT_FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Scheduled event CSV requires columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            try:
                route = int(row["route"])
                day = date.fromisoformat(row["date"])
                hour = int(row["hour"])
                values = tuple(float(row[name]) for name in EVENT_FEATURES)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid scheduled event row {line}") from error
            if not 0 <= hour < 24 or any(v not in (0, 1) for v in values):
                raise ValueError(f"Invalid hour or binary event flag at row {line}")
            key = (route, day, hour)
            if key in rows:
                raise ValueError(f"Duplicate scheduled event row: {key}")
            rows[key] = values
    if not rows:
        raise ValueError("Scheduled event CSV is empty")
    routes = tuple(sorted({key[0] for key in rows}))
    start = min(key[1] for key in rows)
    end = max(key[1] for key in rows) + timedelta(days=1)
    values = np.full((len(routes), len(EVENT_FEATURES), (end - start).days * 24),
                     np.nan, dtype=np.float32)
    indices = {route: i for i, route in enumerate(routes)}
    for (route, day, hour), flags in rows.items():
        values[indices[route], :, (day - start).days * 24 + hour] = flags
    values.setflags(write=False)
    return ScheduledEvents(values, routes, start)
