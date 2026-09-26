"""Strict source parsing and shared calendar contracts."""

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np


@dataclass(frozen=True, order=True)
class SampleIdentity:
    route: int
    cutoff: date
    requested: date

    @property
    def lead(self):
        return (self.requested - self.cutoff).days + 1


def parse_route(value):
    match = re.fullmatch(r"(\d+)(?:\s+трамвай)?", str(value).strip())
    if not match:
        raise ValueError(f"invalid route: {value!r}")
    return int(match.group(1))


def calendar(timestamps):
    """Return [6,T], including calendar information for empty hours."""
    t = list(timestamps)
    phases = (
        np.array(
            [[x.hour / 24, x.weekday() / 7, (x.day - 1) / 31] for x in t],
            dtype=np.float64,
        )
        * 2
        * np.pi
    )
    return np.stack(
        [f(phases[:, i]) for i in range(3) for f in (np.sin, np.cos)]
    ).astype(np.float32)


def request_calendar(days):
    return calendar([datetime.combine(d, time.min) for d in days])[2:].T


def label_grid(paths, routes, start, end):
    """Validate all rows; fill [start,end). Return counts and source-presence grids."""
    if end <= start:
        raise ValueError("end must follow start")
    width = (end - start).days * 24
    values = np.zeros((len(routes), width), dtype=np.float32)
    present = np.zeros_like(values, dtype=bool)
    route_index = {r: i for i, r in enumerate(routes)}
    seen = set()
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if not {"route", "date", "hour", "boardings"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError(f"{path}: missing label columns")
            for line, row in enumerate(reader, 2):
                try:
                    r = parse_route(row["route"])
                    d = date.fromisoformat(row["date"])
                    h = int(row["hour"])
                    y = int(row["boardings"])
                    if not 0 <= h < 24 or y < 0:
                        raise ValueError("invalid hour/count")
                    key = (r, d, h)
                    if key in seen:
                        raise ValueError(f"duplicate label key {key}")
                    seen.add(key)
                    if r in route_index and start <= d < end:
                        i = route_index[r]
                        j = (d - start).days * 24 + h
                        values[i, j] = y
                        present[i, j] = True
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line}: {exc}") from exc
    return values, present


def count_scale(counts):
    if counts.size == 0 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("invalid count grid")
    return max(1.0, float(counts.mean(dtype=np.float64)))
