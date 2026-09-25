"""Stream route-hour history windows from the provided boarding labels.

The input files are read sequentially. Memory use depends on the requested
window and horizon, rather than on the number of rows in the dataset.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from itertools import groupby
from pathlib import Path
from typing import Iterator, Sequence


HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class HourlyCount:
    route: int
    timestamp: datetime
    boardings: int
    observed: bool


@dataclass(frozen=True)
class BoardingWindow:
    route: int
    history_start: datetime
    history_end: datetime
    target_time: datetime
    history_boardings: tuple[int, ...]
    history_observed: tuple[bool, ...]
    target_boardings: int
    target_observed: bool


def _read_labels(handle, routes: set[int] | None) -> Iterator[HourlyCount]:
    reader = csv.DictReader(handle, delimiter=";")
    required = {"route", "date", "hour", "boardings"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing label columns: {required - set(reader.fieldnames or [])}")

    previous_key = None
    for row in reader:
        route = int(row["route"])
        timestamp = datetime.combine(date.fromisoformat(row["date"]), time(int(row["hour"])))
        boardings = int(row["boardings"])
        key = (route, timestamp)
        if previous_key is not None and key <= previous_key:
            raise ValueError("Each label file must be sorted by route, date and hour without duplicates")
        previous_key = key
        if boardings < 0:
            raise ValueError(f"Negative boardings at {key}")
        if routes is None or route in routes:
            yield HourlyCount(route, timestamp, boardings, True)


def iter_hourly_counts(
    paths: Sequence[str | Path],
    *,
    series_start: date,
    series_end: date,
    routes: set[int] | None = None,
) -> Iterator[HourlyCount]:
    """Merge sorted label files and fill absent hours with zero counts.

    No output is produced for a route with no label rows, such as route 5.
    ``observed=False`` marks a filled hour, which may also reflect missing data.
    """
    if not paths:
        raise ValueError("At least one label path is required")
    if series_end < series_start:
        raise ValueError("series_end must be on or after series_start")

    first_hour = datetime.combine(series_start, time.min)
    last_hour = datetime.combine(series_end, time(23))

    with ExitStack() as stack:
        streams = [
            _read_labels(stack.enter_context(open(path, encoding="utf-8-sig", newline="")), routes)
            for path in paths
        ]
        merged = heapq.merge(*streams, key=lambda item: (item.route, item.timestamp))
        for route, route_rows in groupby(merged, key=lambda item: item.route):
            next_hour = first_hour
            seen_in_range = False
            for row in route_rows:
                if row.timestamp < first_hour or row.timestamp > last_hour:
                    continue
                seen_in_range = True
                if row.timestamp < next_hour:
                    raise ValueError(f"Duplicate route-hour across label files: {route}, {row.timestamp}")
                while next_hour < row.timestamp:
                    yield HourlyCount(route, next_hour, 0, False)
                    next_hour += HOUR
                yield row
                next_hour = row.timestamp + HOUR
            if not seen_in_range:
                continue
            while next_hour <= last_hour:
                yield HourlyCount(route, next_hour, 0, False)
                next_hour += HOUR


def iter_boarding_windows(
    paths: Sequence[str | Path],
    *,
    series_start: date,
    series_end: date,
    history_hours: int,
    horizon_hours: int = 1,
    stride: int = 1,
    target_start: date | None = None,
    target_end: date | None = None,
    routes: set[int] | None = None,
) -> Iterator[BoardingWindow]:
    """Yield fixed length history and one target at the requested horizon.

    For horizon 1, the target is the hour immediately after the history.
    Date filters apply to target time; earlier hours remain usable as history.
    """
    if min(history_hours, horizon_hours, stride) < 1:
        raise ValueError("history_hours, horizon_hours and stride must be positive")
    if target_start and target_end and target_end < target_start:
        raise ValueError("target_end must be on or after target_start")

    buffer: deque[HourlyCount] = deque(maxlen=history_hours + horizon_hours)
    previous_route = None
    candidate = 0
    for row in iter_hourly_counts(
        paths, series_start=series_start, series_end=series_end, routes=routes
    ):
        if row.route != previous_route:
            buffer.clear()
            candidate = 0
            previous_route = row.route
        buffer.append(row)
        if len(buffer) < buffer.maxlen:
            continue

        target = buffer[-1]
        target_date = target.timestamp.date()
        use = candidate % stride == 0
        candidate += 1
        if not use or (target_start and target_date < target_start) or (
            target_end and target_date > target_end
        ):
            continue

        history = list(buffer)[:history_hours]
        yield BoardingWindow(
            route=row.route,
            history_start=history[0].timestamp,
            history_end=history[-1].timestamp,
            target_time=target.timestamp,
            history_boardings=tuple(item.boardings for item in history),
            history_observed=tuple(item.observed for item in history),
            target_boardings=target.boardings,
            target_observed=target.observed,
        )


def sample_boarding_windows(
    paths: Sequence[str | Path],
    *,
    sample_size: int,
    seed: int = 42,
    **window_options,
) -> list[BoardingWindow]:
    """Uniformly sample windows in one pass using bounded memory."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    sample: list[BoardingWindow] = []
    for seen, window in enumerate(iter_boarding_windows(paths, **window_options), start=1):
        if seen <= sample_size:
            sample.append(window)
        else:
            replacement = rng.randrange(seen)
            if replacement < sample_size:
                sample[replacement] = window
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Sorted hourly label CSV files")
    parser.add_argument("--series-start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--series-end", type=date.fromisoformat, default=date(2025, 8, 31))
    parser.add_argument("--history-hours", type=int, default=64)
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-start", type=date.fromisoformat)
    parser.add_argument("--target-end", type=date.fromisoformat)
    parser.add_argument("--route", action="append", type=int)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample = sample_boarding_windows(
        args.paths,
        sample_size=args.sample_size,
        seed=args.seed,
        series_start=args.series_start,
        series_end=args.series_end,
        history_hours=args.history_hours,
        horizon_hours=args.horizon_hours,
        stride=args.stride,
        target_start=args.target_start,
        target_end=args.target_end,
        routes=set(args.route) if args.route else None,
    )
    for window in sample:
        print(json.dumps(asdict(window), default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
