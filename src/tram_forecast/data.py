"""Hourly boarding history: load a labels CSV, sample windows for the model."""

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as day_time

import numpy as np
import torch
from torch.utils.data import Dataset

from .events import EVENT_FEATURES, ScheduledEvents, load_events
from .russian_calendar import day_flags

HISTORY_CALENDAR_FEATURES = (
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_day_sin",
    "month_day_cos", "is_holiday", "is_day_off", "is_short_working_day",
)
REQUEST_CALENDAR_FEATURES = HISTORY_CALENDAR_FEATURES[2:]
HISTORY_INPUT_FEATURES = HISTORY_CALENDAR_FEATURES + EVENT_FEATURES
REQUEST_INPUT_FEATURES = REQUEST_CALENDAR_FEATURES + EVENT_FEATURES

HISTORY_DAYS = 21
ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)


def parse_route(value):
    return int(re.fullmatch(r"(\d+)(?:\s+трамвай)?", str(value).strip()).group(1))


def calendar(timestamps, *, missing_zero=False):
    """Cyclic date features followed by holiday, day-off and short-day flags."""
    t = list(timestamps)
    phases = np.array(
        [[x.hour / 24, x.weekday() / 7, (x.day - 1) / 31] for x in t], dtype=np.float64
    ) * 2 * np.pi
    cyclic = np.stack([f(phases[:, i]) for i in range(3) for f in (np.sin, np.cos)])
    flags = [day_flags(x.date(), missing_zero=missing_zero) for x in t]
    binary = np.array([
        [f.is_holiday, f.is_day_off, f.is_short_working_day] for f in flags
    ], dtype=np.float32).T
    return np.concatenate((cyclic, binary), axis=0).astype(np.float32)


def request_calendar(days, *, missing_zero=False):
    return calendar([datetime.combine(d, day_time.min) for d in days],
                    missing_zero=missing_zero)[2:].T


@dataclass
class Boardings:
    """Boarding labels and a shared event schedule, with independent coverage."""
    counts: np.ndarray   # [routes, hours]
    routes: tuple
    start: date
    scale: float
    scheduled_events: ScheduledEvents | None = None

    @classmethod
    def load(cls, path, routes=ROUTES, start=date(2025, 1, 1), end=date(2026, 1, 1),
             events_path=None):
        width = (end - start).days * 24
        counts = np.zeros((len(routes), width), dtype=np.float32)
        index = {r: i for i, r in enumerate(routes)}
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                r = parse_route(row["route"])
                d = date.fromisoformat(row["date"])
                if r in index and start <= d < end:
                    j = (d - start).days * 24 + int(row["hour"])
                    counts[index[r], j] = int(row["boardings"])
        events = load_events(events_path) if events_path is not None else None
        return cls(counts, tuple(routes), start, max(1.0, float(counts.mean())), events)

    def history(self, route, cutoff, history_days):
        stop = (cutoff - self.start).days * 24
        start = stop - history_days * 24
        if start < 0 or stop > self.counts.shape[1]:
            raise ValueError(f"cutoff {cutoff} out of range for {history_days}-day history "
                            f"(need data from {self.start} to {self.start + timedelta(days=self.counts.shape[1]//24)})")
        return self.counts[self.routes.index(route), start:stop].copy()

    def target(self, route, day):
        offset = (day - self.start).days * 24
        return self.counts[self.routes.index(route), offset : offset + 24].copy()


def history_inputs(boardings, route, cutoff, history_days, *, missing_zero=False):
    """Prepare one history without requiring any future schedule or labels."""
    if boardings.scheduled_events is None and not missing_zero:
        raise ValueError("Load Boardings with events_path before sampling model inputs")
    first = datetime.combine(cutoff, day_time.min) - timedelta(days=history_days)
    history_events = (
        boardings.scheduled_events.window(route, first.date(), history_days * 24,
                                          missing_zero=missing_zero)
        if boardings.scheduled_events is not None else np.zeros((len(EVENT_FEATURES), history_days * 24), dtype=np.float32)
    )
    return {
        "counts": boardings.history(route, cutoff, history_days)[None, :],
        "calendar": np.concatenate((
            calendar((first + timedelta(hours=i) for i in range(history_days * 24)),
                     missing_zero=missing_zero),
            history_events,
        )),
    }


def request_inputs(events, route, days, *, missing_zero=False):
    """Prepare each requested day's calendar and daily event reductions."""
    if events is None and not missing_zero:
        raise ValueError("Load Boardings with events_path before sampling model inputs")
    days = list(days)
    daily_events = np.stack([
        events.window(route, day, 24, missing_zero=missing_zero).max(axis=1)
        if events is not None else np.zeros(len(EVENT_FEATURES), dtype=np.float32)
        for day in days
    ])
    return np.concatenate((request_calendar(days, missing_zero=missing_zero), daily_events), axis=1)


def sample(boardings, route, cutoff, forecast_days, history_days, *, include_target=True):
    """Assemble calendar and hourly event inputs; targets are optional for inference."""
    requested = [cutoff + timedelta(days=d) for d in range(forecast_days)]
    result = {
        **history_inputs(boardings, route, cutoff, history_days),
        "route_index": ROUTES.index(route) + 1,
        "request_calendar": request_inputs(boardings.scheduled_events, route, requested),
        "lead": np.arange(1, forecast_days + 1, dtype=np.float32),
    }
    if include_target:
        result["target"] = np.stack([boardings.target(route, d) for d in requested])
    return result


class ForecastDataset(Dataset):
    def __init__(self, boardings, forecast_days, history_days):
        self.boardings = boardings
        self.forecast_days = forecast_days
        self.history_days = history_days
        days_available = boardings.counts.shape[1] // 24
        self.index = [
            (r, boardings.start + timedelta(days=c))
            for r in boardings.routes
            for c in range(history_days, days_available - forecast_days + 1)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        route, cutoff = self.index[i]
        return sample(self.boardings, route, cutoff, self.forecast_days, self.history_days)

def collate_samples(samples):
    """DataLoader collate_fn. Returns CPU tensors; move to device in the loop."""
    history = {
        "counts": torch.as_tensor(np.stack([s["counts"] for s in samples])),
        "calendar": torch.as_tensor(np.stack([s["calendar"] for s in samples])),
    }
    request = {
        "context_indices": torch.tensor(
            [i for i, s in enumerate(samples) for _ in s["lead"]]
        ),
        "route_indices": torch.tensor(
            [s["route_index"] for s in samples for _ in s["lead"]]
        ),
        "calendar": torch.as_tensor(
            np.concatenate([s["request_calendar"] for s in samples])
        ),
        "lead": torch.as_tensor(
            np.concatenate([s["lead"] for s in samples]), dtype=torch.float32
        ),
    }
    targets = torch.as_tensor(np.concatenate([s["target"] for s in samples]))
    return history, request, targets
