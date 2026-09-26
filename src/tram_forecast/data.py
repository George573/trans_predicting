"""Hourly boarding data: parse labels, prepare artifacts and batch histories."""

import csv
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as day_time
from pathlib import Path

import numpy as np

from .config import HISTORY_DAYS, ROUTES
from .io import digest, source_fingerprints, write_json


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
    return calendar([datetime.combine(d, day_time.min) for d in days])[2:].T


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


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        m = self.metadata
        if m["schema_version"] != 2 or digest(m["identity"]) != m["fingerprint"]:
            raise ValueError("incompatible or corrupt artifact metadata; prepare boardings in a new directory")
        self.start = date.fromisoformat(m["start"])
        self.end = date.fromisoformat(m["end"])
        self.routes = tuple(m["routes"])
        self.route_index = {r: i for i, r in enumerate(self.routes)}
        scaling = json.loads((self.path / "scaling.json").read_text())
        if (
            scaling["scale"] != m["scale"]
            or scaling["start"] != m["start"]
            or scaling["end"] != m["end"]
        ):
            raise ValueError("scaling metadata mismatch")
        self.counts = np.load(
            self.path / "boardings.npy", mmap_mode="r", allow_pickle=False
        )
        shape = (len(self.routes), (self.end - self.start).days * 24)
        if self.counts.shape != shape or self.counts.dtype != np.float32:
            raise ValueError("artifact boarding grid shape/dtype mismatch")
        if not np.isfinite(self.counts).all() or (self.counts < 0).any():
            raise ValueError("invalid artifact boarding values")
        self.contract = {
            "fingerprint": m["fingerprint"],
            "scale": m["scale"],
            "routes": self.routes,
            "start": m["start"],
            "end": m["end"],
        }
        self.contract_hash = digest(self.contract)

    def history(self, route, cutoff, history_days=21):
        if type(history_days) is not int or history_days < 1:
            raise ValueError("history_days must be a positive integer")
        if route not in self.route_index:
            raise ValueError("unsupported route")
        stop = (cutoff - self.start).days * 24
        begin = stop - history_days * 24
        if begin < 0 or stop > self.counts.shape[1]:
            raise ValueError("history outside fitting artifact")
        r = self.route_index[route]
        counts = np.array(self.counts[r, begin:stop], copy=True)
        return counts

    def target(self, route, day):
        offset = (day - self.start).days * 24
        if not 0 <= offset <= self.counts.shape[1] - 24:
            raise ValueError("target outside fitting interval")
        return np.array(
            self.counts[self.route_index[route], offset : offset + 24], copy=True
        )

    def evaluation_targets(self):
        if self.metadata["regime"] != "validation":
            raise ValueError("final artifact has no observed future targets")
        values = np.load(
            self.path / "evaluation.npy", mmap_mode="r", allow_pickle=False
        )
        expected = (
            len(self.routes) + 1,
            (date.fromisoformat(self.metadata["evaluation_end"]) - self.end).days * 24,
        )
        if values.shape != expected:
            raise ValueError("evaluation target shape mismatch")
        return values


SCHEMA_VERSION = 2


def preparation_identity(settings, regime):
    if regime not in ("validation", "final"):
        raise ValueError("unknown preparation regime")
    data = settings.data
    end = data.validation_cutoff if regime == "validation" else data.final_cutoff
    return {
        "schema_version": SCHEMA_VERSION,
        "regime": regime,
        "start": data.start,
        "end": end,
        "data": settings.to_dict()["data"],
        "sources": source_fingerprints(data.label_paths),
    }


def prepare(settings, regime, output=None):
    """Publish immutable artifacts atomically; refuse to overwrite incompatible data."""
    identity = preparation_identity(settings, regime)
    fingerprint = digest(identity)
    output = Path(output or Path(settings.data.output_root) / regime)
    if output.exists():

        existing = Store(output)
        if existing.metadata["fingerprint"] != fingerprint:
            raise ValueError(
                f"{output}: incompatible existing artifact; use a new output directory"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < settings.data.min_free_disk_bytes:
        raise OSError(f"insufficient free space in {output.parent}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.monotonic()
    try:
        _build(settings, identity, fingerprint, stage)
        if (
            source_fingerprints(settings.data.label_paths)
            != identity["sources"]
        ):
            raise ValueError("source files changed during preparation")
        stage.rename(output)
        print(f"Prepared {output} in {time.monotonic() - started:.1f}s", flush=True)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return output


def _build(settings, identity, fingerprint, stage):
    data = settings.data
    routes = tuple(data.routes)
    start = date.fromisoformat(identity["start"])
    end = date.fromisoformat(identity["end"])
    # Route 5 is deliberately checked even though it has no neural index.
    counts, presence = label_grid(data.label_paths, (*routes, 5), start, end)
    if presence[-1].any():
        raise ValueError(
            "route 5 has labels: conflicts with the no-history fallback policy"
        )
    counts = counts[:-1]
    presence = presence[:-1]
    if not presence.any(axis=1).all():
        raise ValueError("a configured neural route has no fitting-period labels")
    np.save(stage / "boardings.npy", counts)
    np.save(stage / "label_present.npy", presence)
    scale = count_scale(counts)
    write_json(
        stage / "scaling.json", {"scale": scale, "start": str(start), "end": str(end)}
    )
    eval_end = date.fromisoformat(
        data.final_cutoff if identity["regime"] == "validation" else data.forecast_end
    )
    if identity["regime"] == "validation":
        targets, target_present = label_grid(
            data.label_paths, (*routes, 5), end, eval_end
        )
        np.save(stage / "evaluation.npy", targets)
        np.save(stage / "evaluation_present.npy", target_present)
    write_json(stage / "metadata.json", {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "regime": identity["regime"],
        "start": str(start),
        "end": str(end),
        "evaluation_end": str(eval_end),
        "routes": routes,
        "label_rows_by_route": presence.sum(axis=1).tolist(),
        "scale": scale,
    })


def validate_horizon(days):
    if type(days) is not int or not 1 <= days <= 61:
        raise ValueError("forecast horizon must be an integer in 1..61 days")


def epoch_order(size, epoch, seed=67):
    if size < 0 or epoch < 0:
        raise ValueError("size and epoch must be nonnegative")
    return np.random.default_rng(seed + epoch).permutation(size)


class ForecastDataset:
    """Context-indexed histories; partial horizons never cross the fitting boundary."""

    def __init__(self, artifact, forecast_days=7):
        validate_horizon(forecast_days)
        self.history_days = HISTORY_DAYS
        self.forecast_days = forecast_days
        self.store = Store(artifact)
        # One identity per route/cutoff, regardless of the number of requested days.
        self.index = np.asarray(
            [
                (r, c)
                for r in self.store.routes
                for c in range(HISTORY_DAYS, (self.store.end - self.store.start).days)
            ],
            dtype=np.int32,
        ).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        route, c = map(int, self.index[index])
        cutoff = self.store.start + timedelta(days=c)
        identity = SampleIdentity(route, cutoff, cutoff)
        result = history_sample(self.store, identity, self.history_days)
        days = min(self.forecast_days, (self.store.end - cutoff).days)
        requested = [cutoff + timedelta(days=h) for h in range(days)]
        result["lead"] = np.arange(1, days + 1, dtype=np.float32)
        result["request_calendar"] = request_calendar(requested)
        result["target"] = np.stack([self.store.target(route, day) for day in requested])
        return result


def history_sample(store, identity, history_days=21):
    if not 1 <= identity.lead <= 61:
        raise ValueError("requested lead outside 1..61")
    counts = store.history(identity.route, identity.cutoff, history_days)
    first = datetime.combine(identity.cutoff, day_time.min) - timedelta(days=history_days)
    return {
        "identity": identity,
        "counts": counts[None, :],
        "calendar": calendar(first + timedelta(hours=i) for i in range(history_days * 24)),
        "route_index": ROUTES.index(identity.route) + 1,
        "request_calendar": request_calendar([identity.requested])[0],
        "lead": identity.lead,
    }


def collate_samples(samples, device="cpu"):
    """Batch B histories and flatten their R requests/targets with context indices."""
    import torch

    if not samples:
        raise ValueError("empty batch")
    history = {
        "counts": torch.as_tensor(
            np.stack([s["counts"] for s in samples]), device=device
        ),
        "calendar": torch.as_tensor(
            np.stack([s["calendar"] for s in samples]), device=device
        ),
    }
    request = {
        "context_indices": torch.tensor(
            [i for i, s in enumerate(samples) for _ in np.atleast_1d(s["lead"])],
            device=device,
        ),
        "route_indices": torch.tensor(
            [s["route_index"] for s in samples for _ in np.atleast_1d(s["lead"])],
            device=device,
        ),
        "calendar": torch.as_tensor(
            np.concatenate([np.atleast_2d(s["request_calendar"]) for s in samples]),
            device=device,
        ),
        "lead": torch.tensor(
            np.concatenate([np.atleast_1d(s["lead"]) for s in samples]),
            device=device,
            dtype=torch.float32,
        ),
    }
    targets = (
        torch.as_tensor(
            np.concatenate([np.atleast_2d(s["target"]) for s in samples]), device=device
        )
        if all("target" in s for s in samples)
        else None
    )
    return history, request, targets
