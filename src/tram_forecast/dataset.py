"""Compact sample identities; storage-backed loading will be added separately."""

from datetime import timedelta

import numpy as np

from .schema import SampleIdentity


def sample_index(routes, start, end):
    if end <= start:
        raise ValueError("end must follow start")
    return tuple(
        SampleIdentity(r, start + timedelta(days=c), start + timedelta(days=c + h - 1))
        for r in routes
        for c in range(21, (end - start).days)
        for h in range(1, min(61, (end - start).days - c) + 1)
    )


def epoch_order(size, epoch, seed=67):
    if size < 0 or epoch < 0:
        raise ValueError("size and epoch must be nonnegative")
    return np.random.default_rng(seed + epoch).permutation(size)


class ForecastDataset:
    """Identity-indexed memory-mapped histories; held-out targets are inaccessible."""

    def __init__(self, artifact, model_kind="full"):
        from .storage import Store

        if model_kind not in ("full", "boarding_only"):
            raise ValueError("unknown model kind")
        self.model_kind = model_kind
        self.store = Store(artifact, events=model_kind == "full")
        # Three integer columns avoid hundreds of thousands of Python date objects.
        self.index = np.asarray(
            [
                (r, c, h)
                for r in self.store.routes
                for c in range(21, (self.store.end - self.store.start).days)
                for h in range(
                    1, min(61, (self.store.end - self.store.start).days - c) + 1
                )
            ],
            dtype=np.int32,
        ).reshape(-1, 3)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        route, c, h = map(int, self.index[index])
        cutoff = self.store.start + timedelta(days=c)
        identity = SampleIdentity(route, cutoff, cutoff + timedelta(days=h - 1))
        result = history_sample(self.store, identity, self.model_kind == "full")
        result["target"] = self.store.target(route, identity.requested)
        return result


def history_sample(store, identity, include_events=True):
    from datetime import datetime, time

    from .config import ROUTES
    from .schema import calendar, request_calendar

    if not 1 <= identity.lead <= 61:
        raise ValueError("requested lead outside 1..61")
    counts, hours = store.history(identity.route, identity.cutoff, include_events)
    first = datetime.combine(identity.cutoff, time.min) - timedelta(hours=504)
    return {
        "identity": identity,
        "counts": counts[None, :],
        "calendar": calendar(first + timedelta(hours=i) for i in range(504)),
        "hours": hours,
        "route_index": ROUTES.index(identity.route) + 1,
        "request_calendar": request_calendar([identity.requested])[0],
        "lead": identity.lead,
    }


def collate_samples(samples, device="cpu"):
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
    event_flags = [s["hours"] is not None for s in samples]
    if any(event_flags) and not all(event_flags):
        raise ValueError("mixed event/boarding-only batch")
    if all(event_flags):
        history["hours"] = [
            torch.from_numpy(hour) for s in samples for hour in s["hours"]
        ]
    request = {
        "route_indices": torch.tensor(
            [s["route_index"] for s in samples], device=device
        ),
        "calendar": torch.as_tensor(
            np.stack([s["request_calendar"] for s in samples]), device=device
        ),
        "lead": torch.tensor(
            [s["lead"] for s in samples], device=device, dtype=torch.float32
        ),
    }
    targets = (
        torch.as_tensor(np.stack([s["target"] for s in samples]), device=device)
        if all("target" in s for s in samples)
        else None
    )
    return history, request, targets
