"""One stored history per context, with all eligible requested-day targets."""

from datetime import timedelta

import numpy as np

from .schema import SampleIdentity


def sample_index(routes, start, end, forecast_days=7, history_days=21):
    validate_horizon(forecast_days)
    validate_history(history_days)
    if end <= start:
        raise ValueError("end must follow start")
    return tuple(
        SampleIdentity(r, start + timedelta(days=c), start + timedelta(days=c + h - 1))
        for r in routes
        for c in range(history_days, (end - start).days)
        for h in range(1, min(forecast_days, (end - start).days - c) + 1)
    )


def validate_history(days):
    if type(days) is not int or days < 1:
        raise ValueError("history_days must be a positive integer")


def validate_horizon(days):
    if type(days) is not int or not 1 <= days <= 61:
        raise ValueError("forecast horizon must be an integer in 1..61 days")


def epoch_order(size, epoch, seed=67):
    if size < 0 or epoch < 0:
        raise ValueError("size and epoch must be nonnegative")
    return np.random.default_rng(seed + epoch).permutation(size)


class ForecastDataset:
    """Context-indexed histories; partial horizons never cross the fitting boundary."""

    def __init__(
        self, artifact, model_kind="boarding_only", forecast_days=7, history_days=21,
        context_start_days=None,
    ):
        from .storage import Store

        if model_kind != "boarding_only":
            raise ValueError("unknown model kind")
        self.model_kind = model_kind
        validate_horizon(forecast_days)
        validate_history(history_days)
        first_context = history_days if context_start_days is None else context_start_days
        validate_history(first_context)
        if first_context < history_days:
            raise ValueError("context_start_days cannot be shorter than history_days")
        self.history_days = history_days
        self.forecast_days = forecast_days
        self.store = Store(artifact)
        # One identity per route/cutoff, regardless of the number of requested days.
        self.index = np.asarray(
            [
                (r, c)
                for r in self.store.routes
                for c in range(first_context, (self.store.end - self.store.start).days)
            ],
            dtype=np.int32,
        ).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from .schema import request_calendar

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
    from datetime import datetime, time

    from .config import ROUTES
    from .schema import calendar, request_calendar

    if not 1 <= identity.lead <= 61:
        raise ValueError("requested lead outside 1..61")
    counts = store.history(identity.route, identity.cutoff, history_days)
    first = datetime.combine(identity.cutoff, time.min) - timedelta(days=history_days)
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
