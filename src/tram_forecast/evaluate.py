"""Fixed-history inference and correctly aggregated metrics."""

from datetime import date, timedelta

import numpy as np
import torch

from .config import HISTORY_DAYS, ROUTES
from .data import (
    SampleIdentity,
    collate_samples,
    history_sample,
    request_calendar,
    validate_horizon,
)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (
        actual.shape != predicted.shape
        or actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or (actual < 0).any()
        or (predicted < 0).any()
    ):
        raise ValueError("metrics require matching finite nonnegative nonempty arrays")
    absolute = float(np.abs(actual - predicted).sum())
    denominator = float(actual.sum())
    wape = absolute / denominator if denominator else None
    return {
        "wape": wape,
        "wape_score": max(0.0, 1 - wape) if wape is not None else None,
        "absolute_error": absolute,
        "actual_total": denominator,
        "mae": absolute / actual.size,
        "positions": actual.size,
        "undefined_reason": "zero actual denominator" if not denominator else None,
    }


def metric_report(actual, predicted, routes):
    if actual.ndim != 3 or actual.shape[0] != len(routes) or actual.shape[-1] != 24:
        raise ValueError("expected [routes,days,24]")
    report = {
        "global": metrics(actual, predicted),
        "routes": {},
        "leads": {},
        "lead_groups": {},
    }
    for i, route in enumerate(routes):
        report["routes"][str(route)] = metrics(actual[i], predicted[i])
    for i in range(actual.shape[1]):
        report["leads"][str(i + 1)] = metrics(actual[:, i], predicted[:, i])
    for low, high in ((1, 7), (8, 14), (15, 30), (31, 61)):
        if low <= actual.shape[1]:
            high = min(high, actual.shape[1])
            report["lead_groups"][f"{low}-{high}"] = metrics(
                actual[:, low - 1 : high], predicted[:, low - 1 : high]
            )
    return report


def weekly_profile(store, days, history_days=21):
    validate_horizon(days)
    if type(history_days) is not int or history_days < 7:
        raise ValueError("weekly profile requires at least seven history days")
    output = []
    dates = [store.end + timedelta(days=i) for i in range(days)]
    historic = [store.end - timedelta(days=history_days) + timedelta(days=i)
                for i in range(history_days)]
    for route in store.routes:
        counts = store.history(route, store.end, history_days)
        counts = counts.reshape(history_days, 24)
        output.append(
            np.stack(
                [
                    counts[[h.weekday() == d.weekday() for h in historic]].mean(axis=0)
                    for d in dates
                ]
            )
        )
    return np.asarray(output)


@torch.no_grad()
def fixed_forecast(model, store, days=7):
    validate_horizon(days)
    model.eval()
    device = next(model.parameters()).device
    results = []
    dates = [store.end + timedelta(days=i) for i in range(days)]
    for route in store.routes:
        sample = history_sample(
            store,
            SampleIdentity(route, store.end, store.end),
            HISTORY_DAYS,
        )
        history, _, _ = collate_samples([sample], device)
        encoded = model.encode_history(history)
        route_indices = torch.full(
            (days,), ROUTES.index(route) + 1, device=device, dtype=torch.long
        )
        request_calendar_tensor = torch.as_tensor(request_calendar(dates), device=device)
        lead = torch.arange(1, days + 1, device=device, dtype=torch.float32)
        prediction = model.predict_day(
            encoded.expand(days, -1), route_indices, request_calendar_tensor, lead,
        )
        results.append(prediction.cpu().numpy())
    return np.stack(results)


def evaluate_model(model, store, days=7):
    validate_horizon(days)
    if store.metadata["regime"] != "validation":
        raise ValueError("evaluation requires a validation artifact")
    available = (date.fromisoformat(store.metadata["evaluation_end"]) - store.end).days
    if days > available:
        raise ValueError("forecast horizon exceeds available validation days")
    actual = np.asarray(store.evaluation_targets()).reshape(
        len(store.routes) + 1, available, 24
    )[:, :days]
    prediction = fixed_forecast(model, store, days)
    baseline = weekly_profile(store, days)
    zeros = np.zeros((1, days, 24), dtype=np.float32)
    return {
        "neural": metric_report(actual[:-1], prediction, store.routes),
        "full_grid_with_route5": metric_report(
            actual, np.concatenate((prediction, zeros)), (*store.routes, 5)
        ),
        "weekly_profile": metric_report(actual[:-1], baseline, store.routes),
        "weekly_profile_full_grid": metric_report(
            actual, np.concatenate((baseline, zeros)), (*store.routes, 5)
        ),
        "cutoff": str(store.end),
        "forecast_days": days,
        "artifact": store.contract_hash,
        "model_kind": model.model_kind,
    }
