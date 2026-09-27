"""Recursive forecasts that never read labels on or after the forecast cutoff."""

from datetime import datetime, time, timedelta

import numpy as np
import torch

from .data import ROUTES, calendar, request_inputs


@torch.inference_mode()
def recursive_forecast(
    model, boardings, cutoff, days=61, block_days=1, anchor=None, recursive_weight=1.0
):
    """Predict a block, append only predictions, then move the history window.

    Optional fixed-cutoff predictions anchor each generated block to limit drift.
    Blended predictions (not targets) are fed back for anchored recursion.
    The caller's observations are never modified.
    """
    if days < 1 or block_days < 1 or not 0 <= recursive_weight <= 1:
        raise ValueError("invalid horizon, block size or recursive weight")
    if anchor is not None and np.shape(anchor) != (len(boardings.routes), days, 24):
        raise ValueError("anchor must have shape [routes, days, 24]")
    model.eval()
    device = next(model.parameters()).device
    history_days = getattr(model, "history_days", 14)
    first = cutoff - timedelta(days=history_days)
    hours = (history_days + days) * 24
    timestamps = [
        datetime.combine(first, time.min) + timedelta(hours=i) for i in range(hours)
    ]
    cal = calendar(timestamps)
    features = np.stack(
        [
            np.concatenate((cal, boardings.scheduled_events.window(r, first, hours)))
            for r in boardings.routes
        ]
    )
    requested_days = [cutoff + timedelta(days=d) for d in range(days)]
    requested = np.stack(
        [
            request_inputs(boardings.scheduled_events, r, requested_days)
            for r in boardings.routes
        ]
    )
    route_indices = torch.tensor(
        [ROUTES.index(r) + 1 for r in boardings.routes], device=device
    )
    # Allocate from past observations only, even when boardings also has held-out labels.
    state = torch.zeros((len(boardings.routes), hours), device=device)
    state[:, : history_days * 24] = torch.as_tensor(
        np.stack(
            [boardings.history(r, cutoff, history_days) for r in boardings.routes]
        ),
        device=device,
    )
    features = torch.as_tensor(features, device=device)
    requested = torch.as_tensor(requested, device=device)
    for offset in range(0, days, block_days):
        length = min(block_days, days - offset)
        stop = (history_days + offset) * 24
        history = {
            "counts": state[:, None, offset * 24 : stop],
            "calendar": features[:, :, offset * 24 : stop],
        }
        encoded = model.encode_history(history).repeat_interleave(length, dim=0)
        predicted = model.predict_day(
            encoded,
            route_indices.repeat_interleave(length),
            requested[:, offset : offset + length].reshape(-1, requested.shape[-1]),
            torch.arange(1, length + 1, device=device, dtype=torch.float32).repeat(
                len(route_indices)
            ),
        ).reshape(len(route_indices), length, 24)
        if anchor is not None:
            predicted = recursive_weight * predicted + (
                1 - recursive_weight
            ) * torch.as_tensor(
                anchor[:, offset : offset + length],
                device=device,
            )
        if not torch.isfinite(predicted).all() or (predicted < 0).any():
            raise ValueError("recursive forecast produced invalid counts")
        state[:, stop : stop + length * 24] = predicted.flatten(1)
    return (
        state[:, history_days * 24 :]
        .reshape(len(route_indices), days, 24)
        .cpu()
        .numpy()
    )
