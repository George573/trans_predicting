"""Fixed-history inference and metrics — no checks."""

import numpy as np
import torch

from .data import ROUTES, sample


@torch.no_grad()
def fixed_forecast(model, boardings, cutoff, days):
    model.eval()
    device = next(model.parameters()).device
    results = []
    for route in boardings.routes:
        s = sample(boardings, route, cutoff, days)
        history = {
            "counts": torch.as_tensor(s["counts"][None], device=device, dtype=torch.float32),
            "calendar": torch.as_tensor(s["calendar"][None], device=device, dtype=torch.float32),
        }
        encoded = model.encode_history(history)
        route_indices = torch.full((days,), ROUTES.index(route) + 1, device=device, dtype=torch.long)
        calendar_tensor = torch.as_tensor(s["request_calendar"], device=device, dtype=torch.float32)
        lead = torch.as_tensor(s["lead"], device=device)
        prediction = model.predict_day(encoded.expand(days, -1), route_indices, calendar_tensor, lead)
        results.append(prediction.cpu().numpy())
    return np.stack(results)


def evaluate_model(model, boardings, cutoff, days):
    actual = np.stack([sample(boardings, route, cutoff, days)["target"] for route in boardings.routes])
    predicted = fixed_forecast(model, boardings, cutoff, days)
    return float(np.abs(actual - predicted).sum() / actual.sum())