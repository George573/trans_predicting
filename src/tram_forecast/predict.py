"""Submission export."""

import csv
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

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


def write_submission(template, output, predictions, start, end):
    expected = {
        (r, start + timedelta(days=d), h)
        for r in (*ROUTES, 5)
        for d in range((end - start).days)
        for h in range(24)
    }
    if set(predictions) != expected:
        raise ValueError("prediction keys must cover the complete submission grid")
    ordered = []
    with open(template, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            ordered.append(
                (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".submission-", dir=output.parent)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        for r, d, h in ordered:
            writer.writerow([r, d.isoformat(), h, format(float(predictions[r, d, h]), ".10g")])
    os.replace(name, output)
    return len(ordered)


def predict(model, boardings, template, output, cutoff, days):
    values = fixed_forecast(model, boardings, cutoff, days)
    predictions = {
        (route, cutoff + timedelta(days=d), h): float(values[i, d, h])
        for i, route in enumerate(boardings.routes)
        for d in range(days)
        for h in range(24)
    }
    for d in range(days):
        for h in range(24):
            predictions[5, cutoff + timedelta(days=d), h] = 0.0
    return write_submission(template, output, predictions, cutoff, cutoff + timedelta(days=days))