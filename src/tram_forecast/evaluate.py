"""Validation metrics using the same assembled inputs as training."""

import numpy as np

from .data import sample
from .predict import fixed_forecast


def evaluate_model(model, boardings, cutoff, days, history_days=14):
    actual = np.stack([
        sample(boardings, route, cutoff, days, history_days)["target"]
        for route in boardings.routes
    ])
    predicted = fixed_forecast(model, boardings, cutoff, days, history_days)
    return float(np.abs(actual - predicted).sum() / actual.sum())
