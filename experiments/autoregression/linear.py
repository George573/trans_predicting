"""Regularized seasonal ARX baseline with calendar and scheduled-event inputs."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from probe import detailed
from run import START, Windows, load_data

from tram_forecast.data import request_inputs

LAGS = (1, 7, 14, 21, 28)


def fit_arx(data, alpha):
    cache = Windows(data, 28)
    models = []
    for r in range(len(data.routes)):
        y = data.counts[r].reshape(-1, 24)
        scale = max(float(y.mean()), 1.0)
        y = y / scale
        indices = np.arange(28, len(y))
        x = np.concatenate(
            [y[indices - lag] for lag in LAGS]
            + [cache.requests[r, indices], np.ones((len(indices), 1))],
            axis=1,
        )
        target = y[indices] - y[indices - 7]
        penalty = np.eye(x.shape[1]) * alpha
        penalty[-1, -1] = alpha * 0.01
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ target)
        models.append((scale, coef))
    return models


def forecast_arx(models, data, cutoff, days=61):
    forecasts = []
    for route, (scale, coef) in zip(data.routes, models):
        history = list(data.history(route, cutoff, 28).reshape(28, 24) / scale)
        requested = request_inputs(
            data.scheduled_events,
            route,
            [cutoff + timedelta(days=d) for d in range(days)],
        )
        for d in range(days):
            x = np.concatenate(
                [history[-lag] for lag in LAGS] + [requested[d], np.ones(1)]
            )
            prediction = np.maximum(0, history[-7] + x @ coef)
            history.append(prediction)
        forecasts.append(np.stack(history[28:]) * scale)
    return np.stack(forecasts)


def main():
    out = Path("outputs/autoregression_linear")
    out.mkdir(exist_ok=False)
    results = {}
    for cutoff in (date(2025, 7, 2), date(2025, 9, 1)):
        train = load_data(cutoff)
        validation = load_data(cutoff + timedelta(days=61))
        actual = validation.counts[:, (cutoff - START).days * 24 :].reshape(9, 61, 24)
        predictions = {}
        for alpha in (1.0, 10.0, 100.0):
            name = f"arx_{alpha}"
            models = fit_arx(train, alpha)
            pred = forecast_arx(models, train, cutoff)
            predictions[name] = pred
            results[f"{cutoff}_{name}"] = detailed(actual, pred)
            print(cutoff, name, results[f"{cutoff}_{name}"]["score"], flush=True)
        np.savez_compressed(out / f"{cutoff}.npz", actual=actual, **predictions)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
