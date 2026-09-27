import csv
from datetime import date, timedelta

import numpy as np
import pytest
import torch

from tram_forecast.data import Boardings, ForecastDataset, collate_samples, sample
from tram_forecast.events import EVENT_FEATURES, load_events
from tram_forecast.evaluate import evaluate_model
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import fixed_forecast
from tram_forecast.train import load_model, train
from tram_forecast.runner import InferenceRunner
from unittest.mock import patch


def write_events(path, days=90):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", *EVENT_FEATURES,
                         "has_route_change", "parking_pct_route", "has_parking"])
        for route in (1, 7):
            for day in range(days):
                for hour in range(24):
                    writer.writerow([route, date(2025, 9, 1) + timedelta(days=day), hour,
                                     int(hour < 4), int(route == 7 and hour == 18),
                                     int(day % 2 == 0 and hour == 23),
                                     "ignored", "ignored", "ignored"])


@pytest.fixture
def store(tmp_path):
    events = tmp_path / "events.csv"
    write_events(events)
    labels = tmp_path / "labels.csv"
    labels.write_text("route;date;hour;boardings\n1;2025-09-01;0;5\n")
    return Boardings.load(labels, routes=(7, 1), start=date(2025, 9, 1),
                          end=date(2025, 9, 18), events_path=events), events


def test_sampling_aligns_route_date_and_hour(store):
    boardings, _ = store
    for route in (1, 7):
        s = sample(boardings, route, date(2025, 9, 15), 2, 14)
        assert s["calendar"].shape == (12, 336)
        assert s["request_calendar"].shape == (2, 10)
        assert s["calendar"].dtype == s["request_calendar"].dtype == np.float32
        history = s["calendar"][-3:].T.reshape(14, 24, 3)
        future = s["request_calendar"][:, 7:]
        np.testing.assert_array_equal(history[0, :5, 0], [1, 1, 1, 1, 0])
        assert history[0, 18, 1] == future[0, 1] == int(route == 7)
        assert future[0, 2] == 1 and future[1, 2] == 0
        np.testing.assert_array_equal(future[:, 0], [1, 1])
        assert history[-1, 23, 2] == 0


def test_schedule_is_cached_and_not_truncated_by_label_split(store, tmp_path):
    boardings, events = store
    assert boardings.scheduled_events is load_events(events)
    assert boardings.scheduled_events.values.shape == (2, 3, 90 * 24)
    sample_without_labels = sample(boardings, 1, date(2025, 9, 15), 61, 14,
                                   include_target=False)
    assert "target" not in sample_without_labels
    assert sample_without_labels["request_calendar"].shape == (61, 10)


def test_forecast_never_reads_target_labels(store, monkeypatch):
    boardings, _ = store
    def forbidden(*args):
        raise AssertionError("Forecast read target labels")
    monkeypatch.setattr(boardings, "target", forbidden)
    result = fixed_forecast(ForecastNetwork(1), boardings, date(2025, 9, 15), 61, 14)
    assert result.shape == (2, 61, 24) and np.isfinite(result).all()


def test_training_checkpoint_and_grouped_predictions(store, tmp_path):
    torch.set_num_threads(1)
    boardings, _ = store
    path = train(boardings, boardings, forecast_days=2, history_days=14,
                 epochs=1, batch_size=4, output=tmp_path / "model.pt")
    model = load_model(path)
    dataset = ForecastDataset(boardings, 2, 14)
    history, request, targets = collate_samples([dataset[0], dataset[-1]])
    with torch.no_grad():
        grouped = model(history, request).numpy().reshape(2, 2, 24)
    assert targets.shape == (4, 24)
    for i, index in enumerate((0, len(dataset) - 1)):
        route, cutoff = dataset.index[index]
        reference = fixed_forecast(model, boardings, cutoff, 2, 14)
        np.testing.assert_allclose(grouped[i], reference[boardings.routes.index(route)],
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind, message", [
    ("duplicate", "Duplicate"), ("missing", "Missing scheduled events"),
    ("invalid", "binary event flag"), ("missing_column", "requires columns"),
])
def test_invalid_schedule_fails_instead_of_silent_zeroes(tmp_path, kind, message):
    path = tmp_path / "events.csv"
    write_events(path, days=1)
    lines = path.read_text().splitlines()
    if kind == "duplicate":
        lines.append(lines[1])
    elif kind == "missing":
        del lines[1]
    elif kind == "invalid":
        fields = lines[1].split(";")
        fields[3] = "2"
        lines[1] = ";".join(fields)
    else:
        lines[0] = lines[0].replace("event_near_route", "wrong_column")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=message):
        events = load_events(path)
        events.window(1, date(2025, 9, 1), 24)


def test_unconfigured_events_are_explicit():
    boardings = Boardings(np.ones((1, 20 * 24), dtype=np.float32),
                          (1,), date(2025, 9, 1), 1.0)
    with pytest.raises(ValueError, match="events_path"):
        sample(boardings, 1, date(2025, 9, 15), 2, 14)


def test_validation_uses_the_same_inputs(store):
    boardings, _ = store
    boardings.counts.fill(1)
    score = evaluate_model(ForecastNetwork(1), boardings, date(2025, 9, 15), 2, 14)
    assert np.isfinite(score)


def test_editing_the_csv_refreshes_the_cache(store):
    boardings, path = store
    previous = boardings.scheduled_events
    lines = path.read_text().splitlines()
    fields = lines[1].split(";")
    fields[3] = "0.0"
    lines[1] = ";".join(fields)
    path.write_text("\n".join(lines) + "\n")
    refreshed = load_events(path)
    assert refreshed is not previous
    assert previous.window(1, date(2025, 9, 1), 24)[0, 0] == 1
    assert refreshed.window(1, date(2025, 9, 1), 24)[0, 0] == 0


def test_runner_reuses_context_and_matches_existing_forecast(store):
    boardings, _ = store
    model = ForecastNetwork(1)
    runner = InferenceRunner(model)
    cutoff = date(2025, 9, 15)
    expected = fixed_forecast(model, boardings, cutoff, 61, 14)
    with patch.object(model, "encode_history", wraps=model.encode_history) as encode:
        context = runner.apply_context(boardings, 7, cutoff)
        for offset in (0, 4, 60):
            got = runner.predict_day(context, cutoff + timedelta(days=offset))
            assert got.shape == (24,) and got.dtype == np.float32
            np.testing.assert_allclose(got, expected[0, offset], rtol=1e-5, atol=1e-5)
        assert encode.call_count == 1
    combined = runner.predict(boardings, 7, cutoff, cutoff)
    np.testing.assert_allclose(combined, expected[0, 0], rtol=1e-5, atol=1e-5)


def test_runner_keeps_contexts_independent_and_rejects_foreign_context(store):
    boardings, _ = store
    runner = InferenceRunner(ForecastNetwork(1))
    cutoff = date(2025, 9, 15)
    context = runner.apply_context(boardings, 7, cutoff)
    expected = runner.predict_day(context, cutoff)
    runner.apply_context(boardings, 1, cutoff + timedelta(days=1))
    np.testing.assert_array_equal(runner.predict_day(context, cutoff), expected)
    other = InferenceRunner(ForecastNetwork(1))
    with pytest.raises(ValueError, match="different runner"):
        other.predict_day(context, cutoff)
    for day in (cutoff - timedelta(days=1), cutoff + timedelta(days=61)):
        with pytest.raises(ValueError, match="within 61 days"):
            runner.predict_day(context, day)


def test_apply_context_does_not_require_future_events_or_labels(store, tmp_path, monkeypatch):
    boardings, _ = store
    events = tmp_path / "history_only.csv"
    write_events(events, days=14)
    boardings.scheduled_events = load_events(events)
    def forbidden(*args):
        raise AssertionError("Runner read future boarding labels")
    monkeypatch.setattr(boardings, "target", forbidden)
    runner = InferenceRunner(ForecastNetwork(1))
    context = runner.apply_context(boardings, 7, date(2025, 9, 15))
    with pytest.raises(ValueError, match="do not cover"):
        runner.predict_day(context, date(2025, 9, 15))
