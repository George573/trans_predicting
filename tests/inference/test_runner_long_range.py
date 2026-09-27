"""Run with: pytest -q --confcutdir=tests/inference tests/inference."""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from tram_forecast.data import Boardings, request_inputs
from tram_forecast.events import ScheduledEvents
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import predict_autoregressive
from tram_forecast.runner import InferenceRunner

torch.set_num_threads(1)


class IncrementModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.leads = []

    def encode_history(self, history):
        return history["counts"][:, 0, -24:] + self.weight

    def predict_day(self, encoded, route_indices, request_calendar, lead):
        self.leads.extend(lead.tolist())
        return encoded + lead[:, None]


def boardings(cutoff, event_days):
    start = cutoff - timedelta(days=14)
    events = ScheduledEvents(np.zeros((1, 3, event_days * 24), dtype=np.float32), (1,), start)
    return Boardings(np.full((1, 14 * 24), 2, dtype=np.float32), (1,), start, 2.0, events)


def test_autoregression_uses_normal_prediction_without_future_labels():
    cutoff = date(2025, 7, 1)
    data = boardings(cutoff, 14 + 120)
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(data, 1, cutoff)
    with patch.object(runner, "predict_day", wraps=runner.predict_day) as normal:
        result = predict_autoregressive(runner, context, cutoff + timedelta(days=119))
    np.testing.assert_array_equal(result, np.full(24, 122, dtype=np.float32))
    assert normal.call_count == 120
    assert runner.model.leads == [1.0] * 120
    np.testing.assert_array_equal(data.counts, np.full((1, 14 * 24), 2))
    np.testing.assert_array_equal(runner.predict_day(context, cutoff), np.full(24, 3))


def test_missing_future_schedule_defaults_to_zero():
    cutoff = date(2025, 7, 1)
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(boardings(cutoff, 14 + 10), 1, cutoff)
    future = cutoff + timedelta(days=20)
    np.testing.assert_array_equal(predict_autoregressive(runner, context, future), np.full(24, 23))
    features = request_inputs(context._events, 1, [future], missing_zero=True)
    np.testing.assert_array_equal(features[0, -3:], np.zeros(3))
    with pytest.raises(ValueError, match="do not cover"):
        context._events.window(1, future, 24)


def test_context_can_be_updated_one_day_at_a_time():
    cutoff = date(2025, 7, 1)
    data = boardings(cutoff, 14 + 3)
    runner = InferenceRunner(IncrementModel())
    original = runner.apply_context(data, 1, cutoff)
    updated = runner.update_context(original, cutoff, np.full(24, 10))
    assert updated.cutoff == cutoff + timedelta(days=1)
    np.testing.assert_array_equal(runner.predict_day(updated, updated.cutoff), np.full(24, 11))
    np.testing.assert_array_equal(runner.predict_day(original, cutoff), np.full(24, 3))
    np.testing.assert_array_equal(data.counts, np.full((1, 14 * 24), 2))
    with pytest.raises(ValueError, match="context cutoff"):
        runner.update_context(updated, cutoff, np.full(24, 10))
    for counts in (np.ones(23), np.full(24, -1), np.full(24, np.nan)):
        with pytest.raises(ValueError, match="24 finite, nonnegative"):
            runner.update_context(original, cutoff, counts)
    with pytest.raises(ValueError, match="different runner"):
        InferenceRunner(IncrementModel()).update_context(original, cutoff, np.ones(24))
    with pytest.raises(TypeError, match="ForecastContext"):
        runner.update_context(None, cutoff, np.ones(24))


def test_context_rejects_invalid_initial_counts_and_accepts_missing_events():
    cutoff = date(2025, 7, 1)
    data = boardings(cutoff, 14)
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(data, 1, cutoff)
    assert runner.update_context(context, cutoff, np.ones(24)).cutoff == cutoff + timedelta(days=1)
    data.counts[0, 0] = np.nan
    with pytest.raises(ValueError, match="Context history"):
        runner.apply_context(data, 1, cutoff)


def test_direct_prediction_accepts_later_year_without_calendar_or_events():
    cutoff = date(2025, 12, 15)
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(boardings(cutoff, 14), 1, cutoff)
    future = date(2026, 3, 15)
    np.testing.assert_array_equal(runner.predict_day(context, future), np.full(24, 93))
    assert runner.model.leads[-1] == (future - cutoff).days + 1
    np.testing.assert_array_equal(predict_autoregressive(runner, context, future), np.full(24, 93))
    flags = request_inputs(context._events, 1, [future], missing_zero=True)
    np.testing.assert_array_equal(flags[0, 4:], np.zeros(6))
    with pytest.raises(ValueError, match="on or after"):
        runner.predict_day(context, cutoff - timedelta(days=1))


def test_foreign_context_is_rejected():
    cutoff = date(2025, 7, 1)
    context = InferenceRunner(IncrementModel()).apply_context(boardings(cutoff, 40), 1, cutoff)
    with pytest.raises(ValueError, match="different runner"):
        predict_autoregressive(InferenceRunner(IncrementModel()), context, cutoff)


def test_cnn_autoregression_beyond_training_horizon():
    cutoff = date(2025, 7, 1)
    runner = InferenceRunner(ForecastNetwork(2))
    context = runner.apply_context(boardings(cutoff, 14 + 70), 1, cutoff)
    result = predict_autoregressive(runner, context, cutoff + timedelta(days=69))
    assert result.shape == (24,) and np.isfinite(result).all() and (result >= 0).all()


def test_inference_accepts_no_event_source():
    cutoff = date(2025, 7, 1)
    data = boardings(cutoff, 14)
    data.scheduled_events = None
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(data, 1, cutoff)
    np.testing.assert_array_equal(runner.predict_day(context, cutoff), np.full(24, 3))
    np.testing.assert_array_equal(predict_autoregressive(runner, context, cutoff + timedelta(days=2)),
                                  np.full(24, 5))


def test_partial_event_schedule_preserves_known_flags_and_zeroes_missing():
    cutoff = date(2025, 7, 1)
    data = boardings(cutoff, 15)
    values = np.full((1, 3, 15 * 24), np.nan, dtype=np.float32)
    values[0, 1, 14 * 24 + 18] = 1
    data.scheduled_events = ScheduledEvents(values, (1,), cutoff - timedelta(days=14))
    runner = InferenceRunner(IncrementModel())
    context = runner.apply_context(data, 1, cutoff)
    flags = request_inputs(context._events, 1, [cutoff, cutoff + timedelta(days=1)],
                           missing_zero=True)
    np.testing.assert_array_equal(flags[:, -3:], [[0, 1, 0], [0, 0, 0]])
    assert np.isfinite(runner.predict_day(context, cutoff + timedelta(days=1))).all()
