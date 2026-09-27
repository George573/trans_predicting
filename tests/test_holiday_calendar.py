from datetime import date, datetime, timedelta

import numpy as np
import pytest
import torch

from tram_forecast.data import Boardings, ForecastDataset, calendar, collate_samples, request_calendar
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import fixed_forecast
from tram_forecast.russian_calendar import is_holiday


@pytest.mark.parametrize("day", [
    date(2025, 1, 1), date(2025, 1, 8), date(2025, 2, 23),
    date(2025, 3, 8), date(2025, 5, 1), date(2025, 5, 2),
    date(2025, 5, 8), date(2025, 5, 9), date(2025, 6, 12),
    date(2025, 6, 13), date(2025, 11, 3), date(2025, 11, 4),
    date(2025, 12, 31),
])
def test_federal_holidays_and_transfers(day):
    assert is_holiday(day)


@pytest.mark.parametrize("day", [
    date(2025, 1, 9), date(2025, 9, 6), date(2025, 11, 1),
])
def test_regular_days_and_working_saturday(day):
    assert not is_holiday(day)


def test_unsupported_year_is_explicit():
    with pytest.raises(ValueError, match="supports 2025"):
        is_holiday(date(2026, 1, 1))


def test_hourly_and_requested_holiday_flags_agree():
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(4)]
    hourly = calendar(datetime.combine(d, datetime.min.time()) + timedelta(hours=h)
                      for d in days for h in range(24))
    requested = request_calendar(days)
    assert hourly.shape == (7, 96)
    assert requested.shape == (4, 5)
    assert hourly.dtype == requested.dtype == np.float32
    np.testing.assert_array_equal(hourly[-1], np.repeat([0, 0, 1, 1], 24))
    np.testing.assert_array_equal(requested[:, -1], [0, 0, 1, 1])


def test_training_and_inference_accept_holiday_features():
    boardings = Boardings(np.ones((1, 45 * 24), dtype=np.float32),
                          (1,), date(2025, 10, 15), 1.0)
    dataset = ForecastDataset(boardings, forecast_days=7, history_days=14)
    history, request, targets = collate_samples([dataset[0]])
    model = ForecastNetwork(1.0)
    prediction = model(history, request)
    assert prediction.shape == targets.shape == (7, 24)
    (prediction - targets).abs().mean().backward()
    assert torch.isfinite(model.encoder[0].weight.grad).all()
    forecast = fixed_forecast(model, boardings, date(2025, 11, 1), 7, 14)
    assert forecast.shape == (1, 7, 24)
    assert np.isfinite(forecast).all()
