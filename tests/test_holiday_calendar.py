from datetime import date, datetime, timedelta
from io import StringIO

import numpy as np
import pytest
import torch

from tram_forecast.data import Boardings, ForecastDataset, calendar, collate_samples, request_calendar
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import fixed_forecast
from tram_forecast.russian_calendar import day_flags, is_holiday, parse_calendar


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
    with pytest.raises(ValueError, match="No Russian production calendar for 2026"):
        is_holiday(date(2026, 1, 1))


def test_hourly_and_requested_holiday_flags_agree():
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(4)]
    hourly = calendar(datetime.combine(d, datetime.min.time()) + timedelta(hours=h)
                      for d in days for h in range(24))
    requested = request_calendar(days)
    assert hourly.shape == (9, 96)
    assert requested.shape == (4, 7)
    assert hourly.dtype == requested.dtype == np.float32
    expected = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(hourly[-3:].T, np.repeat(expected, 24, axis=0))
    np.testing.assert_array_equal(requested[:, -3:], expected)


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


def test_production_calendar_year_totals():
    flags = [day_flags(date(2025, 1, 1) + timedelta(days=i)) for i in range(365)]
    assert sum(f.is_holiday for f in flags) == 19
    assert sum(f.is_day_off for f in flags) == 118
    assert sum(f.is_short_working_day for f in flags) == 4
    assert not day_flags(date(2025, 11, 1)).is_day_off


@pytest.mark.parametrize("record, message", [
    ('<day d="05.02"/>', "include d and t"),
    ('<day d="02.30" t="1"/>', "Invalid calendar day"),
    ('<day d="05.02" t="9"/>', "Unknown calendar day type"),
    ('<day d="05.02" t="2" h="5"/>', "must be a day off"),
    ('<day d="05.02" t="1" f="02.30"/>', "Invalid calendar day"),
    ('<day d="05.02" t="1"/><day d="05.02" t="1"/>', "Duplicate"),
])
def test_invalid_overrides_are_rejected(record, message):
    xml = StringIO(f'<calendar year="2025"><days>{record}</days></calendar>')
    with pytest.raises(ValueError, match=message):
        parse_calendar(xml, 2025)


def test_calendar_year_validation_and_working_day_override():
    xml = '<calendar year="2026"><days><day d="01.03" t="3"/></days></calendar>'
    with pytest.raises(ValueError, match="year mismatch"):
        parse_calendar(StringIO(xml), 2025)
    flags = parse_calendar(StringIO(xml), 2026)[date(2026, 1, 3)]
    assert not flags.is_day_off and not flags.is_short_working_day


def test_missing_days_section_is_rejected():
    with pytest.raises(ValueError, match="missing its days"):
        parse_calendar(StringIO('<calendar year="2025"/>'), 2025)
