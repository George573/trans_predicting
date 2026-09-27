from datetime import date

import numpy as np
import pytest
import torch
from torch import nn

from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.data import Boardings


class ZeroEvents:
    def window(self, route, start, hours):
        return np.zeros((3, hours), dtype=np.float32)


class RampModel(nn.Module):
    history_days = 7

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def encode_history(self, history):
        return history["counts"][:, 0, -24:] + self.offset

    def predict_day(self, encoded, routes, calendar, lead):
        return encoded + lead[:, None]


def data():
    return Boardings(
        np.full((1, 92 * 24), 2.0, dtype=np.float32),
        (1,),
        date(2025, 8, 1),
        2.0,
        ZeroEvents(),
    )


@pytest.mark.parametrize("block", [1, 7])
def test_recursion_uses_own_predictions_without_future_labels(block):
    boardings = data()
    cutoff = date(2025, 9, 1)
    original = boardings.counts.copy()
    predicted = recursive_forecast(RampModel(), boardings, cutoff, block_days=block)
    expected = np.broadcast_to(np.arange(3, 64)[None, :, None], (1, 61, 24))
    np.testing.assert_allclose(predicted, expected)
    np.testing.assert_array_equal(boardings.counts, original)
    boardings.counts[:, 31 * 24 :] = 1e8
    np.testing.assert_array_equal(
        predicted, recursive_forecast(RampModel(), boardings, cutoff, block_days=block)
    )
    # The same result is available with the future labels physically absent.
    boardings.counts = original[:, : 31 * 24]
    np.testing.assert_array_equal(
        predicted, recursive_forecast(RampModel(), boardings, cutoff, block_days=block)
    )


def test_anchor_feedback_and_input_validation():
    boardings = data()
    anchor = np.full((1, 61, 24), 10.0, dtype=np.float32)
    predicted = recursive_forecast(
        RampModel(), boardings, date(2025, 9, 1), anchor=anchor, recursive_weight=0.0
    )
    np.testing.assert_array_equal(predicted, anchor)
    blended = recursive_forecast(
        RampModel(), boardings, date(2025, 9, 1), anchor=anchor, recursive_weight=0.5
    )
    assert blended[0, 0, 0] == 6.5
    assert blended[0, 1, 0] == 8.75  # Blended day one is fed into day two.
    with pytest.raises(ValueError, match="invalid horizon"):
        recursive_forecast(RampModel(), boardings, date(2025, 9, 1), block_days=0)
    with pytest.raises(ValueError, match="anchor must"):
        recursive_forecast(
            RampModel(), boardings, date(2025, 9, 1), anchor=anchor[:, :3]
        )
