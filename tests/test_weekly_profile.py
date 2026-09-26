from datetime import date

import numpy as np
import pytest

from tram_forecast.evaluate import weekly_profile


class WeeklyStore:
    end = date(2025, 9, 1)
    routes = (1,)

    def history(self, route, cutoff, include_events, history_days):
        assert cutoff == self.end and not include_events
        # Three weeks with values 10, 20 and 30 respectively.
        counts = np.repeat([10., 20., 30.], 7 * 24)
        return counts[-history_days * 24:], None


def test_weekly_baselines_use_requested_history():
    store = WeeklyStore()
    np.testing.assert_array_equal(weekly_profile(store, 14, 7), np.full((1, 14, 24), 30.))
    np.testing.assert_array_equal(weekly_profile(store, 14, 21), np.full((1, 14, 24), 20.))
    with pytest.raises(ValueError, match='seven'):
        weekly_profile(store, 7, 6)
