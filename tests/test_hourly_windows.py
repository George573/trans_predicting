import csv
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from experiments.legacy.hourly_windows import (
    iter_boarding_windows,
    iter_hourly_counts,
    sample_boarding_windows,
)


class HourlyWindowsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.train = root / "train.csv"
        self.test = root / "test.csv"
        self.write_rows(
            self.train,
            [
                (1, "2025-08-31", 22, 4),
                (1, "2025-08-31", 23, 6),
                (7, "2025-08-31", 23, 9),
            ],
        )
        self.write_rows(
            self.test,
            [
                (1, "2025-09-01", 1, 8),
                (7, "2025-09-01", 1, 3),
            ],
        )

    @staticmethod
    def write_rows(path, rows):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["route", "date", "hour", "boardings"])
            writer.writerows(rows)

    def options(self):
        return dict(
            series_start=date(2025, 8, 31),
            series_end=date(2025, 9, 1),
            history_hours=2,
            horizon_hours=1,
            target_start=date(2025, 9, 1),
            routes={1, 5},
        )

    def test_gap_fill_and_cross_file_history(self):
        windows = list(iter_boarding_windows([self.train, self.test], **self.options()))
        first = windows[0]
        self.assertEqual(first.route, 1)
        self.assertEqual(first.target_time, datetime(2025, 9, 1, 0))
        self.assertEqual(first.history_boardings, (4, 6))
        self.assertEqual(first.target_boardings, 0)
        self.assertFalse(first.target_observed)
        self.assertEqual(windows[1].history_boardings, (6, 0))
        self.assertEqual(windows[1].history_observed, (True, False))
        self.assertEqual(windows[1].target_boardings, 8)
        self.assertTrue(all(window.route == 1 for window in windows))

    def test_horizon_and_route_boundary(self):
        options = self.options()
        options.update(horizon_hours=2, routes=None)
        windows = list(iter_boarding_windows([self.train, self.test], **options))
        route_one = next(
            window
            for window in windows
            if window.route == 1 and window.target_time == datetime(2025, 9, 1, 1)
        )
        route_seven = next(
            window
            for window in windows
            if window.route == 7 and window.target_time == datetime(2025, 9, 1, 1)
        )
        self.assertEqual(route_one.history_boardings, (4, 6))
        self.assertEqual(route_one.target_time, datetime(2025, 9, 1, 1))
        self.assertEqual(route_one.target_boardings, 8)
        self.assertEqual(route_seven.history_boardings[-1], 9)

    def test_sampling_is_reproducible(self):
        first = sample_boarding_windows(
            [self.train, self.test], sample_size=2, seed=7, **self.options()
        )
        second = sample_boarding_windows(
            [self.train, self.test], sample_size=2, seed=7, **self.options()
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_duplicate_across_files_is_rejected(self):
        self.write_rows(self.test, [(1, "2025-08-31", 23, 6)])
        with self.assertRaisesRegex(ValueError, "Duplicate route-hour"):
            list(
                iter_hourly_counts(
                    [self.train, self.test],
                    series_start=date(2025, 8, 31),
                    series_end=date(2025, 9, 1),
                )
            )


if __name__ == "__main__":
    unittest.main()
