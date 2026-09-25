#!/usr/bin/env python3
"""Independently check a completed tram profile and its detail files."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


REQUIRED = {"metadata", "coverage", "integrity", "hourly_grid", "reconciliation",
            "event_lengths", "padding_simulation", "categorical_profiles",
            "timestamp_quality", "target_statistics", "sample_availability",
            "recommendations", "limitations", "detail_files"}


def reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def check(path: Path):
    with open(path, encoding="utf-8") as handle:
        profile = json.load(handle, parse_constant=reject_nonfinite)
    assert REQUIRED <= set(profile), REQUIRED - set(profile)
    meta = profile["metadata"]
    coverage = profile["coverage"]
    grid = profile["hourly_grid"]["summary_by_period"]
    periods = meta["configuration"]
    route_count = len(coverage["routes"]["submission"])
    for name in ("train", "validation"):
        start, end = (datetime.fromisoformat(value) for value in periods[name])
        expected = route_count * (end - start).days * 24
        assert grid[name]["expected_hours"] == expected, (name, expected)
        assert grid[name]["observed_label_hours"] + grid[name]["absent_label_hours"] == expected
    bad = profile["integrity"]["raw_parse_failures"]
    parsed = sum(coverage["raw_rows_by_event_time_period"].values())
    bad_rows = sum(bad.values())
    assert sum(coverage["raw_rows_by_file"].values()) == parsed + bad_rows
    for period_name, fields in profile["categorical_profiles"].items():
        rows = coverage["raw_rows_by_event_time_period"].get(period_name, 0)
        for field, item in fields.items():
            assert item["missing_count"] + item["nonmissing_rows"] == rows, (period_name, field)
    for name in ("train", "validation"):
        item = profile["reconciliation"]["by_period"][name]
        assert item["raw_success"] - item["label_boardings"] == item["signed_difference"]
        assert item["compared_hours"] <= grid[name]["expected_hours"]
    samples = profile["sample_availability"]
    assert samples["total_sample_identities"] == sum(samples["sample_identities_by_lead"].values())
    for name, item in profile["detail_files"].items():
        detail = path.parent / item["relative_path"]
        assert detail.is_file(), detail
        with open(detail, encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle)) == item["schema"], name
    assert all(meta["consistency_checks"].values())
    print(f"Validated {path} and {len(profile['detail_files'])} detail tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    check(parser.parse_args().profile)
