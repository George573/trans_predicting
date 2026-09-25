#!/usr/bin/env python3
"""Profile the full tram dataset with bounded memory.

Run from the repository root, for example:

    python tools/profile_dataset.py --output-dir outputs/data_profile \
      --temp-dir outputs/data_profile/tmp --progress-every 5000000

No model is built. Raw CSVs are streamed once; route-hour aggregates and
categorical counters are retained in memory. Exact raw-row duplicate checking
uses an external sort when --exact-duplicates is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


RAW_COLUMNS = ["tran_no", "device_no", "tran_date_time", "begin_date_time",
               "input_date_time", "crd_hashcode", "validation_result",
               "tran_type_id", "place_id", "good_type", "pass_route",
               "ngpt_route", "bus_exit_no", "garage_number"]
LABEL_COLUMNS = ["route", "date", "hour", "boardings"]
CATEGORICAL = ["device_no", "validation_result", "tran_type_id", "place_id",
               "good_type", "pass_route", "ngpt_route", "bus_exit_no",
               "garage_number"]
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


def period(ts, train_start, val_start, val_end):
    if train_start <= ts < val_start:
        return "train"
    if val_start <= ts < val_end:
        return "validation"
    return "out_of_period"


def parse_dt(value):
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def route_num(value):
    try:
        return int(value.split()[0]) if value else None
    except (ValueError, IndexError):
        return None


def describe(values, *, approximate=False, missing=0):
    values = sorted(values)
    n = len(values)
    if not n:
        return {"count": 0, "missing_count": missing, "min": None, "p50": None,
                "p90": None, "p95": None, "p99": None, "max": None,
                "mean": None, "std": None, "method": "unavailable: no values"}
    def q(p):
        position = (n - 1) * p
        low = int(position)
        high = min(low + 1, n - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": n, "missing_count": missing, "min": values[0],
            "p50": q(.5), "p90": q(.9), "p95": q(.95), "p99": q(.99),
            "max": values[-1], "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "method": "approximate reservoir + linear interpolation" if approximate
            else "exact sorted values + linear interpolation"}


class Reservoir:
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.values = []

    def add(self, value):
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
        else:
            i = self.rng.randrange(self.seen)
            if i < self.capacity:
                self.values[i] = value

    def summary(self, missing=0):
        result = describe(self.values, approximate=self.seen > len(self.values), missing=missing)
        result["population_count"] = self.seen
        result["reservoir_size"] = len(self.values)
        return result


class ProgressBar:
    """Small dependency-free progress bar for streamed CSV scans."""

    def __init__(self, name, source_bytes, row_limit=None, enabled=True):
        self.name = name
        self.source_bytes = source_bytes
        self.row_limit = row_limit
        self.enabled = enabled
        self.interactive = sys.stderr.isatty()
        self.started = time.monotonic()
        self.last_length = 0
        self.last_rows = None

    def update(self, rows, bytes_read, *, done=False):
        if not self.enabled:
            return
        if done and rows == self.last_rows:
            if self.interactive:
                sys.stderr.write("\n")
                sys.stderr.flush()
            return
        self.last_rows = rows
        denominator = self.row_limit or self.source_bytes
        numerator = rows if self.row_limit else bytes_read
        fraction = min(1.0, numerator / denominator) if denominator else 1.0
        width = 24
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(time.monotonic() - self.started, 0.001)
        rate = rows / elapsed
        message = (f"{self.name} [{bar}] {fraction:5.1%} "
                   f"{rows:,} rows  {rate:,.0f} rows/s  {elapsed:.0f}s")
        if self.interactive:
            sys.stderr.write("\r" + message + " " * max(0, self.last_length - len(message)))
            self.last_length = len(message)
            if done:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(message + "\n")
        sys.stderr.flush()


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def category_report(counter, total_rows):
    nonmissing = sum(counter.values())
    ranked = counter.most_common()
    covered = 0
    cover = {}
    for i, (_, count) in enumerate(ranked, 1):
        covered += count
        for threshold in (.95, .99):
            if threshold not in cover and nonmissing and covered / nonmissing >= threshold:
                cover[threshold] = i
    rare = {}
    for name, predicate in [("once", lambda c: c == 1),
                            ("lt5", lambda c: c < 5), ("lt20", lambda c: c < 20)]:
        count = sum(1 for v in counter.values() if predicate(v))
        rows = sum(v for v in counter.values() if predicate(v))
        rare[name] = {"categories": count, "rows": rows,
                      "row_pct_nonmissing": 100 * rows / nonmissing if nonmissing else None}
    return {"distinct_nonmissing": len(counter), "missing_count": total_rows - nonmissing,
            "missing_pct": 100 * (total_rows - nonmissing) / total_rows if total_rows else None,
            "nonmissing_rows": nonmissing,
            "top20": [{"value": k, "count": v,
                       "pct_nonmissing": 100 * v / nonmissing if nonmissing else None}
                      for k, v in ranked[:20]],
            "rare": rare, "categories_for_95pct": cover.get(.95),
            "categories_for_99pct": cover.get(.99),
            "embedding_parameters": {str(d): (len(counter) + 2) * d for d in (4, 8, 16)},
            "method": "exact counter; embedding count includes distinct missing and unknown tokens"}


def scan_raw(args, output):
    rng = random.Random(args.seed)
    sources = [("train.csv", args.raw_train), ("test.csv", args.raw_validation)]
    raw_hour = Counter()
    success_hour = Counter()
    raw_day = Counter()
    raw_route_month = Counter()
    raw_file_route_month = Counter()
    raw_period = Counter()
    raw_routes = defaultdict(lambda: [None, None])
    missing_month = Counter()
    invalid_month = Counter()
    timestamp_bounds = {name: [None, None] for name in ("tran_date_time", "begin_date_time", "input_date_time")}
    cat = {p: {f: Counter() for f in CATEGORICAL} for p in ("train", "validation")}
    cat_month = defaultdict(set)
    cat_route = defaultdict(set)
    cat_routes = defaultdict(set)
    new_month = Counter()
    seen_train = {f: set() for f in CATEGORICAL}
    seen_validation = Counter()
    val_nonmissing = Counter()
    row_counts = Counter()
    bad_rows = Counter()
    out_of_file_period = Counter()
    reversals = Counter()
    previous_file_ts = {}
    previous_route_ts = {}
    delay = {"input_minus_event_seconds": Reservoir(args.reservoir_size, rng),
             "begin_minus_event_seconds": Reservoir(args.reservoir_size, rng)}
    delay_by_result = defaultdict(lambda: Reservoir(min(20000, args.reservoir_size), rng))
    delay_counts = Counter()
    timestamp_precision = Counter()
    digest_path = args.temp_dir / "raw_row_digests.tsv"
    digest_handle = open(digest_path, "w", encoding="ascii") if args.exact_duplicates else None
    sequence_path = args.temp_dir / "route_second_keys.tsv"
    sequence_handle = open(sequence_path, "w", encoding="ascii") if args.sequence_sort else None
    file_bounds = {}
    file_schemas = {}
    file_scanned_bytes = {}
    row_limit_hit = {}
    scan_started = time.monotonic()

    try:
        for file_name, path in sources:
            print(f"Scanning {path} ...", flush=True)
            file_bounds[file_name] = [None, None]
            progress = ProgressBar(file_name, path.stat().st_size,
                                   args.max_raw_rows_per_file,
                                   enabled=not args.no_progress_bar)
            progress.update(0, 0)
            with open(path, encoding="utf-8-sig", newline="", buffering=1024 * 1024) as handle:
                reader = csv.reader(handle, delimiter=";")
                header = next(reader)
                file_schemas[file_name] = header
                if header != RAW_COLUMNS:
                    raise ValueError(f"Unexpected raw schema in {path}: {header}")
                for cells in reader:
                    if (args.max_raw_rows_per_file is not None and
                            row_counts[file_name] >= args.max_raw_rows_per_file):
                        row_limit_hit[file_name] = True
                        break
                    row_counts[file_name] += 1
                    if len(cells) != len(RAW_COLUMNS):
                        bad_rows[(file_name, "field_count")] += 1
                        continue
                    row = dict(zip(RAW_COLUMNS, cells))
                    event_time = parse_dt(row["tran_date_time"])
                    if event_time:
                        timestamp_precision["fractional_seconds" if event_time.microsecond else "whole_seconds"] += 1
                    month = event_time.strftime("%Y-%m") if event_time else "unknown"
                    for field, value in row.items():
                        if not value:
                            missing_month[(month, field)] += 1
                    for field in timestamp_bounds:
                        val = parse_dt(row[field])
                        if row[field] and val is None:
                            invalid_month[(month, field)] += 1
                        if val is not None:
                            bounds = timestamp_bounds[field]
                            bounds[0] = min(bounds[0], val) if bounds[0] else val
                            bounds[1] = max(bounds[1], val) if bounds[1] else val
                    if event_time is None:
                        bad_rows[(file_name, "invalid_event_time")] += 1
                        continue
                    p = period(event_time, args.train_start, args.validation_start, args.validation_end)
                    raw_period[p] += 1
                    expected = "train" if file_name == "train.csv" else "validation"
                    if p != expected:
                        out_of_file_period[(file_name, p)] += 1
                    route = route_num(row["ngpt_route"])
                    if route is None:
                        invalid_month[(month, "ngpt_route")] += 1
                    else:
                        hour = event_time.replace(minute=0, second=0, microsecond=0)
                        key = (route, hour)
                        raw_hour[key] += 1
                        raw_day[(route, event_time.date())] += 1
                        raw_route_month[(p, route, month)] += 1
                        raw_file_route_month[(file_name, route, month)] += 1
                        bounds = raw_routes[route]
                        bounds[0] = min(bounds[0], event_time) if bounds[0] else event_time
                        bounds[1] = max(bounds[1], event_time) if bounds[1] else event_time
                        if row["validation_result"] == "1":
                            success_hour[key] += 1
                        if sequence_handle:
                            sequence_handle.write(f"{route:03d}\t{event_time.isoformat()}\n")
                    if row["validation_result"] and not row["validation_result"].isdigit():
                        invalid_month[(month, "validation_result")] += 1
                    old = previous_file_ts.get(file_name)
                    if old and event_time < old:
                        reversals[(file_name, "overall")] += 1
                    previous_file_ts[file_name] = event_time
                    if route is not None:
                        old = previous_route_ts.get((file_name, route))
                        if old and event_time < old:
                            reversals[(file_name, str(route))] += 1
                        previous_route_ts[(file_name, route)] = event_time
                    bounds = file_bounds[file_name]
                    bounds[0] = min(bounds[0], event_time) if bounds[0] else event_time
                    bounds[1] = max(bounds[1], event_time) if bounds[1] else event_time
                    if p in cat:
                        for field in CATEGORICAL:
                            value = row[field]
                            if not value:
                                continue
                            cat[p][field][value] += 1
                            cat_month[(p, month, field)].add(value)
                            if route is not None:
                                cat_route[(p, route, field)].add(value)
                                cat_routes[(p, field, value)].add(route)
                            if p == "train":
                                if value not in seen_train[field]:
                                    new_month[(month, field)] += 1
                                    seen_train[field].add(value)
                            else:
                                val_nonmissing[field] += 1
                                if value not in seen_train[field]:
                                    seen_validation[field] += 1
                    for field, result_key in [("input_date_time", "input_minus_event_seconds"),
                                              ("begin_date_time", "begin_minus_event_seconds")]:
                        parsed = parse_dt(row[field])
                        if parsed:
                            seconds = (parsed - event_time).total_seconds()
                            delay[result_key].add(seconds)
                            if field == "input_date_time":
                                if seconds < 0: delay_counts["negative_load_delay"] += 1
                                if seconds > 3600: delay_counts["load_delay_over_1h"] += 1
                                if seconds > 86400: delay_counts["load_delay_over_1d"] += 1
                            else:
                                delay_by_result[row["validation_result"]].add(seconds)
                                if abs(seconds) > 86400: delay_counts["begin_over_1d_abs"] += 1
                    if digest_handle:
                        # A hash only selects candidates; a later full-row comparison confirms them.
                        digest = hashlib.blake2b(";".join(cells).encode(), digest_size=16).hexdigest()
                        digest_handle.write(f"{digest}\t{file_name}\n")
                    if args.progress_every and row_counts[file_name] % args.progress_every == 0:
                        bytes_read = handle.buffer.tell()
                        if args.no_progress_bar:
                            print(f"  {file_name}: {row_counts[file_name]:,} rows, "
                                  f"elapsed {time.monotonic()-scan_started:.0f}s", flush=True)
                        else:
                            progress.update(row_counts[file_name], bytes_read)
                file_scanned_bytes[file_name] = handle.buffer.tell()
                progress.update(row_counts[file_name], file_scanned_bytes[file_name], done=True)
    finally:
        if digest_handle:
            digest_handle.close()
        if sequence_handle:
            sequence_handle.close()

    return locals()


def scan_labels(args):
    files = [("train", args.labels_train), ("validation", args.labels_validation)]
    values = {}
    duplicates = Counter()
    conflicts = []
    rows_by_route_month = Counter()
    route_bounds = defaultdict(lambda: [None, None])
    row_counts = Counter()
    invalid = Counter()
    schemas = {}
    for file_name, path in files:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            schemas[file_name] = reader.fieldnames
            if reader.fieldnames != LABEL_COLUMNS:
                raise ValueError(f"Unexpected label schema in {path}: {reader.fieldnames}")
            for row in reader:
                row_counts[file_name] += 1
                try:
                    route = int(row["route"])
                    day = date.fromisoformat(row["date"])
                    hour = int(row["hour"])
                    count = int(row["boardings"])
                    if not 0 <= hour <= 23:
                        invalid[(file_name, "hour")] += 1
                        continue
                    if count < 0:
                        invalid[(file_name, "negative_boardings")] += 1
                        continue
                except (ValueError, TypeError):
                    invalid[(file_name, "parse_failure")] += 1
                    continue
                timestamp = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                key = (route, timestamp)
                if key in values:
                    if values[key] == count:
                        duplicates["identical"] += 1
                    else:
                        duplicates["conflicting"] += 1
                        conflicts.append((route, timestamp.isoformat(), values[key], count))
                    continue
                values[key] = count
                p = period(timestamp, args.train_start, args.validation_start, args.validation_end)
                rows_by_route_month[(p, route, day.strftime("%Y-%m"))] += 1
                bounds = route_bounds[route]
                bounds[0] = min(bounds[0], timestamp) if bounds[0] else timestamp
                bounds[1] = max(bounds[1], timestamp) if bounds[1] else timestamp
    return locals()


def scan_submission(args):
    routes = set()
    keys = set()
    invalid = Counter()
    with open(args.submission, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        schema = reader.fieldnames
        for row in reader:
            try:
                route = int(row["route"])
                day = date.fromisoformat(row["date"])
                hour = int(row["hour"])
                if not 0 <= hour <= 23:
                    raise ValueError()
                routes.add(route)
                keys.add((route, day, hour))
            except (ValueError, TypeError):
                invalid["parse_failure"] += 1
    return {"routes": sorted(routes), "keys": len(keys), "schema": schema,
            "invalid": dict(invalid)}


def make_grid(args, raw, labels, submission, output):
    routes = submission["routes"]
    label_values = labels["values"]
    raw_hour = raw["raw_hour"]
    success_hour = raw["success_hour"]
    conflict_keys = {(r, datetime.fromisoformat(t)) for r, t, _, _ in labels["conflicts"]}
    route_series = {}
    coverage = defaultdict(lambda: Counter())
    quadrants = Counter()
    recon = defaultdict(lambda: Counter())
    recon_abs = []
    recon_abs_by_period = defaultdict(list)
    recon_nonzero = Counter()
    mismatches = []
    absent_runs = []
    absent_hour_weekday = Counter()
    entire_days = []
    entire_weeks = []
    positive_bounds = []
    suspicious = []
    total = Counter()
    for route in routes:
        series = []
        for p, start, end in [("train", args.train_start, args.validation_start),
                              ("validation", args.validation_start, args.validation_end)]:
            timestamp = start
            run_start = None
            daily_label_rows = Counter()
            weekly_label_rows = Counter()
            daily_positive = defaultdict(list)
            while timestamp < end:
                key = (route, timestamp)
                present = key in label_values
                label = label_values.get(key, 0)
                events = raw_hour.get(key, 0)
                success = success_hour.get(key, 0)
                month = timestamp.strftime("%Y-%m")
                c = coverage[(p, route, month)]
                c["expected_hours"] += 1
                c["observed_label_hours"] += present
                c["absent_label_hours"] += not present
                c["explicit_zero_labels"] += present and label == 0
                c["raw_events"] += events
                c["label_boardings"] += label
                quadrants[(p, bool(events), present)] += 1
                total[(p, "label_boardings")] += label
                total[(p, "raw_success")] += success
                total[(p, "raw_events")] += events
                if not present:
                    absent_hour_weekday[(p, route, timestamp.hour, timestamp.weekday())] += 1
                    if run_start is None:
                        run_start = timestamp
                elif run_start is not None:
                    absent_runs.append((p, route, run_start, timestamp - HOUR,
                                        int((timestamp - run_start) / HOUR)))
                    run_start = None
                if present:
                    daily_label_rows[timestamp.date()] += 1
                    weekly_label_rows[timestamp.date() - timedelta(days=timestamp.weekday())] += 1
                if label > 0:
                    daily_positive[timestamp.date()].append(timestamp.hour)
                if key not in conflict_keys:
                    delta = success - label
                    d = recon[(p, route, month, timestamp.hour)]
                    d["hours"] += 1
                    d["raw_success"] += success
                    d["label_boardings"] += label
                    d["signed_difference"] += delta
                    d["absolute_difference"] += abs(delta)
                    d["exact_match_hours"] += delta == 0
                    recon_abs.append(abs(delta))
                    recon_abs_by_period[p].append(abs(delta))
                    if success or label:
                        recon_nonzero[(p, "hours")] += 1
                        recon_nonzero[(p, "exact_matches")] += delta == 0
                    if delta:
                        mismatches.append((abs(delta), p, route, timestamp, success, label, present))
                if events and not present and len(suspicious) < 100:
                    suspicious.append((p, route, timestamp, events, success, present))
                if present and not events and len(suspicious) < 100:
                    suspicious.append((p, route, timestamp, events, success, present))
                series.append((timestamp, label, present, events, success))
                timestamp += HOUR
            if run_start is not None:
                absent_runs.append((p, route, run_start, end - HOUR,
                                    int((end - run_start) / HOUR)))
            day = start.date()
            while day < end.date():
                if daily_label_rows[day] == 0:
                    entire_days.append((p, route, day))
                hours = daily_positive.get(day)
                positive_bounds.append((p, route, day, min(hours) if hours else None,
                                        max(hours) if hours else None))
                day += DAY
            week_start = start.date() - timedelta(days=start.weekday())
            while week_start + timedelta(days=7) <= end.date():
                if week_start >= start.date() and weekly_label_rows[week_start] == 0:
                    entire_weeks.append((p, route, week_start))
                week_start += timedelta(days=7)
        route_series[route] = series
    write_csv(output / "grid_coverage.csv",
              ["period", "route", "month", "expected_hours", "observed_label_hours",
               "absent_label_hours", "explicit_zero_labels", "coverage_pct", "raw_events", "label_boardings"],
              [(p, r, m, c["expected_hours"], c["observed_label_hours"],
                c["absent_label_hours"], c["explicit_zero_labels"],
                100 * c["observed_label_hours"] / c["expected_hours"],
                c["raw_events"], c["label_boardings"])
               for (p, r, m), c in sorted(coverage.items())])
    write_csv(output / "largest_mismatches.csv",
              ["period", "route", "timestamp", "raw_success", "label_boardings",
               "label_row_present", "signed_difference"],
              [(p, r, t.isoformat(), a, b, int(pr), a-b)
               for _, p, r, t, a, b, pr in sorted(mismatches, reverse=True)[:50]])
    write_csv(output / "longest_absent_runs.csv",
              ["period", "route", "start", "end", "hours"],
              [(p, r, s.isoformat(), e.isoformat(), n)
               for p, r, s, e, n in sorted(absent_runs, key=lambda x: x[4], reverse=True)[:50]])
    write_csv(output / "raw_label_inconsistency_examples.csv",
              ["period", "route", "timestamp", "raw_events", "raw_success", "label_present"],
              [(p, r, t.isoformat(), n, s, int(pr)) for p, r, t, n, s, pr in suspicious])
    write_csv(output / "first_last_positive_hour.csv",
              ["period", "route", "date", "first_hour", "last_hour"], positive_bounds)
    return locals()


def autocorrelation(values, lag):
    if len(values) <= lag:
        return {"value": None, "reason": "insufficient aligned observations"}
    left, right = values[:-lag], values[lag:]
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = sum((a - mean_left) ** 2 for a in left)
    denom_right = sum((b - mean_right) ** 2 for b in right)
    if denom_left == 0 or denom_right == 0:
        return {"value": None, "reason": "constant aligned series"}
    return {"value": numerator / math.sqrt(denom_left * denom_right),
            "method": "exact Pearson correlation of chronological lagged pairs"}


def analyze_series(args, grid, output):
    route_series = grid["route_series"]
    event_all = []
    event_nonempty = []
    event_days = defaultdict(int)
    event_month = defaultdict(list)
    target_all = []
    target_by_route = {}
    target_by_month = defaultdict(list)
    daily = defaultdict(int)
    weekly = defaultdict(int)
    profile = defaultdict(list)
    log_counts = []
    hourly_largest = []
    rolling_windows = []
    route_daily = defaultdict(dict)
    week_boundaries = {}
    for route, series in route_series.items():
        boardings = [row[1] for row in series]
        events = [row[3] for row in series]
        target_by_route[route] = describe(boardings)
        for timestamp, label, present, event_count, _ in series:
            p = period(timestamp, args.train_start, args.validation_start, args.validation_end)
            month = timestamp.strftime("%Y-%m")
            event_all.append(event_count)
            if event_count:
                event_nonempty.append(event_count)
            event_days[(route, timestamp.date())] += event_count
            event_month[(p, route, month)].append(event_count)
            target_all.append(label)
            target_by_month[(p, route, month)].append(label)
            daily[(p, route, timestamp.date())] += label
            weekly[(p, route, timestamp.date() - timedelta(days=timestamp.weekday()))] += label
            profile[(route, timestamp.weekday(), timestamp.hour)].append(label)
            log_counts.append(math.log1p(label))
            hourly_largest.append((label, route, timestamp, present))
        prefix_events = [0]
        prefix_nonempty = [0]
        for n in events:
            prefix_events.append(prefix_events[-1] + n)
            prefix_nonempty.append(prefix_nonempty[-1] + int(n > 0))
        for i in range(504, len(series), 24):
            cutoff = series[i][0]
            counts = events[i-504:i]
            rolling_windows.append((prefix_events[i]-prefix_events[i-504], route,
                                    cutoff, prefix_nonempty[i]-prefix_nonempty[i-504],
                                    max(counts)))
    day_changes_1 = []
    day_changes_7 = []
    daily_largest = []
    for (p, route, day), value in daily.items():
        route_daily[(p, route)][day] = value
        daily_largest.append((value, p, route, day))
    for values in route_daily.values():
        for day, value in values.items():
            for lag, output_values in [(1, day_changes_1), (7, day_changes_7)]:
                previous = values.get(day - timedelta(days=lag))
                if previous is not None:
                    output_values.append(value - previous)
    weekly_changes = []
    complete_weekly = []
    for (p, route, monday), value in weekly.items():
        start = args.train_start.date() if p == "train" else args.validation_start.date()
        end = args.validation_start.date() if p == "train" else args.validation_end.date()
        complete = monday >= start and monday + timedelta(days=7) <= end
        week_boundaries[(p, route, monday)] = complete
        if complete:
            complete_weekly.append(value)
            previous = weekly.get((p, route, monday - timedelta(days=7)))
            if previous is not None and week_boundaries.get((p, route, monday - timedelta(days=7)), False):
                weekly_changes.append(value - previous)
    route_total = Counter()
    for (p, route, day), value in daily.items():
        route_total[route] += value
    total_boardings = sum(route_total.values())
    write_csv(output / "weekday_hour_profiles.csv",
              ["route", "weekday_monday0", "hour", "observations", "mean_boardings", "median_boardings"],
              [(r, d, h, len(vals), statistics.fmean(vals), statistics.median(vals))
               for (r, d, h), vals in sorted(profile.items())])
    write_csv(output / "largest_hours.csv",
              ["route", "timestamp", "boardings", "label_present"],
              [(r, t.isoformat(), n, int(p)) for n, r, t, p in sorted(hourly_largest, reverse=True)[:50]])
    write_csv(output / "largest_days.csv",
              ["period", "route", "date", "boardings"],
              [(p, r, d.isoformat(), n) for n, p, r, d in sorted(daily_largest, reverse=True)[:50]])
    write_csv(output / "largest_event_hours.csv",
              ["route", "timestamp", "raw_events"],
              [(r, t.isoformat(), n) for n, r, t, _, n in sorted(
                  [(row[3], r, row[0], row[1], row[3]) for r, s in route_series.items() for row in s],
                  reverse=True)[:50]])
    write_csv(output / "largest_event_windows.csv",
              ["route", "cutoff", "events_504h", "nonempty_hours", "max_events_one_hour"],
              [(r, t.isoformat(), n, h, m) for n, r, t, h, m in sorted(rolling_windows, reverse=True)[:50]])
    return {
        "event_lengths": {"all_route_hours": describe(event_all),
                          "nonempty_route_hours": describe(event_nonempty),
                          "route_days": describe(list(event_days.values())),
                          "rolling_504h_events": describe([x[0] for x in rolling_windows]),
                          "rolling_504h_nonempty_hours": describe([x[3] for x in rolling_windows]),
                          "rolling_504h_peak_hour_events": describe([x[4] for x in rolling_windows]),
                          "by_route_month": {f"{p}/{r}/{m}": describe(v)
                                             for (p, r, m), v in sorted(event_month.items())}},
        "target_statistics": {"all_hours": describe(target_all),
                              "zero_hours": sum(x == 0 for x in target_all),
                              "zero_pct": 100 * sum(x == 0 for x in target_all) / len(target_all),
                              "by_route": {str(r): s for r, s in target_by_route.items()},
                              "by_route_month": {f"{p}/{r}/{m}": describe(v)
                                                 for (p, r, m), v in sorted(target_by_month.items())},
                              "daily_totals": describe(list(daily.values())),
                              "complete_weekly_totals": describe(complete_weekly),
                              "incomplete_boundary_weeks": sum(not c for c in week_boundaries.values()),
                              "daily_change_lag1": describe(day_changes_1),
                              "daily_absolute_change_lag1": describe([abs(x) for x in day_changes_1]),
                              "daily_change_lag7": describe(day_changes_7),
                              "daily_absolute_change_lag7": describe([abs(x) for x in day_changes_7]),
                              "complete_weekly_change": describe(weekly_changes),
                              "log1p_boardings": describe(log_counts),
                              "autocorrelation_by_route": {str(r): {str(lag): autocorrelation(
                                  [row[1] for row in series], lag) for lag in (1, 24, 168)}
                                  for r, series in route_series.items()},
                              "route_shares_pct": {str(r): 100 * n / total_boardings
                                                   for r, n in route_total.items()}},
        "rolling_windows": rolling_windows,
    }


def sample_availability(args, grid, output):
    routes = [r for r, series in grid["route_series"].items()
              if any(row[2] for row in series if row[0] < args.validation_start)]
    cutoffs_by_route = {}
    lead_counts = Counter()
    route_month = Counter()
    target_reuse = Counter()
    groups = []
    total = 0
    suspicious_windows = 0
    cutoff_index = {}
    for route in routes:
        series = grid["route_series"][route]
        count_by_day = {row[0].date(): 0 for row in series}
        for timestamp, _, present, events, _ in series:
            if events:
                count_by_day[timestamp.date()] += 1
        cutoffs = []
        cutoff = args.train_start.date() + timedelta(days=21)
        while cutoff + timedelta(days=1) <= args.validation_start.date():
            max_lead = min(61, (args.validation_start.date() - cutoff).days)
            if max_lead >= 1:
                cutoffs.append(cutoff)
                cutoff_index[(route, cutoff)] = len(groups)
                groups.append((route, cutoff, max_lead, total))
                total += max_lead
                for lead in range(1, max_lead + 1):
                    target = cutoff + timedelta(days=lead - 1)
                    lead_counts[lead] += 1
                    route_month[(route, target.strftime("%Y-%m"))] += 1
                    target_reuse[(route, target)] += 1
                    if any(count_by_day[cutoff - timedelta(days=d)] == 0 for d in range(1, 22)):
                        suspicious_windows += 1
            cutoff += DAY
        cutoffs_by_route[route] = cutoffs
    write_csv(output / "sample_counts_by_route_month.csv",
              ["route", "requested_month", "sample_identities"],
              [(r, m, n) for (r, m), n in sorted(route_month.items())])
    reuse_hist = Counter(target_reuse.values())
    first_validation_cutoff = args.validation_start.date()
    first_submission_cutoff = args.validation_end.date()
    history_check = {}
    for name, cutoff in [("validation", first_validation_cutoff),
                         ("submission", first_submission_cutoff)]:
        hstart = datetime.combine(cutoff - timedelta(days=21), datetime.min.time())
        hend = datetime.combine(cutoff, datetime.min.time())
        history_check[name] = {
            str(r): {"history_start": hstart.isoformat(), "cutoff": hend.isoformat(),
                     "hours_in_grid": sum(hstart <= row[0] < hend for row in series),
                     "hours_with_label_row": sum(hstart <= row[0] < hend and row[2] for row in series),
                     "hours_with_raw_events": sum(hstart <= row[0] < hend and row[3] > 0 for row in series),
                     "available_504h_grid": sum(hstart <= row[0] < hend for row in series) == 504}
            for r, series in grid["route_series"].items()}
    return {
        "report": {"definition": "(route, midnight cutoff after final observed day, requested day)",
                   "history_hours": 504, "lead_days": [1, 61],
                   "routes_with_label_history": routes,
                   "excluded_routes_without_label_history": sorted(set(grid["route_series"]) - set(routes)),
                   "total_sample_identities": total,
                   "valid_cutoffs_by_route": {str(r): len(v) for r, v in cutoffs_by_route.items()},
                   "first_last_cutoff_by_route": {str(r): [v[0].isoformat(), v[-1].isoformat()]
                                                  for r, v in cutoffs_by_route.items() if v},
                   "sample_identities_by_lead": {str(k): v for k, v in sorted(lead_counts.items())},
                   "distinct_route_target_days": len(target_reuse),
                   "distinct_calendar_target_days": len({day for _, day in target_reuse}),
                   "reuse_of_route_target_day": {str(k): v for k, v in sorted(reuse_hist.items())},
                   "windows_with_any_zero_raw_event_day_in_history": suspicious_windows,
                   "suspicious_gap_rule": "at least one of 21 history days has zero raw events; not an outage assertion",
                   "training_targets_enter_validation": False,
                   "uniform_identity_sampling": "each valid identity has probability 1/N per draw; without replacement within an epoch; route/lead/month counts in this section and detail CSV",
                   "fixed_history_checks": history_check,
                   "validation_target_days": (args.validation_end.date() - args.validation_start.date()).days,
                   "submission_target_days": (args.forecast_end.date() - args.validation_end.date()).days},
        "groups": groups,
    }


def padding_simulation(args, grid, availability):
    import bisect
    rng = random.Random(args.seed)
    groups = availability["groups"]
    starts = [x[3] for x in groups]
    total = availability["report"]["total_sample_identities"]
    route_counts = {r: [row[3] for row in s] for r, s in grid["route_series"].items()}
    first = args.train_start.date()
    results = {}
    for batch_size in (1, 2, 4, 8, 16):
        if total < batch_size:
            results[str(batch_size)] = {"status": "infeasible", "reason": "fewer identities than batch size"}
            continue
        scenarios = {"global": [], "chunk_32": [], "chunk_64": [], "chunk_128": []}
        for _ in range(max(200, args.padding_batches)):
            identities = rng.sample(range(total), batch_size)
            windows = []
            for identity in identities:
                group = groups[bisect.bisect_right(starts, identity) - 1]
                route, cutoff = group[0], group[1]
                offset = (cutoff - first).days * 24
                windows.append(route_counts[route][offset-504:offset])
            actual = sum(sum(window) for window in windows)
            nonempty = sum(sum(v > 0 for v in window) for window in windows)
            if actual == 0:
                for vals in scenarios.values(): vals.append((None, 0, 0, 0))
                continue
            max_hour = max(max(window) for window in windows)
            padded = max_hour * nonempty
            scenarios["global"].append((padded / actual, actual, padded, nonempty))
            for chunk in (32, 64, 128):
                padded = 0
                for start in range(0, 504, chunk):
                    values = [v for window in windows for v in window[start:start+chunk] if v > 0]
                    if values:
                        padded += len(values) * max(values)
                scenarios[f"chunk_{chunk}"].append((padded / actual, actual, padded, nonempty))
        result = {}
        for mode, vals in scenarios.items():
            valid = [v for v in vals if v[0] is not None]
            padded_values = [v[2] for v in valid]
            result[mode] = {"status": "exact simulation on compact hourly counts",
                            "batches": len(vals), "all_empty_batches": len(vals)-len(valid),
                            "padding_multiplier": describe([v[0] for v in valid]),
                            "actual_event_positions_total": sum(v[1] for v in valid),
                            "padded_event_positions_total": sum(padded_values),
                            "empty_hours_total": sum(batch_size * 504 - v[3] for v in vals),
                            "input_tensor_bytes_at_p95_padded_positions": {
                                dtype: {str(width): int(describe(padded_values)["p95"] * width * bytes_per)
                                        if padded_values else None for width in (16, 32, 64)}
                                for dtype, bytes_per in [("float32", 4), ("float16", 2)]},
                            "memory_scope": "encoded input tensor only; excludes activations, gradients, parameters and optimizer"}
        results[str(batch_size)] = result
    return {"seed": args.seed, "distinct_identities_within_each_batch": True,
            "hour_policy": "empty hours use zero vectors and bypass event CNN",
            "chunk_policy": "same chunk index across batch, padding nonempty hours to maximum within that chunk",
            "all_empty_policy": "null multiplier and excluded from multiplier distribution",
            "batches_per_size": max(200, args.padding_batches), "batch_sizes": results}


def sorted_lines(path, args):
    env = dict(os.environ, LC_ALL="C")
    process = subprocess.Popen(
        ["sort", "-T", str(args.temp_dir), "--buffer-size", args.sort_buffer, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    assert process.stdout is not None
    progress = ProgressBar("sort:" + path.name, path.stat().st_size,
                           enabled=not args.no_progress_bar)
    progress.update(0, 0)
    emitted_bytes = 0
    emitted_lines = 0
    try:
        for line in process.stdout:
            emitted_bytes += len(line)
            emitted_lines += 1
            if emitted_lines % args.progress_every == 0:
                progress.update(emitted_lines, emitted_bytes)
            yield line
        progress.update(emitted_lines, emitted_bytes, done=True)
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr else ""
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f"External sort failed ({returncode}): {stderr}")


def check_exact_duplicates(args, raw):
    if not args.exact_duplicates:
        return {"status": "skipped", "reason": "--exact-duplicates not enabled"}
    print("External sorting row digests for exact-duplicate candidates ...", flush=True)
    path = raw["digest_path"]
    candidates = set()
    previous = None
    previous_file = None
    digest_repeats = 0
    for line in sorted_lines(path, args):
        digest, file_name = line.rstrip("\n").split("\t")
        if digest == previous:
            digest_repeats += 1
            candidates.add(digest)
        previous, previous_file = digest, file_name
    if not candidates:
        return {"status": "exact", "rule": "equal parsed 14-column row tuples",
                "within_train": 0, "within_validation": 0, "across_files": 0,
                "candidate_hash_repeats": 0, "hash_collisions": 0}
    print(f"Verifying {len(candidates):,} candidate digests against full rows ...", flush=True)
    rows = defaultdict(Counter)
    for file_name, source in [("train.csv", args.raw_train), ("test.csv", args.raw_validation)]:
        progress = ProgressBar("verify:" + file_name, source.stat().st_size,
                               args.max_raw_rows_per_file,
                               enabled=not args.no_progress_bar)
        progress.update(0, 0)
        with open(source, encoding="utf-8-sig", newline="", buffering=1024 * 1024) as handle:
            reader = csv.reader(handle, delimiter=";")
            next(reader)
            row_number = 0
            for row_number, cells in enumerate(reader, 1):
                if (args.max_raw_rows_per_file is not None and
                        row_number > args.max_raw_rows_per_file):
                    row_number = args.max_raw_rows_per_file
                    break
                if len(cells) != 14:
                    continue
                digest = hashlib.blake2b(";".join(cells).encode(), digest_size=16).hexdigest()
                if digest in candidates:
                    rows[tuple(cells)][file_name] += 1
                if row_number % args.progress_every == 0:
                    progress.update(row_number, handle.buffer.tell())
            progress.update(row_number, handle.buffer.tell(), done=True)
    within_train = sum(max(0, c["train.csv"] - 1) for c in rows.values())
    within_validation = sum(max(0, c["test.csv"] - 1) for c in rows.values())
    across_files = sum(min(c["train.csv"], c["test.csv"]) for c in rows.values())
    confirmed = sum(sum(c.values()) - 1 for c in rows.values() if sum(c.values()) > 1)
    return {"status": "exact", "rule": "equal parsed 14-column row tuples; hash candidates confirmed by full tuple",
            "within_train": within_train, "within_validation": within_validation,
            "across_files": across_files, "confirmed_extra_rows": confirmed,
            "candidate_hash_repeats": digest_repeats,
            "hash_collisions": digest_repeats - confirmed}


def sorted_sequence_quality(args, raw):
    if not args.sequence_sort:
        return {"status": "skipped", "reason": "--sequence-sort not enabled"}
    print("External sorting route-second keys for ties and event gaps ...", flush=True)
    rng = random.Random(args.seed + 1)
    gaps = Reservoir(args.reservoir_size, rng)
    groups = Counter()
    groups_by_route = Counter()
    tied_events_by_route = Counter()
    previous_route = None
    previous_time = None
    group_size = 0
    for line in sorted_lines(raw["sequence_path"], args):
        route_s, time_s = line.rstrip("\n").split("\t")
        route = int(route_s)
        timestamp = datetime.fromisoformat(time_s)
        if route == previous_route and timestamp == previous_time:
            group_size += 1
            gaps.add(0)
            continue
        if group_size > 1:
            groups[group_size] += 1
            groups_by_route[previous_route] += 1
            tied_events_by_route[previous_route] += group_size
        if route == previous_route and previous_time is not None:
            gaps.add((timestamp - previous_time).total_seconds())
        previous_route, previous_time, group_size = route, timestamp, 1
    if group_size > 1:
        groups[group_size] += 1
        groups_by_route[previous_route] += 1
        tied_events_by_route[previous_route] += group_size
    return {"status": "exact ties; approximate gap quantiles",
            "same_second_group_size_distribution": {str(k): v for k, v in sorted(groups.items())},
            "tied_groups_by_route": dict(groups_by_route),
            "events_in_tied_groups_by_route": dict(tied_events_by_route),
            "inter_event_gap_seconds": gaps.summary()}


def assemble_report(args, raw, labels, submission, grid, series, availability,
                    padding, duplicates, sequence, started, output):
    files = [args.raw_train, args.raw_validation, args.labels_train,
             args.labels_validation, args.submission]
    file_meta = [{"path": str(path), "bytes": path.stat().st_size}
                 for path in files]
    raw_routes = set(raw["raw_routes"])
    label_routes = set(labels["route_bounds"])
    submission_routes = set(submission["routes"])
    route_dates = {
        "raw": {str(r): [a.isoformat() if a else None, b.isoformat() if b else None]
                for r, (a, b) in raw["raw_routes"].items()},
        "labels": {str(r): [a.isoformat() if a else None, b.isoformat() if b else None]
                   for r, (a, b) in labels["route_bounds"].items()},
    }
    months = sorted({m for _, _, m in raw["raw_route_month"]}
                    | {m for m, _ in raw["missing_month"]}
                    | {m for m, _ in raw["invalid_month"]})
    write_csv(output / "missing_invalid_by_month.csv",
              ["month", "column", "missing", "invalid"],
              [(m, f, raw["missing_month"][(m, f)], raw["invalid_month"][(m, f)])
               for m in months for f in RAW_COLUMNS])
    write_csv(output / "raw_counts_by_route_month.csv",
              ["period", "route", "month", "raw_events"],
              [(p, r, m, n) for (p, r, m), n in sorted(raw["raw_route_month"].items())])
    write_csv(output / "label_rows_by_route_month.csv",
              ["period", "route", "month", "label_rows"],
              [(p, r, m, n) for (p, r, m), n in sorted(labels["rows_by_route_month"].items())])
    write_csv(output / "reconciliation_by_route_month_hour.csv",
              ["period", "route", "month", "hour", "hours", "raw_success",
               "label_boardings", "signed_difference", "absolute_difference", "exact_match_hours"],
              [(p, r, m, h, c["hours"], c["raw_success"], c["label_boardings"],
                c["signed_difference"], c["absolute_difference"], c["exact_match_hours"])
               for (p, r, m, h), c in sorted(grid["recon"].items())])
    write_csv(output / "absent_by_hour_weekday.csv",
              ["period", "route", "hour", "weekday_monday0", "absent_hours"],
              [(p, r, h, d, n) for (p, r, h, d), n in sorted(grid["absent_hour_weekday"].items())])
    write_csv(output / "entire_days_without_labels.csv",
              ["period", "route", "date"],
              [(p, r, day.isoformat()) for p, r, day in grid["entire_days"]])
    write_csv(output / "entire_weeks_without_labels.csv",
              ["period", "route", "week_start_monday"],
              [(p, r, day.isoformat()) for p, r, day in grid["entire_weeks"]])
    categories = {}
    for p in ("train", "validation"):
        categories[p] = {}
        total_rows = raw["raw_period"][p]
        for field in CATEGORICAL:
            counter = raw["cat"][p][field]
            profile = category_report(counter, total_rows)
            profile["distinct_by_month"] = {
                month: len(values) for (pp, month, ff), values in raw["cat_month"].items()
                if pp == p and ff == field}
            profile["distinct_by_route"] = {
                str(route): len(values) for (pp, route, ff), values in raw["cat_route"].items()
                if pp == p and ff == field}
            profile["categories_on_multiple_routes"] = sum(
                len(routes) > 1 for (pp, ff, _), routes in raw["cat_routes"].items()
                if pp == p and ff == field)
            normalized = defaultdict(list)
            for value in counter:
                key = re.sub(r"\s*\.\s*", ".", value.strip()).casefold()
                normalized[key].append(value)
            variants = [sorted(values, key=lambda v: -counter[v]) for values in normalized.values()
                        if len(values) > 1]
            variants.sort(key=lambda group: -sum(counter[v] for v in group))
            profile["whitespace_case_variant_groups_top20"] = [
                [{"value": v, "count": counter[v]} for v in group] for group in variants[:20]]
            if p == "validation":
                unseen = sum(n for value, n in counter.items()
                             if value not in raw["cat"]["train"][field])
                profile["unseen_in_train"] = {"categories": sum(
                    value not in raw["cat"]["train"][field] for value in counter),
                    "rows": unseen,
                    "pct_validation_nonmissing": 100 * unseen / profile["nonmissing_rows"]
                    if profile["nonmissing_rows"] else None}
            categories[p][field] = profile
    for field in CATEGORICAL:
        seen = set()
        growth = []
        for month in sorted({m for p, m, f in raw["cat_month"] if p == "train" and f == field}):
            values = raw["cat_month"][("train", month, field)]
            growth.append({"month": month, "new_categories": len(values - seen),
                           "cumulative_categories": len(seen | values)})
            seen |= values
        categories["train"][field]["vocabulary_growth"] = growth
    coverage_summary = {}
    for p in ("train", "validation"):
        expected = sum(c["expected_hours"] for (pp, _, _), c in grid["coverage"].items() if pp == p)
        observed = sum(c["observed_label_hours"] for (pp, _, _), c in grid["coverage"].items() if pp == p)
        coverage_summary[p] = {"expected_hours": expected, "observed_label_hours": observed,
                               "absent_label_hours": expected - observed,
                               "coverage_pct": 100 * observed / expected if expected else None,
                               "explicit_zero_labels": sum(c["explicit_zero_labels"] for (pp, _, _), c
                                                           in grid["coverage"].items() if pp == p)}
    reconciliation = {}
    for p in ("train", "validation"):
        rows = [c for (pp, _, _, _), c in grid["recon"].items() if pp == p]
        hours = sum(c["hours"] for c in rows)
        successes = sum(c["raw_success"] for c in rows)
        boardings = sum(c["label_boardings"] for c in rows)
        absolute = sum(c["absolute_difference"] for c in rows)
        matches = sum(c["exact_match_hours"] for c in rows)
        nonzero = grid["recon_nonzero"][(p, "hours")]
        nonzero_matches = grid["recon_nonzero"][(p, "exact_matches")]
        reconciliation[p] = {"status": "exact" if not labels["conflicts"] else "conflicted keys excluded",
                             "compared_hours": hours, "raw_success": successes,
                             "label_boardings": boardings,
                             "signed_difference": successes - boardings,
                             "absolute_difference_total": absolute,
                             "absolute_difference_distribution": describe(
                                 grid["recon_abs_by_period"][p]),
                             "exact_match_pct_full_grid": 100 * matches / hours if hours else None,
                             "exact_match_pct_nonzero_union": 100 * nonzero_matches / nonzero
                             if nonzero else None,
                             "nonzero_union_hours": nonzero,
                             "absolute_difference_over_label_sum": absolute / boardings
                             if boardings else None}
    # The exact distribution includes zero-difference hours as well as mismatches.
    all_abs = grid["recon_abs"]
    reconciliation["combined_absolute_difference_distribution"] = describe(all_abs)
    timestamp_quality = {
        "timezone_convention": {"value": None, "reason": "not documented in dataset README"},
        "timestamp_precision": dict(raw["timestamp_precision"]),
        "original_order_reversals": {f"{f}/{r}": n for (f, r), n in raw["reversals"].items()},
        "input_minus_event_seconds": raw["delay"]["input_minus_event_seconds"].summary(),
        "begin_minus_event_seconds": raw["delay"]["begin_minus_event_seconds"].summary(),
        "begin_minus_event_by_validation_result": {
            k: v.summary() for k, v in raw["delay_by_result"].items()},
        "delay_flags": dict(raw["delay_counts"]),
        "sorted_sequence": sequence,
    }
    report = {
        "metadata": {"files_scanned": file_meta, "source_delimiter": ";", "encoding": "UTF-8",
                     "raw_scan_scope": "first_N_rows_per_file" if args.max_raw_rows_per_file else "full_files",
                     "raw_scan_is_representative": False if args.max_raw_rows_per_file else None,
                     "raw_bytes_read_by_file": raw["file_scanned_bytes"],
                     "raw_row_limit_hit_by_file": raw["row_limit_hit"],
                     "python_version": sys.version.split()[0], "platform": platform.platform(),
                     "sort_version": subprocess.run(["sort", "--version"], capture_output=True,
                                                    text=True, check=False).stdout.splitlines()[0],
                     "runtime_seconds": time.monotonic() - started, "random_seed": args.seed,
                     "available_ram_bytes_at_start": args.available_ram,
                     "free_disk_bytes_at_start": args.free_disk,
                     "configuration": {"train": [args.train_start.isoformat(), args.validation_start.isoformat()],
                                       "validation": [args.validation_start.isoformat(), args.validation_end.isoformat()],
                                       "submission": [args.validation_end.isoformat(), args.forecast_end.isoformat()],
                                       "reservoir_size": args.reservoir_size,
                                       "sort_buffer": args.sort_buffer,
                                       "max_raw_rows_per_file": args.max_raw_rows_per_file,
                                       "temp_dir": str(args.temp_dir),
                                       "exact_duplicates": args.exact_duplicates,
                                       "sequence_sort": args.sequence_sort}},
        "coverage": {"raw_rows_by_file": dict(raw["row_counts"]),
                     "raw_rows_by_event_time_period": dict(raw["raw_period"]),
                     "label_rows_by_file": dict(labels["row_counts"]),
                     "raw_file_event_time_bounds": {f: [x.isoformat() if x else None for x in b]
                                                    for f, b in raw["file_bounds"].items()},
                     "raw_event_time_outside_named_file_period": {
                         f"{f}/{p}": n for (f, p), n in raw["out_of_file_period"].items()},
                     "timestamp_bounds": {f: [x.isoformat() if x else None for x in b]
                                          for f, b in raw["timestamp_bounds"].items()},
                     "route_first_last": route_dates,
                     "routes": {"raw": sorted(raw_routes), "labels": sorted(label_routes),
                                "submission": sorted(submission_routes),
                                "raw_only": sorted(raw_routes-label_routes),
                                "label_only": sorted(label_routes-raw_routes),
                                "submission_without_labels": sorted(submission_routes-label_routes)},
                     "submission_template": submission},
        "integrity": {"raw_schema_by_file": raw["file_schemas"],
                      "label_schema_by_file": labels["schemas"],
                      "raw_invalidity_rules": {
                          "tran_date_time,begin_date_time,input_date_time": "nonempty value must parse as ISO datetime",
                          "ngpt_route": "nonempty first token must parse as integer route",
                          "validation_result": "nonempty value must contain decimal digits",
                          "other_columns": "no domain validity rule supplied; blanks counted, invalid count zero does not certify semantics"},
                      "raw_parse_failures": {f"{f}/{reason}": n for (f, reason), n in raw["bad_rows"].items()},
                      "label_invalid": {f"{f}/{reason}": n for (f, reason), n in labels["invalid"].items()},
                      "label_duplicate_keys": dict(labels["duplicates"]),
                      "label_conflicting_examples": labels["conflicts"][:20],
                      "exact_raw_duplicate_rows": duplicates,
                      "missing_invalid_by_month": "missing_invalid_by_month.csv",
                      "file_time_range_overlap": "see coverage.raw_file_event_time_bounds; overlap does not imply duplicate records"},
        "hourly_grid": {"status": "exact; absent label rows filled with zero and presence retained",
                        "summary_by_period": coverage_summary,
                        "route_month_detail": "grid_coverage.csv",
                        "raw_label_presence_quadrants": {f"{p}/raw={int(raw_present)}/label={int(label_present)}": n
                                                         for (p, raw_present, label_present), n in grid["quadrants"].items()},
                        "absent_run_length_hours": describe([x[4] for x in grid["absent_runs"]]),
                        "longest_absent_runs": "longest_absent_runs.csv",
                        "entire_days_without_labels": len(grid["entire_days"]),
                        "entire_weeks_without_labels": len(grid["entire_weeks"]),
                        "first_last_positive_hour": "first_last_positive_hour.csv",
                        "absent_by_hour_weekday": "absent_by_hour_weekday.csv",
                        "presence_inconsistency_examples": "raw_label_inconsistency_examples.csv",
                        "route_5": {"label_rows": sum(r == 5 for r, _ in labels["values"]),
                                    "raw_events": sum(n for (r, _), n in raw["raw_hour"].items() if r == 5),
                                    "submission_required": 5 in submission_routes}},
        "reconciliation": {"by_period": reconciliation,
                           "by_route_month_hour": "reconciliation_by_route_month_hour.csv",
                           "largest_50_mismatches": "largest_mismatches.csv",
                           "duplicate_policy": "original label rows checked first; conflicting keys excluded, not silently resolved",
                           "exact_deduplicated_view": {"status": "not_computed" if duplicates.get("confirmed_extra_rows", 0)
                                                       else "same as original; no exact duplicates found"}},
        "event_lengths": series["event_lengths"],
        "padding_simulation": padding,
        "categorical_profiles": categories,
        "timestamp_quality": timestamp_quality,
        "target_statistics": series["target_statistics"],
        "sample_availability": availability["report"],
        "recommendations": [
            {"decision": "keep_candidate", "fields": ["tran_date_time", "ngpt_route",
                "validation_result", "device_no", "tran_type_id", "place_id", "good_type",
                "pass_route", "bus_exit_no", "garage_number"],
             "evidence": "event time defines target alignment; categorical distributions and missingness are profiled here"},
            {"decision": "defer", "fields": ["begin_date_time", "input_date_time"],
             "evidence": "delay distributions and load-time quality are reported under timestamp_quality"},
            {"decision": "exclude_from_vocabularies", "fields": ["tran_no", "crd_hashcode"],
             "evidence": "transaction number and card hash are identifiers, not reusable category vocabularies"},
            {"decision": "preserve_label_presence", "fields": ["boardings"],
             "evidence": "absent hourly label rows are zero-filled by convention; original presence remains available for diagnostics"},
        ],
        "limitations": [],
        "detail_files": {},
    }
    for path in sorted(output.glob("*.csv")):
        with open(path, encoding="utf-8", newline="") as handle:
            schema = next(csv.reader(handle))
        report["detail_files"][path.name] = {"relative_path": path.name, "schema": schema}
    if args.max_raw_rows_per_file:
        report["integrity"]["exact_raw_duplicate_rows"]["scope"] = "scanned raw prefixes only"
        report["timestamp_quality"]["sorted_sequence"]["scope"] = "scanned raw prefixes only"
        for item in report["reconciliation"]["by_period"].values():
            if isinstance(item, dict) and "compared_hours" in item:
                item["status"] = "partial_raw_prefix_vs_full_labels; not a valid reconciliation rate"
        report["sample_availability"]["windows_with_any_zero_raw_event_day_in_history"] = None
        report["sample_availability"]["suspicious_gap_rule"] = (
            "unavailable: raw prefix scan cannot distinguish empty history days from unscanned days")
        report["limitations"].append(
            "Raw diagnostics use the first N rows of each file, not a representative sample. "
            "Raw-to-label reconciliation, raw route coverage, event lengths and padding estimates "
            "do not describe the full dataset. Label-grid and structural sample counts remain full-data results.")
    return report


def validate_report(report):
    checks = {}
    coverage = report["coverage"]
    checks["raw_file_row_total"] = (sum(coverage["raw_rows_by_file"].values()) >=
                                     sum(coverage["raw_rows_by_event_time_period"].values()))
    checks["grid_hours"] = all(
        p["expected_hours"] == p["observed_label_hours"] + p["absent_label_hours"]
        for p in report["hourly_grid"]["summary_by_period"].values())
    checks["reconciliation_totals"] = all(
        p["signed_difference"] == p["raw_success"] - p["label_boardings"]
        for p in report["reconciliation"]["by_period"].values()
        if isinstance(p, dict) and "signed_difference" in p)
    checks["categorical_totals"] = all(
        c["missing_count"] + c["nonmissing_rows"] ==
        coverage["raw_rows_by_event_time_period"].get(period_name, 0)
        for period_name, fields in report["categorical_profiles"].items()
        for c in fields.values())
    lead_total = sum(report["sample_availability"]["sample_identities_by_lead"].values())
    checks["sample_identity_total"] = lead_total == report["sample_availability"]["total_sample_identities"]
    report["metadata"]["consistency_checks"] = checks
    if not all(checks.values()):
        raise ValueError(f"Internal consistency checks failed: {checks}")


def write_summary(report, output):
    coverage = report["coverage"]
    grid = report["hourly_grid"]
    recon = report["reconciliation"]["by_period"]
    sample = report["sample_availability"]
    route5 = grid["route_5"]
    max_events = report["event_lengths"]["nonempty_route_hours"]["max"]
    p95_events = report["event_lengths"]["nonempty_route_hours"]["p95"]
    partial = report["metadata"]["raw_scan_scope"] != "full_files"
    raw_routes_line = (
        f"- Raw routes {'in scanned prefixes' if partial else 'in full files'}: "
        f"{coverage['routes']['raw']}; label routes: {coverage['routes']['labels']}; "
        f"submission routes: {coverage['routes']['submission']}.")
    route5_line = (
        f"- Route 5 has {route5['raw_events']:,} raw events "
        f"{'in scanned prefixes' if partial else 'in the full raw files'} and "
        f"{route5['label_rows']:,} label rows; submission requires it: "
        f"{route5['submission_required']}.")
    recon_line = (
        "- Raw-to-label reconciliation requires a full raw scan; prefix counts "
        "are intentionally not interpreted."
        if partial else
        f"- Raw-to-label signed difference: train {recon['train']['signed_difference']:,}; "
        f"validation {recon['validation']['signed_difference']:,}. "
        f"Exact-match rate on full grid: {recon['train']['exact_match_pct_full_grid']:.2f}% "
        f"and {recon['validation']['exact_match_pct_full_grid']:.2f}%.")
    event_line = (
        f"- Events per nonempty route-hour in scanned prefixes: p95 {p95_events:.0f}, "
        f"maximum {max_events:,}; these do not size full-data batches."
        if partial and p95_events is not None else
        f"- Events per nonempty route-hour: p95 {p95_events:.0f}, maximum {max_events:,}. "
        "See `largest_event_hours.csv` and padding estimates before sizing event batches."
        if p95_events is not None else
        "- No nonempty raw route-hour was available for event-size estimates.")
    scope_line = (
        f"Scanned {sum(coverage['raw_rows_by_file'].values()):,} raw rows and "
        f"{sum(coverage['label_rows_by_file'].values()):,} label rows. "
        "Periods follow `tran_date_time`, not source filenames. "
        f"Runtime: {report['metadata']['runtime_seconds']:.0f} seconds. "
        + ("Raw files were limited to a deterministic first-N-row prefix. " if partial else "")
        + "Numerical grid quantiles are exact; timestamp-delay and sorted-gap "
        "quantiles use fixed-size reservoirs with linear interpolation.")
    validation = sample["fixed_history_checks"]["validation"]
    submission = sample["fixed_history_checks"]["submission"]
    text = [
        "# Tram data profile",
        "",
        "## Scope",
        "",
        scope_line,
        "",
        "## Main findings",
        "",
        raw_routes_line,
        route5_line,
        f"- Hourly grid: train {grid['summary_by_period']['train']['observed_label_hours']:,}/"
        f"{grid['summary_by_period']['train']['expected_hours']:,} observed; validation "
        f"{grid['summary_by_period']['validation']['observed_label_hours']:,}/"
        f"{grid['summary_by_period']['validation']['expected_hours']:,} observed.",
        recon_line,
        event_line,
        f"- Structurally valid training identities: {sample['total_sample_identities']:,}; "
        f"routes with label history: {sample['routes_with_label_history']}.",
        f"- Fixed 504-hour grid at 2025-09-01: "
        f"{sum(v['available_504h_grid'] for v in validation.values())} routes, "
        f"of which {sum(v['hours_with_label_row'] > 0 for v in validation.values())} "
        f"have label history; at 2025-11-01: "
        f"{sum(v['available_504h_grid'] for v in submission.values())} route grids, "
        f"of which {sum(v['hours_with_label_row'] > 0 for v in submission.values())} have label history.",
        "",
        "## Decisions for the data pipeline",
        "",
        "- Keep event time, route, validation result, fare product, device, place, "
        "run, garage number, and the raw pass-route text as candidate transaction fields. "
        "Treat identifier-like fields as categories, not ordered numbers.",
        "- Exclude transaction number and card hash from vocabularies. Defer trip-start and "
        "load timestamps until their quality and availability implications are resolved.",
        "- Preserve raw `pass_route` values. Inspect the variant groups in `profile.json` "
        "before normalizing spaces around dots; never reorder components.",
        "- Use separate missing and unknown tokens for embeddings. Start with dimensions "
        "4 or 8 for small vocabularies; inspect the 95%/99% coverage counts before "
        "setting rare-category thresholds for larger fields.",
        "- Process nonempty hours in chunks rather than padding every hour in a batch "
        "to the largest event count. The event input memory estimates exclude model state.",
        "- Fill absent label hours with zero for the specified target convention and "
        "retain the presence indicator for diagnostics. Entire-day gaps need review "
        "before treating them as ordinary zero demand.",
        "",
        "## Limits and unresolved issues",
        "",
    ]
    for item in report["limitations"]:
        text.append(f"- {item}")
    if not report["limitations"]:
        text.append("- No unresolved profiling check was recorded.")
    text += ["", "Detailed results and methods are in `profile.json`; compact tables are linked under `detail_files`."]
    (output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--raw-validation", type=Path, default=Path("dataset/test.csv"))
    parser.add_argument("--labels-train", type=Path, default=Path("dataset/labels/labels_day_train.csv"))
    parser.add_argument("--labels-validation", type=Path, default=Path("dataset/labels/labels_day_test.csv"))
    parser.add_argument("--submission", type=Path, default=Path("dataset/test_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_profile"))
    parser.add_argument("--temp-dir", type=Path, default=Path("outputs/data_profile/tmp"))
    parser.add_argument("--train-start", type=datetime.fromisoformat, default=datetime(2025, 1, 1))
    parser.add_argument("--validation-start", type=datetime.fromisoformat, default=datetime(2025, 9, 1))
    parser.add_argument("--validation-end", type=datetime.fromisoformat, default=datetime(2025, 11, 1))
    parser.add_argument("--forecast-end", type=datetime.fromisoformat, default=datetime(2026, 1, 1))
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--reservoir-size", type=int, default=100000)
    parser.add_argument("--padding-batches", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=250000,
                        help="Update progress after this many raw rows per file")
    parser.add_argument("--no-progress-bar", action="store_true",
                        help="Use plain periodic progress lines instead of a bar")
    parser.add_argument("--max-raw-rows-per-file", type=int,
                        help="Read only the first N data rows of each raw CSV (quick, nonrepresentative scan)")
    parser.add_argument("--sort-buffer", default="2G")
    parser.add_argument("--memory-limit-gb", type=float, default=12)
    parser.add_argument("--skip-exact-duplicates", dest="exact_duplicates", action="store_false")
    parser.add_argument("--skip-sequence-sort", dest="sequence_sort", action="store_false")
    parser.set_defaults(exact_duplicates=True, sequence_sort=True)
    args = parser.parse_args()
    if args.max_raw_rows_per_file is not None and args.max_raw_rows_per_file < 1:
        parser.error("--max-raw-rows-per-file must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    args.available_ram = int(next((line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                   if line.startswith("MemAvailable:")), "0")) * 1024
    args.free_disk = shutil.disk_usage(Path.cwd()).free
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    if args.memory_limit_gb:
        try:
            import resource
            limit = int(args.memory_limit_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError) as error:
            print(f"Memory limit could not be applied: {error}", file=sys.stderr, flush=True)
    for path in (args.raw_train, args.raw_validation, args.labels_train,
                 args.labels_validation, args.submission):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = scan_labels(args)
    submission = scan_submission(args)
    print("Label and submission scans complete.", flush=True)
    raw = scan_raw(args, args.output_dir)
    print("Raw scan complete; building compact-grid analyses ...", flush=True)
    grid = make_grid(args, raw, labels, submission, args.output_dir)
    series = analyze_series(args, grid, args.output_dir)
    availability = sample_availability(args, grid, args.output_dir)
    padding = padding_simulation(args, grid, availability)
    try:
        duplicates = check_exact_duplicates(args, raw)
    except (OSError, RuntimeError, MemoryError) as error:
        duplicates = {"status": "failed", "reason": str(error)}
        print(f"Exact duplicate check failed; continuing: {error}", file=sys.stderr, flush=True)
    try:
        sequence = sorted_sequence_quality(args, raw)
    except (OSError, RuntimeError, MemoryError) as error:
        sequence = {"status": "failed", "reason": str(error)}
        print(f"Sorted sequence check failed; continuing: {error}", file=sys.stderr, flush=True)
    report = assemble_report(args, raw, labels, submission, grid, series,
                             availability, padding, duplicates, sequence, started, args.output_dir)
    if duplicates.get("confirmed_extra_rows", 0):
        report["limitations"].append("Exact duplicate rows exist; an exact-deduplicated reconciliation view was not computed.")
    if labels["conflicts"]:
        report["limitations"].append("Conflicting label keys were excluded from reconciliation and need a source decision.")
    if not args.exact_duplicates:
        report["limitations"].append("Exact raw-row duplicates were skipped.")
    if not args.sequence_sort:
        report["limitations"].append("Same-second groups and sorted inter-event gaps were skipped.")
    if duplicates["status"] == "failed":
        report["limitations"].append("Exact raw-row duplicate check failed: " + duplicates["reason"])
    if sequence["status"] == "failed":
        report["limitations"].append("Sorted event-sequence check failed: " + sequence["reason"])
    report["limitations"].append("Timestamp timezone is undocumented; no localization or conversion was applied.")
    report["limitations"].append("Delay and inter-event-gap quantiles use reservoir samples; counts and grid statistics are exact.")
    validate_report(report)
    with open(args.output_dir / "profile.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    write_summary(report, args.output_dir)
    print(f"Wrote {args.output_dir / 'profile.json'} and summary.md", flush=True)


if __name__ == "__main__":
    main()
