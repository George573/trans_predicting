"""Export the selected trained hybrid for the unseen November–December period."""

import csv
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.data import ROUTES, Boardings
from tram_forecast.predict import fixed_forecast
from tram_forecast.train import load_model

ROOT = Path(__file__).resolve().parents[2]


def export_submission():
    cutoff, days = date(2025, 11, 1), 61
    output = ROOT / "outputs"
    selection = json.loads(
        (output / "autoregression_selected/submission_metadata.json").read_text()
    )
    recipe = selection["exports"]["submission.csv"]["recipe"]
    if recipe != {"patch_direct": 0.5, "cnn_recursive": 0.5}:
        raise ValueError("Selected recipe changed; review the export configuration")
    source_paths = [
        ROOT / "dataset/labels/labels_day_train.csv",
        ROOT / "dataset/labels/labels_day_test.csv",
    ]
    events = ROOT / "dataset/parking_events_hourly.csv"
    data = [
        Boardings.load(path, start=date(2025, 1, 1), end=cutoff, events_path=events)
        for path in source_paths
    ]
    boardings = data[0]
    boardings.counts += data[1].counts
    boardings.scale = max(1.0, float(boardings.counts.mean()))
    # Both models were already refitted on all available January–October labels.
    paths = {name: ROOT / selection["checkpoints"][name] for name in recipe}
    patch = load_model(paths["patch_direct"])
    cnn = load_model(paths["cnn_recursive"])
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        direct = fixed_forecast(patch, boardings, cutoff, days, patch.history_days)
        recursive = recursive_forecast(cnn, boardings, cutoff, days, block_days=1)
    finally:
        torch.set_num_threads(old_threads)
    forecast = recipe["patch_direct"] * direct + recipe["cnn_recursive"] * recursive
    if (
        forecast.shape != (len(ROUTES), days, 24)
        or not np.isfinite(forecast).all()
        or (forecast < 0).any()
    ):
        raise ValueError("Invalid generated forecasts")
    # Preserve the official template's exact key order, as in the training notebook.
    with (ROOT / "dataset/test_submission.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    keys = [
        (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
        for row in rows
    ]
    expected = {
        (r, cutoff + timedelta(days=d), h)
        for r in (*ROUTES, 5)
        for d in range(days)
        for h in range(24)
    }
    if len(keys) != 14640 or set(keys) != expected:
        raise ValueError(
            "Submission template must contain the complete November–December grid"
        )
    predictions = []
    for route, day, hour in keys:
        value = (
            0.0
            if route == 5
            else float(forecast[ROUTES.index(route), (day - cutoff).days, hour])
        )
        predictions.append((route, day.isoformat(), hour, f"{value:.6f}"))
    exports = [
        ("submission_nov_dec.csv", "prediction"),
        ("predicted_labels_nov_dec.csv", "boardings"),
    ]
    for filename, target in exports:
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["route", "date", "hour", target])
            writer.writerows(predictions)
        with (output / filename).open(newline="") as handle:
            saved = list(csv.DictReader(handle, delimiter=";"))
        assert len(saved) == len(expected)
        assert [
            (int(r["route"]), date.fromisoformat(r["date"]), int(r["hour"]))
            for r in saved
        ] == keys
        np.testing.assert_allclose(
            [float(r[target]) for r in saved],
            [float(r[3]) for r in predictions],
            rtol=0,
            atol=0,
        )
    fingerprints = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*source_paths, events, *paths.values()]
    }
    metadata = dict(
        cutoff=str(cutoff),
        forecast_end="2025-12-31",
        days=days,
        rows=14640,
        observed_labels_end="2025-10-31",
        recipe=recipe,
        checkpoints={name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        source_sha256=fingerprints,
        files={name: target for name, target in exports},
        note="Both CSVs contain predictions, not observed labels. No November–December targets were used.",
    )
    (output / "submission_nov_dec_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return output / exports[0][0], output / exports[1][0]


if __name__ == "__main__":
    export_submission()
