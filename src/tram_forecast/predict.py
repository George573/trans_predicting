"""Validated submission export, with route 5 outside the neural model."""

import csv
import hashlib
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from .checkpoint import load_model
from .config import ROUTES
from .evaluate import fixed_forecast
from .io import write_json
from .storage import Store


def submission_keys(start=date(2025, 11, 1), end=date(2026, 1, 1)):
    return {
        (r, start + timedelta(days=d), h)
        for r in (*ROUTES, 5)
        for d in range((end - start).days)
        for h in range(24)
    }


def write_submission(
    template, output, predictions, start=date(2025, 11, 1), end=date(2026, 1, 1)
):
    expected = submission_keys(start, end)
    if set(predictions) != expected:
        raise ValueError("prediction keys must cover the complete submission grid")
    if any(not math.isfinite(float(v)) or v < 0 for v in predictions.values()):
        raise ValueError("predictions must be finite/nonnegative")
    ordered = []
    seen = set()
    with open(template, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames != ["route", "date", "hour", "prediction"]:
            raise ValueError("unexpected template columns/order")
        for row in reader:
            key = (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
            if key not in expected or key in seen:
                raise ValueError("extra or duplicate template key")
            seen.add(key)
            ordered.append(key)
    if seen != expected:
        raise ValueError("missing template keys")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".submission-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["route", "date", "hour", "prediction"])
            for r, d, h in ordered:
                writer.writerow(
                    [r, d.isoformat(), h, format(float(predictions[r, d, h]), ".10g")]
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, output)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return len(ordered)


def predict(checkpoint, artifact, template, output, device="cpu"):
    from .checkpoint import read_checkpoint

    payload = read_checkpoint(checkpoint)
    store = Store(artifact, events=payload["model_kind"] == "full")
    if (
        store.metadata["regime"] != "final"
        or store.end != date(2025, 11, 1)
        or set(store.routes) != set(ROUTES)
    ):
        raise ValueError(
            "submission requires all neural routes in the final October artifact"
        )
    model, _ = load_model(checkpoint, store, device)
    values = fixed_forecast(model, store, 61)
    predictions = {}
    for i, route in enumerate(store.routes):
        for d in range(61):
            for h in range(24):
                predictions[route, store.end + timedelta(days=d), h] = float(
                    values[i, d, h]
                )
    for d in range(61):
        for h in range(24):
            predictions[5, store.end + timedelta(days=d), h] = 0.0
    rows = write_submission(template, output, predictions)
    sha = hashlib.sha256()
    with open(checkpoint, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    write_json(
        str(output) + ".provenance.json",
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": sha.hexdigest(),
            "artifact_hash": store.contract_hash,
            "settings": payload["settings"],
            "rows": rows,
            "cutoff": str(store.end),
            "route5_fallback": 0,
        },
    )
    return Path(output)
