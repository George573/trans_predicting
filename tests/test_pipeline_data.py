import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.preprocess import prepare
from tram_forecast.storage import Store


def test_preparation_dataset_and_reuse(prepared):
    settings, path = prepared
    store = Store(path)
    assert len(store.events) == 3
    assert "HELDOUT" not in store.vocab["good_type"]
    assert store.events[0, 4] != 1 and store.events[1, 4] == 1
    counts, hours = store.history(1, date(2025, 1, 22))
    assert counts.shape == (504,) and len(hours[-1]) == 2
    assert store.evaluation_targets()[0, 0] == 10
    dataset = ForecastDataset(path)
    history, request, targets = collate_samples([dataset[0]])
    assert targets.shape == (1, 24) and request["lead"].item() == 1
    assert len(history["hours"]) == 504
    assert prepare(settings, "validation") == path
    baseline = ForecastDataset(path, "boarding_only")
    assert baseline.store.events is None and baseline[0]["hours"] is None
    with pytest.raises(ValueError):
        store.target(1, date(2025, 2, 1))
    changed = replace(settings, model=replace(settings.model, min_frequency=2))
    with pytest.raises(ValueError, match="incompatible"):
        prepare(changed, "validation")


def test_artifact_corruption_rejected(prepared):
    _, path = prepared
    scaling = json.loads((path / "scaling.json").read_text())
    scaling["scale"] = 12
    (path / "scaling.json").write_text(json.dumps(scaling))
    with pytest.raises(ValueError, match="scaling"):
        Store(path)


def test_final_vocab_sees_new_observations(prepared):
    settings, _ = prepared
    final = Store(prepare(settings, "final"))
    assert "HELDOUT" in final.vocab["good_type"]
    with pytest.raises(ValueError, match="no observed"):
        final.evaluation_targets()


def test_zero_event_artifact_and_failed_preparation_cleanup(prepared, tmp_path):
    settings, _ = prepared
    raw = Path(settings.data.raw_paths[0])
    original = raw.read_text()
    raw.write_text(original.splitlines()[0] + "\n")
    empty_path = prepare(settings, "validation", tmp_path / "empty")
    empty = Store(empty_path)
    assert empty.events.shape == (0, 5) and empty.offsets.max() == 0
    raw.write_text(original.replace("2025-01-01 00:00:00", "invalid timestamp"))
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="timestamp"):
        prepare(settings, "validation", destination)
    assert not destination.exists() and not list(tmp_path.glob(".invalid-*"))
