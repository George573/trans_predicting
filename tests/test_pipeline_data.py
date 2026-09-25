import csv
import json
from dataclasses import replace
from datetime import date

import pytest

from tram_forecast.config import Config
from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.preprocess import prepare
from tram_forecast.settings import DataSettings, Settings
from tram_forecast.storage import Store


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw.csv"
    labels = tmp_path / "labels.csv"
    with raw.open("w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "ngpt_route",
                "tran_date_time",
                "validation_result",
                "tran_type_id",
                "good_type",
                "place_id",
                "pass_route",
            ]
        )
        writer.writerows(
            [
                ["1 трамвай", "2025-01-21 23:59:00", "1", "52", "fare", "1", ""],
                ["1", "2025-01-01 00:00:00", "1", "52", "fare", "1", "0"],
                ["1", "2025-01-21 23:59:00", "1", "52", "second", "1", "nan"],
                ["1", "2025-02-01 00:00:00", "1", "52", "HELDOUT", "1", "new"],
            ]
        )
    labels.write_text(
        "route;date;hour;boardings\n1;2025-01-01;0;2\n1;2025-01-22;0;3\n1;2025-02-01;0;10\n"
    )
    data = DataSettings(
        raw_paths=(str(raw),),
        label_paths=(str(labels),),
        routes=(1,),
        start="2025-01-01",
        validation_cutoff="2025-02-01",
        final_cutoff="2025-03-01",
        forecast_end="2025-04-01",
        output_root=str(tmp_path / "prepared"),
        temp_dir=str(tmp_path / "tmp"),
        min_free_disk_bytes=0,
        fetch_rows=2,
    )
    settings = Settings(model=Config(min_frequency=1), data=data)
    path = prepare(settings, "validation")
    return settings, path


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
