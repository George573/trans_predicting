from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from tram_forecast.data import ForecastDataset, Store, collate_samples, prepare


def test_preparation_dataset_and_reuse(prepared):
    settings, path = prepared
    store = Store(path)
    assert set(p.name for p in path.iterdir()) == {
        'boardings.npy', 'label_present.npy', 'scaling.json', 'metadata.json',
        'evaluation.npy', 'evaluation_present.npy',
    }
    counts = store.history(1, date(2025, 1, 22))
    assert counts.shape == (504,) and counts.sum() == 2
    assert store.evaluation_targets()[0, 0] == 10
    dataset = ForecastDataset(path)
    history, request, targets = collate_samples([dataset[0]])
    assert targets.shape == (7, 24) and request['lead'].tolist() == list(range(1, 8))
    assert request['context_indices'].tolist() == [0] * 7
    assert set(history) == {'counts', 'calendar'}
    assert prepare(settings, 'validation') == path
    # Model geometry does not change the reusable label artifact.
    changed = replace(settings, training=replace(settings.training, forecast_days=14))
    assert prepare(changed, 'validation') == path
    with pytest.raises(ValueError):
        store.target(1, date(2025, 2, 1))
    labels = Path(settings.data.label_paths[0])
    labels.write_text(labels.read_text().replace(';10\n', ';100\n'))
    with pytest.raises(ValueError, match='incompatible'):
        prepare(settings, 'validation')


def test_failed_preparation_cleanup(prepared, tmp_path):
    settings, _ = prepared
    labels = Path(settings.data.label_paths[0])
    labels.write_text(labels.read_text().replace('2025-01-01', 'invalid date'))
    destination = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        prepare(settings, 'validation', destination)
    assert not destination.exists() and not list(tmp_path.glob('.invalid-*'))
