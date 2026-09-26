import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.preprocess import prepare
from tram_forecast.storage import Store


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
    changed = replace(settings, model=replace(settings.model, history_days=7))
    assert prepare(changed, 'validation') == path
    with pytest.raises(ValueError):
        store.target(1, date(2025, 2, 1))
    labels = Path(settings.data.label_paths[0])
    labels.write_text(labels.read_text().replace(';10\n', ';100\n'))
    with pytest.raises(ValueError, match='incompatible'):
        prepare(settings, 'validation')


def test_artifact_corruption_rejected(prepared):
    _, path = prepared
    scaling = json.loads((path / 'scaling.json').read_text())
    scaling['scale'] = 12
    (path / 'scaling.json').write_text(json.dumps(scaling))
    with pytest.raises(ValueError, match='scaling'):
        Store(path)


def test_final_history_includes_validation_observations(prepared):
    settings, path = prepared
    validation = Store(path)
    final = Store(prepare(settings, 'final'))
    assert validation.counts.sum() == 5
    assert final.counts.sum() == 15
    with pytest.raises(ValueError, match='no observed'):
        final.evaluation_targets()


def test_preparation_accepts_mixed_csv_line_endings(prepared, tmp_path):
    settings, _ = prepared
    labels = Path(settings.data.label_paths[0])
    lines = labels.read_bytes().splitlines()
    labels.write_bytes(b'\n'.join(lines[:2]) + b'\n' + b'\r\n'.join(lines[2:]) + b'\r\n')
    mixed = Store(prepare(settings, 'validation', tmp_path / 'mixed'))
    assert mixed.counts.sum() == 5


def test_failed_preparation_cleanup(prepared, tmp_path):
    settings, _ = prepared
    labels = Path(settings.data.label_paths[0])
    labels.write_text(labels.read_text().replace('2025-01-01', 'invalid date'))
    destination = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        prepare(settings, 'validation', destination)
    assert not destination.exists() and not list(tmp_path.glob('.invalid-*'))


def test_invalid_boarding_array_rejected(prepared):
    _, path = prepared
    counts = np.load(path / 'boardings.npy')
    counts[0, 0] = np.nan
    np.save(path / 'boardings.npy', counts)
    with pytest.raises(ValueError, match='invalid artifact boarding'):
        Store(path)
