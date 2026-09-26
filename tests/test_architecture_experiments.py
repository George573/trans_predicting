"""Small boarding histories, pooling and checkpoint contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tram_forecast.checkpoint import load_model, save_checkpoint
from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.evaluate import fixed_forecast
from tram_forecast.experiments import run_suite, variants
from tram_forecast.model import ForecastNetwork
from tram_forecast.settings import Settings


def small():
    return Settings.load('configs/boarding_week_small.json').model


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('pool', ['max', 'avg'])
def test_week_history_forward_backward_checkpoint(prepared, tmp_path, depth, pool):
    settings, artifact = prepared
    config = replace(small(), shared_depth=depth, temporal_pool=pool)
    settings = replace(settings, model=config)
    data = ForecastDataset(artifact, 'boarding_only', 7, history_days=7)
    assert data.index[0, 1] == 7
    sample = data[14]  # Cutoff January 22.
    history, request, target = collate_samples([sample])
    assert history['counts'].shape == (1, 1, 168)
    assert set(history) == {'counts', 'calendar'}
    assert sample['identity'].cutoff.isoformat() == '2025-01-22'
    model = ForecastNetwork(1, config, 'boarding_only')
    prediction = model(history, request)
    assert prediction.shape == target.shape
    prediction.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    path = tmp_path / 'small.pt'
    save_checkpoint(path, model, data.store, settings)
    restored, _ = load_model(path, data.store)
    torch.testing.assert_close(restored(history, request), model(history, request))
    assert fixed_forecast(restored, data.store, 7).shape == (1, 7, 24)


def test_suite_preview_and_default_checkpoint_geometry():
    rows = run_suite('configs/experiments/boarding_week.json')
    assert len(rows) == 8 and max(r['parameters'] for r in rows) < 50_000
    assert all(r['history_days'] == 7 for r in rows)
    model = ForecastNetwork(1)
    assert model.head[0].weight.shape == (250, 685)
    assert model.encoded_width == 672


@pytest.mark.parametrize('values', [dict(history_days=0), dict(shared_depth=4),
    dict(temporal_pool='median'), dict(pooled_hours=99), dict(head_width=0)])
def test_invalid_architectures(values):
    with pytest.raises(ValueError):
        replace(small(), **values)


def test_suite_rejects_changed_contexts():
    suite = json.loads(Path('configs/experiments/boarding_week.json').read_text())
    suite['variants'][0]['model']['history_days'] = 21
    with pytest.raises(ValueError, match='share history'):
        variants(suite)


@pytest.mark.parametrize("kind", ["boarding_only"])
def test_suite_orchestration_without_optimization(prepared, tmp_path, monkeypatch, kind):
    settings, artifact = prepared
    suite = {"base": replace(settings, model=small()).to_dict(),
             "variants": [{"name": "baseline"}, {"name": "avg", "model": {"temporal_pool": "avg"}}]}
    suite["model_kind"] = kind
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    calls = []

    def fake_train(settings, artifact, model_kind, output):
        from tram_forecast.io import write_json
        from tram_forecast.storage import Store
        store = Store(artifact)
        model = ForecastNetwork(1, settings.model, model_kind)
        calls.append(model_kind)
        checkpoint = output / "best.pt"
        save_checkpoint(checkpoint, model, store, settings, best_epoch=1)
        write_json(output / "history.json", [{"epoch": 1, "seconds": 2.0}])
        return checkpoint

    monkeypatch.setattr("tram_forecast.experiments.train", fake_train)
    output = tmp_path / "runs"
    rows = run_suite(path, artifact, output, run=True)
    assert calls == [kind, kind]
    assert all(row["training_seconds"] == 2 and row["model_kind"] == kind for row in rows)
    saved = json.loads((output / "comparison.json").read_text())
    assert saved[0]["wape"] <= saved[1]["wape"]
    with pytest.raises(ValueError, match="already exists"):
        run_suite(path, artifact, output, run=True)
    previous = (output / "baseline_seed67" / "best.pt").read_bytes()
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep")
    run_suite(path, artifact, output, run=False, overwrite=True)
    assert not (output / "archive").exists()
    run_suite(path, artifact, output, run=True, overwrite=True)
    archives = list((output / "archive").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "baseline_seed67" / "best.pt").read_bytes() == previous
    assert json.loads((archives[0] / "comparison.json").read_text()) == saved
    assert (output / "baseline_seed67" / "best.pt").is_file()
    assert unrelated.read_text() == "keep"
    assert calls == [kind] * 4


def test_experiment_notebook_compiles():
    notebook = json.loads(Path("notebooks/architecture_experiments.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")


def test_notebook_setup_from_unrelated_kernel_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/architecture_experiments.ipynb').read_text())
    setup = next(c for c in notebook['cells'] if c['id'] == 'setup')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('TRAM_FORECAST_ROOT', raising=False)
    namespace = {}
    exec(compile(''.join(setup['source']), 'setup', 'exec'), namespace)
    assert namespace['ROOT'] == root
    assert namespace['suite']['base']['model']['history_days'] == 7
    assert namespace['find_project_root'](root) == root
    with pytest.raises(FileNotFoundError, match='Sync src/'):
        namespace['find_project_root'](tmp_path)


def test_quick_followup_uses_identical_training_targets(prepared):
    import numpy as np
    _, artifact = prepared
    suite = json.loads(Path('configs/experiments/quick_followup.json').read_text())
    runs = variants(suite)
    assert len(runs) == 4
    indices = []
    for _, settings in runs:
        assert settings.training.epochs == 5 and settings.training.patience == 2
        data = ForecastDataset(artifact, 'boarding_only', 7, settings.model.history_days,
                               settings.training.context_start_days)
        indices.append(data.index)
    for index in indices[1:]:
        np.testing.assert_array_equal(index, indices[0])


def test_boarding_notebook_setup_and_preview(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/boarding_only_experiments.ipynb").read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAM_FORECAST_ROOT", raising=False)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            code = compile("".join(cell["source"]), cell["id"], "exec")
            if cell["id"] in ("setup", "configure", "preview"):
                if cell["id"] == "preview":
                    namespace.update(OUTPUT=tmp_path / "runs", ARTIFACT=tmp_path / "missing")
                exec(code, namespace)
    assert namespace["suite"]["model_kind"] == "boarding_only"
    assert len(namespace["preview"]) == len(namespace["suite"]["variants"])
    assert all(r["model_kind"] == "boarding_only" for r in namespace["preview"])
