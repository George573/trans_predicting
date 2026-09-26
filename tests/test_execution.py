"""Core forecasting workflows using tiny synthetic boarding data."""

import csv
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tram_forecast.checkpoint import load_model, restore_rng, rng_state, save_checkpoint, seed_all
from tram_forecast.data import ForecastDataset, Store, collate_samples, prepare
from tram_forecast.evaluate import evaluate_model, fixed_forecast, metrics, weekly_profile
from tram_forecast.train import refit
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import submission_keys, write_submission


def test_aggregated_metrics_and_zero_denominator():
    report = metrics(np.array([1, 99]), np.array([2, 90]))
    assert report["wape"] == 0.1 and report["mae"] == 5
    assert metrics(np.zeros(2), np.ones(2))["wape"] is None
    with pytest.raises(ValueError):
        metrics(np.ones(2), np.ones(3))


def test_baseline_network_and_fixed_evaluation(prepared):
    settings, path = prepared
    store = Store(path)
    model = ForecastNetwork(store.metadata["scale"])
    assert not hasattr(model, "events") and not hasattr(model, "raw")
    dataset = ForecastDataset(path)
    inputs, request, target = collate_samples([dataset[0]])
    assert "hours" not in inputs and model(inputs, request).shape == (7, 24)
    expected = fixed_forecast(model, store, 28)
    report = evaluate_model(model, store)
    assert report["forecast_days"] == 7
    assert report["neural"]["global"]["positions"] == 7 * 24
    assert report["full_grid_with_route5"]["global"]["positions"] == 2 * 7 * 24
    assert evaluate_model(model, store, 28)["neural"]["global"]["positions"] == 28 * 24
    assert weekly_profile(store, 28).shape == (1, 28, 24)
    # Held-out target edits alter metrics but never predictions/history.
    evaluation = np.load(path / "evaluation.npy")
    evaluation[0] += 100
    np.save(path / "evaluation.npy", evaluation)
    np.testing.assert_array_equal(expected, fixed_forecast(model, store, 28))
    assert (
        report["neural"]["global"]["wape"]
        != evaluate_model(model, store)["neural"]["global"]["wape"]
    )


def test_rng_checkpoint_and_contract(prepared, tmp_path):
    settings, path = prepared
    store = Store(path)
    seed_all(67)
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2])
    model = ForecastNetwork(store.metadata["scale"]).eval()
    checkpoint = tmp_path / "fixture.pt"
    save_checkpoint(
        checkpoint,
        model,
        store,
        settings,
        epoch=2,
        step=0,
        best=0.5,
        best_epoch=2,
        stale=0,
    )
    restored, payload = load_model(checkpoint, store)
    assert payload["optimizer"] is None
    np.testing.assert_array_equal(
        fixed_forecast(model, store, 2), fixed_forecast(restored, store, 2)
    )
    final = prepare(settings, "final")
    with pytest.raises(ValueError, match="contract"):
        load_model(checkpoint, Store(final))
    with patch(
        "tram_forecast.train.train", return_value=tmp_path / "final.pt"
    ) as train_call:
        assert refit(checkpoint, final) == tmp_path / "final.pt"
        chosen = train_call.call_args.args[0]
        assert chosen.training.epochs == 2
        assert train_call.call_args.kwargs["final"] is True

    from tram_forecast.checkpoint import read_checkpoint
    payload = torch.load(checkpoint, weights_only=True)
    payload['version'] = 2
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='retrain'):
        read_checkpoint(checkpoint)


def test_submission_complete_grid_and_template_order(tmp_path):
    keys = sorted(submission_keys(), reverse=True)
    template = tmp_path / "template.csv"
    with template.open("w") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        writer.writerows((r, d, h, 999) for r, d, h in keys)
    output = tmp_path / "submission.csv"
    values = {key: 0.0 if key[0] == 5 else 1.25 for key in keys}
    assert write_submission(template, output, values) == 14640
    with output.open() as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert int(rows[0]["route"]) == keys[0][0] and float(rows[0]["prediction"]) == 1.25
    values[keys[0]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_submission(template, output, values)
    assert len(rows) == 14640


def test_training_and_resume(prepared, tmp_path):
    from dataclasses import replace
    from tram_forecast.checkpoint import read_checkpoint
    from tram_forecast.train import train

    settings, artifact = prepared
    settings = replace(settings, training=replace(
        settings.training, batch_size=4, accumulation=2, epochs=1,
    ))
    output = tmp_path / "run"
    best = train(settings, artifact, output=output)
    assert best.is_file()
    first = read_checkpoint(output / "latest.pt")
    assert first["progress"]["epoch"] == 1
    settings = replace(settings, training=replace(settings.training, epochs=2))
    train(settings, artifact, resume=output / "latest.pt")
    resumed = read_checkpoint(output / "latest.pt")
    assert resumed["progress"]["epoch"] == 2
    assert resumed["progress"]["step"] > first["progress"]["step"]
    model, _ = load_model(best, Store(artifact))
    assert np.isfinite(fixed_forecast(model, Store(artifact), 7)).all()
