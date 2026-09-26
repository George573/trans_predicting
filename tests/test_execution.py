"""Orchestration checks use fixtures/untrained weights; no optimizer steps run."""

import csv
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tram_forecast.checkpoint import (
    load_model,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
)
from tram_forecast.cli import build_parser, main
from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.evaluate import (
    evaluate_model,
    fixed_forecast,
    metrics,
    weekly_profile,
)
from tram_forecast.losses import mae, normalize_gradients
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import submission_keys, write_submission
from tram_forecast.preprocess import prepare
from tram_forecast.storage import Store
from tram_forecast.train import refit


def test_aggregated_metrics_and_zero_denominator():
    report = metrics(np.array([1, 99]), np.array([2, 90]))
    assert report["wape"] == 0.1 and report["mae"] == 5
    assert metrics(np.zeros(2), np.ones(2))["wape"] is None
    with pytest.raises(ValueError):
        metrics(np.ones(2), np.ones(3))


def test_baseline_network_and_fixed_evaluation(prepared):
    settings, path = prepared
    store = Store(path)
    model = ForecastNetwork(
        store.metadata["scale"],
        settings.model,
        "boarding_only",
    )
    assert not hasattr(model, "events") and not hasattr(model, "raw")
    dataset = ForecastDataset(path, "boarding_only")
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
    model = ForecastNetwork(
        store.metadata["scale"],
        settings.model,
        "boarding_only",
    ).eval()
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


def test_partial_accumulation_matches_sample_mean():
    p = torch.nn.Parameter(torch.tensor(2.0))
    prediction = p.expand(3, 24)
    target = torch.zeros(3, 24)
    (mae(prediction[:2], target[:2]) * 2).backward(retain_graph=True)
    (mae(prediction[2:], target[2:]) * 1).backward()
    normalize_gradients([p], 3)
    assert p.grad.item() == pytest.approx(1.0)
    assert p.item() == 2.0  # No optimizer or model training.


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


def test_cli_contracts_without_execution(prepared, capsys):
    _, path = prepared
    main(["inspect", "--artifact", str(path)])
    assert "fingerprint" in capsys.readouterr().out
    for command in (
        "prepare",
        "inspect",
        "smoke",
        "train",
        "evaluate",
        "compare",
        "refit",
        "predict",
        "export-head",
    ):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args([command, "--help"])
        assert exc.value.code == 0


def test_training_control_flow_without_training(prepared, tmp_path):
    """Exercise epochs/accumulation/early stopping with inert model/loss/optimizer doubles."""
    from dataclasses import replace
    from unittest.mock import MagicMock

    from tram_forecast.train import train

    settings, artifact = prepared
    settings = replace(
        settings,
        training=replace(
            settings.training, batch_size=2, accumulation=2, epochs=4, patience=1
        ),
    )
    store = Store(artifact)
    dataset = MagicMock()
    dataset.store = store
    dataset.__len__.return_value = 5
    dataset.__getitem__.return_value = {}
    model = MagicMock()
    model.to.return_value = model
    model.parameters.return_value = []
    model.parameter_report.return_value = {"total": 1}
    loss = MagicMock()
    loss.__mul__.return_value = loss
    loss.detach.return_value = 1.0
    optimizer = MagicMock()
    saved = []

    def inert_save(path, *args, **progress):
        Path(path).touch()
        saved.append((Path(path).name, progress))

    with (
        patch("tram_forecast.train.ForecastDataset", return_value=dataset),
        patch("tram_forecast.train.ForecastNetwork", return_value=model),
        patch("tram_forecast.train.torch.optim.AdamW", return_value=optimizer),
        patch(
            "tram_forecast.train.collate_samples",
            side_effect=lambda samples, device: ({}, {}, torch.zeros(len(samples) * 7, 24)),
        ),
        patch("tram_forecast.train.mae", return_value=loss),
        patch("tram_forecast.train.normalize_gradients") as normalization,
        patch("tram_forecast.train.torch.nn.utils.clip_grad_norm_"),
        patch("tram_forecast.train.save_checkpoint", side_effect=inert_save),
        patch(
            "tram_forecast.train.evaluate_model",
            side_effect=[
                {"neural": {"global": {"wape": 0.5}}},
                {"neural": {"global": {"wape": 0.6}}},
            ],
        ),
    ):
        result = train(settings, artifact, "boarding_only", tmp_path / "run")
    assert result.name == "best.pt"
    assert optimizer.step.call_count == 4  # Spy only; no real optimizer exists.
    assert [call.args[1] for call in normalization.call_args_list] == [28, 7, 28, 7]
    assert [name for name, _ in saved] == ["latest.pt", "best.pt", "latest.pt"]
    assert saved[-1][1]["epoch"] == 2 and saved[-1][1]["best_epoch"] == 1


def test_smoke_is_backward_only(prepared):
    from tram_forecast.smoke import smoke

    settings, path = prepared
    with patch(
        "torch.optim.AdamW.step", side_effect=AssertionError("must not optimize")
    ):
        result = smoke(settings, path, "boarding_only")
    assert result["optimizer_steps"] == 0
    assert [item["case"] for item in result["checks"]] == ["typical", "busiest"]
    assert all(item["process_peak_rss_bytes"] > 0 for item in result["checks"])


def test_checkpoint_migration_and_event_model_rejection(prepared, tmp_path):
    from tram_forecast.checkpoint import read_checkpoint
    from tram_forecast.io import digest
    from tram_forecast.settings import Settings

    settings, artifact = prepared
    store = Store(artifact)
    model = ForecastNetwork(store.metadata['scale'], settings.model).eval()
    checkpoint = tmp_path / 'legacy.pt'
    save_checkpoint(checkpoint, model, store, settings, best_epoch=2)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload['version'] == 2 and 'vocab_sizes' not in payload
    # Simulate the previous boarding-only serialization contract.
    payload['version'] = 1
    payload['settings']['model'].update(event_depth=2, category_caps=[32, 128, 512, 32, 8192])
    payload['settings']['data'].update(raw_paths=['missing-events.csv'], memory_limit='2GB')
    payload['vocab_sizes'] = [3] * 5
    payload['artifact_contract']['vocab'] = {'legacy': {}}
    payload['artifact_hash'] = digest(payload['artifact_contract'])
    torch.save(payload, checkpoint)
    migrated = read_checkpoint(checkpoint)
    assert Settings.from_dict(migrated['settings']) == settings
    restored = ForecastNetwork(migrated['scale'], settings.model).eval()
    restored.load_state_dict(migrated['model'])
    np.testing.assert_array_equal(fixed_forecast(restored, store, 7), fixed_forecast(model, store, 7))
    with pytest.raises(ValueError, match='contract mismatch'):
        load_model(checkpoint, store)
    final = prepare(settings, 'final')
    with patch('tram_forecast.train.train', return_value=tmp_path / 'refit.pt') as call:
        refit(checkpoint, final)
    assert call.call_args.args[0].model == settings.model
    payload['model_kind'] = 'full'
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='event-stream checkpoints'):
        read_checkpoint(checkpoint)


def test_cli_defaults_to_boardings_and_rejects_full():
    parser = build_parser()
    for command in ('train', 'smoke'):
        args = parser.parse_args([command, '--artifact', 'unused'])
        assert args.model == 'boarding_only'
        with pytest.raises(SystemExit):
            parser.parse_args([command, '--artifact', 'unused', '--model', 'full'])
