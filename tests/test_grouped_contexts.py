"""Shared-context requests, split boundaries, and horizon persistence."""

import csv
import json
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tram_forecast.checkpoint import read_checkpoint, save_checkpoint
from tram_forecast.cli import main
from tram_forecast.dataset import ForecastDataset, collate_samples
from tram_forecast.evaluate import evaluate_model
from tram_forecast.losses import mae, normalize_gradients
from tram_forecast.model import ForecastNetwork
from tram_forecast.predict import predict, submission_keys
from tram_forecast.settings import TrainSettings
from tram_forecast.storage import Store
from tram_forecast.train import train


@pytest.mark.parametrize("days", [0, 62, 1.5, True])
def test_invalid_horizon(days):
    with pytest.raises(ValueError, match="forecast_days"):
        TrainSettings(forecast_days=days)


def test_contexts_have_all_eligible_targets_and_one_history(prepared):
    _, artifact = prepared
    dataset = ForecastDataset(artifact)
    assert len(dataset) == 10  # January 22..31, one route.
    assert len(set(map(tuple, dataset.index))) == len(dataset)
    for sample in dataset:
        cutoff = sample["identity"].cutoff
        days = min(7, (dataset.store.end - cutoff).days)
        assert sample["lead"].tolist() == list(range(1, days + 1))
        for lead, target in enumerate(sample["target"]):
            np.testing.assert_array_equal(
                target, dataset.store.target(1, cutoff + timedelta(days=lead))
            )
    history, request, targets = collate_samples([dataset[0], dataset[-1]])
    assert history["counts"].shape == (2, 1, 504)
    assert set(history) == {"counts", "calendar"}
    assert targets.shape == (8, 24)
    assert request["context_indices"].tolist() == [0] * 7 + [1]
    assert request["lead"].tolist() == list(range(1, 8)) + [1]
    longer = ForecastDataset(artifact, forecast_days=14)
    assert len(longer) == len(dataset)
    assert longer[0]["target"].shape == (10, 24)


@pytest.mark.parametrize("kind", ["boarding_only"])
def test_grouped_predictions_and_gradients_match_individual_requests(prepared, kind):
    settings, artifact = prepared
    dataset = ForecastDataset(artifact, kind)
    history, request, target = collate_samples([dataset[0], dataset[-1]])
    store = dataset.store
    model = ForecastNetwork(
        store.metadata["scale"],
        replace(settings.model, dropout=0), kind,
    ).train()
    reference = deepcopy(model)
    with patch.object(model, "encode_history", wraps=model.encode_history) as encode:
        grouped = model(history, request)
    assert encode.call_count == 1
    assert encode.call_args.args[0]["counts"].shape[0] == 2
    individual = []
    for i, context in enumerate(request["context_indices"].tolist()):
        single_history = {
            "counts": history["counts"][context:context + 1],
            "calendar": history["calendar"][context:context + 1],
        }
        single_request = {k: v[i:i + 1] for k, v in request.items() if k != "context_indices"}
        individual.append(reference(single_history, single_request))
    separate = torch.cat(individual)
    torch.testing.assert_close(grouped, separate, atol=1e-5, rtol=1e-4)
    mae(grouped, target).backward()
    mae(separate, target).backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.boarding.parameters())


def test_partial_horizons_accumulate_by_requested_day():
    p = torch.nn.Parameter(torch.tensor(2.0))
    # Seven targets below prediction and one above: gradient (7 - 1) / 8.
    (mae(p.expand(7, 24), torch.zeros(7, 24)) * 7).backward()
    mae(p.expand(1, 24), torch.full((1, 24), 4.0)).backward()
    normalize_gradients([p], 8)
    assert p.grad.item() == pytest.approx(0.75)


def test_evaluation_ignores_targets_after_horizon(prepared):
    settings, artifact = prepared
    store = Store(artifact)
    model = ForecastNetwork(store.metadata["scale"], settings.model, "boarding_only")
    before = evaluate_model(model, store)
    values = np.load(artifact / "evaluation.npy")
    values[:, 7 * 24:] = 9999
    np.save(artifact / "evaluation.npy", values)
    assert evaluate_model(model, store) == before
    assert set(before["neural"]["leads"]) == {str(i) for i in range(1, 8)}
    with pytest.raises(ValueError, match="available"):
        evaluate_model(model, store, 29)


@pytest.mark.parametrize("days", [7, 14])
def test_checkpoint_horizon_and_resume_contract(prepared, tmp_path, capsys, days):
    settings, artifact = prepared
    settings = replace(settings, training=replace(settings.training, forecast_days=days))
    store = Store(artifact)
    model = ForecastNetwork(store.metadata["scale"], settings.model, "boarding_only")
    path = tmp_path / "run" / "latest.pt"
    save_checkpoint(path, model, store, settings)
    assert read_checkpoint(path)["settings"]["training"]["forecast_days"] == days
    main(["evaluate", "--checkpoint", str(path), "--artifact", str(artifact)])
    assert json.loads(capsys.readouterr().out)["forecast_days"] == days
    changed = replace(settings, training=replace(settings.training, forecast_days=21))
    with pytest.raises(ValueError, match="resume configuration"):
        train(changed, artifact, "boarding_only", resume=path)
    payload = torch.load(path, weights_only=True)
    del payload["settings"]["training"]["forecast_days"]
    del payload["training_layout"]
    torch.save(payload, path)
    assert read_checkpoint(path)["settings"]["training"]["forecast_days"] == 61
    with pytest.raises(ValueError, match="legacy training layout"):
        train(settings, artifact, "boarding_only", resume=path)


def test_compare_rejects_different_horizons(tmp_path):
    reports = []
    for days in (7, 14):
        path = tmp_path / f"report-{days}.json"
        path.write_text(json.dumps({"artifact": "same", "cutoff": "2025-09-01", "forecast_days": days}))
        reports.append(str(path))
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--reports", *reports])
    assert exc.value.code == 2


@pytest.mark.parametrize("days", [7, 61])
def test_prediction_export_uses_checkpoint_horizon(tmp_path, days):
    from unittest.mock import MagicMock

    from tram_forecast.config import ROUTES

    store = MagicMock()
    store.metadata = {"regime": "final", "evaluation_end": "2026-01-01"}
    store.end = date(2025, 11, 1)
    store.routes = ROUTES
    store.contract_hash = "fixture"
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"provenance fixture")
    template = tmp_path / "week.csv"
    keys = sorted(submission_keys(store.end, store.end + timedelta(days=days)), reverse=True)
    with template.open("w") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        writer.writerows((*key, 999) for key in keys)
    output = tmp_path / "forecast.csv"
    payload = {"model_kind": "boarding_only", "settings": {"training": {"forecast_days": days}}}
    with (
        patch("tram_forecast.checkpoint.read_checkpoint", return_value=payload),
        patch("tram_forecast.predict.Store", return_value=store),
        patch("tram_forecast.predict.load_model", return_value=(object(), payload)),
        patch("tram_forecast.predict.fixed_forecast", return_value=np.ones((9, days, 24))) as forecast,
    ):
        predict(checkpoint, "unused", template, output)
    assert forecast.call_args.args[2] == days
    with output.open() as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert len(rows) == 10 * days * 24
    assert [int(row["route"]) for row in rows] == [key[0] for key in keys]
    assert all(float(row["prediction"]) == 0 for row in rows if row["route"] == "5")
    assert json.loads(Path(str(output) + ".provenance.json").read_text())["forecast_days"] == days
