"""Shared-context requests, split boundaries, and horizon persistence."""

from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tram_forecast.data import ForecastDataset, collate_samples
from tram_forecast.train import mae, normalize_gradients
from tram_forecast.model import ForecastNetwork


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


def test_grouped_predictions_and_gradients_match_individual_requests(prepared):
    settings, artifact = prepared
    dataset = ForecastDataset(artifact)
    history, request, target = collate_samples([dataset[0], dataset[-1]])
    store = dataset.store
    model = ForecastNetwork(store.metadata["scale"]).eval()
    reference = deepcopy(model)
    with patch.object(model, "encode_history", wraps=model.encode_history) as encode:
        grouped = model(history, request)
    assert grouped.shape == target.shape
    assert torch.isfinite(grouped).all() and (grouped >= 0).all()
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
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_partial_horizons_accumulate_by_requested_day():
    p = torch.nn.Parameter(torch.tensor(2.0))
    # Seven targets below prediction and one above: gradient (7 - 1) / 8.
    (mae(p.expand(7, 24), torch.zeros(7, 24)) * 7).backward()
    mae(p.expand(1, 24), torch.full((1, 24), 4.0)).backward()
    normalize_gradients([p], 8)
    assert p.grad.item() == pytest.approx(0.75)
