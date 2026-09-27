"""Independent checks: pytest --confcutdir=experiments experiments/patch_transformer."""

from datetime import timedelta

import numpy as np
import pytest
import torch
from run import CUTOFF, FINAL, START, Windows, load_data, save

from tram_forecast.data import collate_samples, sample
from tram_forecast.patch_model import PatchForecastNetwork
from tram_forecast.predict import fixed_forecast
from tram_forecast.runner import InferenceRunner
from tram_forecast.train import load_model

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def boardings():
    return load_data(FINAL)


def test_cached_windows_match_public_features(boardings):
    windows = Windows(boardings, 28, leads=61)
    for index in (0, 143, len(windows) - 1):
        fast = windows[index]
        r, c = windows.index[index]
        regular = sample(
            boardings,
            boardings.routes[r],
            START + timedelta(days=c),
            min(61, windows.days - c),
            28,
        )
        for key in regular:
            np.testing.assert_allclose(fast[key], regular[key], atol=1e-6)
    assert len(windows[-1]["lead"]) == 1


def test_seasonal_initialization_and_grouped_contexts(boardings):
    samples = [sample(boardings, r, CUTOFF, 61, 28) for r in (1, 7)]
    history, request, _ = collate_samples(samples)
    model = PatchForecastNetwork(boardings.scale).eval()
    grouped = model(history, request)
    expected = np.median(history["counts"].numpy().reshape(2, 4, 7, 24), axis=1)
    expected = expected[:, np.arange(61) % 7].reshape(-1, 24)
    np.testing.assert_allclose(grouped.detach(), expected, rtol=1e-5, atol=0.002)
    separate = torch.cat([model(*collate_samples([s])[:2]) for s in samples])
    torch.testing.assert_close(grouped, separate)
    # After the residual head updates, gradients reach the attention encoder.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        prediction = model(history, request)
        (prediction - 123).abs().mean().backward()
        optimizer.step()
    assert any(
        p.grad.abs().sum() > 0 for p in model.encoder.parameters() if p.grad is not None
    )
    assert torch.isfinite(prediction).all() and (prediction >= 0).all()


def test_future_labels_cannot_change_forecast_and_checkpoint(boardings, tmp_path):
    model = PatchForecastNetwork(boardings.scale).eval()
    before = fixed_forecast(model, boardings, CUTOFF, 61, 28)
    offset = (CUTOFF - START).days * 24
    original = boardings.counts[:, offset:].copy()
    try:
        boardings.counts[:, offset:] = 1e8
        np.testing.assert_array_equal(
            before, fixed_forecast(model, boardings, CUTOFF, 61, 28)
        )
    finally:
        boardings.counts[:, offset:] = original
    path = tmp_path / "patch.pt"
    save(model, "patch", 28, 1, 67, path)
    restored = load_model(path)
    np.testing.assert_array_equal(
        before, fixed_forecast(restored, boardings, CUTOFF, 61, 28)
    )
    runner = InferenceRunner(restored)
    predicted = runner.predict(boardings, 1, CUTOFF, CUTOFF + timedelta(days=60))
    np.testing.assert_allclose(before[0, 60], predicted, atol=0.002, rtol=1e-5)


def test_train_cutoff_excludes_validation_labels():
    train = load_data(CUTOFF)
    assert train.counts.shape[1] == (CUTOFF - START).days * 24
    windows = Windows(train, 28)
    assert all(c < (CUTOFF - START).days for _, c in windows.index)
    assert windows[-1]["target"].shape == (1, 24)
    with pytest.raises(ValueError, match="multiple of seven"):
        PatchForecastNetwork(history_days=15)
    with pytest.raises(ValueError, match="history hours"):
        PatchForecastNetwork().encode_history({"counts": torch.zeros(1, 1, 336)})


def test_existing_cnn_checkpoint_remains_loadable(tmp_path):
    from tram_forecast.model import ForecastNetwork

    model = ForecastNetwork(100).eval()
    legacy_path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "scale": 100}, legacy_path)
    restored = load_model(legacy_path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    unknown_path = tmp_path / "unknown.pt"
    torch.save({"architecture": "unknown"}, unknown_path)
    with pytest.raises(ValueError, match="Unknown checkpoint architecture"):
        load_model(unknown_path)
