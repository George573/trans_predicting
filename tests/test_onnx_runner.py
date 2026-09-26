from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from tram_forecast.checkpoint import save_checkpoint
from tram_forecast.config import ROUTES
from tram_forecast.evaluate import fixed_forecast
from tram_forecast.model import ForecastNetwork
from tram_forecast.storage import Store


def test_onnx_runner_matches_torch_forecast(prepared, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from tram_forecast.onnx_export import export_onnx
    from tram_forecast.serve import Runner

    settings, path = prepared
    store = Store(path, events=False)
    torch.manual_seed(0)
    model = ForecastNetwork(
        store.metadata["vocab_sizes"],
        store.metadata["scale"],
        settings.model,
        "boarding_only",
    ).eval()
    save_checkpoint(tmp_path / "model.pt", model, store, settings, epoch=1)
    export_onnx(tmp_path / "model.pt", tmp_path / "onnx")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        Path(settings.data.label_paths[0]).read_text()
        + "".join(f"{r};2025-01-15;8;{r}\n" for r in ROUTES if r != 1)
    )
    runner = Runner(tmp_path / "onnx", [labels])
    values = runner.predict([1, 5], date(2025, 2, 1), 7)
    np.testing.assert_allclose(
        values[:1], fixed_forecast(model, store, 7), rtol=1e-4, atol=1e-3
    )
    assert not values[1].any()
    with pytest.raises(ValueError, match="forecast covers"):
        runner.predict([1], date(2025, 2, 2), 7)
    with pytest.raises(ValueError, match="unknown routes"):
        runner.predict([99], date(2025, 2, 1), 1)
