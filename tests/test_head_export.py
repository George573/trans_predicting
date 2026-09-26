import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tram_forecast.checkpoint import save_checkpoint
from tram_forecast.config import ROUTES
from tram_forecast.evaluate import fixed_forecast
from tram_forecast.head_export import export_head
from tram_forecast.model import ForecastNetwork
from tram_forecast.data import Store, request_calendar


def test_exported_head_matches_torch_forecast(prepared, tmp_path):
    settings, path = prepared
    store = Store(path)
    torch.manual_seed(0)
    model = ForecastNetwork(store.metadata["scale"]).eval()
    save_checkpoint(tmp_path / "model.pt", model, store, settings, epoch=1)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        Path(settings.data.label_paths[0]).read_text()
        + "".join(f"{r};2025-01-15;8;{r}\n" for r in ROUTES if r != 1)
    )
    head = json.loads(
        export_head(tmp_path / "model.pt", [labels], tmp_path / "head.json").read_text()
    )
    dates = [store.end + timedelta(days=d) for d in range(7)]
    hidden = (
        torch.tensor(head["hidden_base"][0])
        + torch.as_tensor(request_calendar(dates))
        @ torch.tensor(head["hidden_calendar"]).T
        + torch.arange(7.0)[:, None] / 60 * torch.tensor(head["hidden_lead"])
    )
    values = (
        F.softplus(
            F.gelu(hidden) @ torch.tensor(head["output_weight"]).T
            + torch.tensor(head["output_bias"])
        )
        * head["scale"]
    )
    np.testing.assert_allclose(
        values[None].numpy(), fixed_forecast(model, store, 7), rtol=1e-4, atol=1e-3
    )
