import hashlib
from pathlib import Path

import torch

from .checkpoint import read_checkpoint
from .config import ROUTES
from .io import write_json
from .model import ForecastNetwork
from .settings import Settings


class HistoryEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, counts, calendar):
        return self.model.encode_history({"counts": counts, "calendar": calendar})


class DayHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoded, route_indices, calendar, lead):
        return self.model.predict_day(encoded, route_indices, calendar, lead)


def export_onnx(checkpoint, output):
    payload = read_checkpoint(checkpoint)
    if payload["model_kind"] != "boarding_only":
        raise ValueError("the runner serves boarding-only checkpoints")
    model = ForecastNetwork(
        payload["vocab_sizes"],
        payload["scale"],
        Settings.from_dict(payload["settings"]).model,
        payload["model_kind"],
    )
    model.load_state_dict(payload["model"])
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(ROUTES)
    counts = torch.rand(n, 1, 504) * payload["scale"]
    calendar = torch.rand(n, 6, 504) * 2 - 1
    torch.onnx.export(
        HistoryEncoder(model),
        (counts, calendar),
        output / "encoder.onnx",
        input_names=["counts", "calendar"],
        output_names=["encoded"],
        dynamic_axes={"counts": {0: "routes"}, "calendar": {0: "routes"}},
        dynamo=False,
    )
    with torch.no_grad():
        encoded = HistoryEncoder(model)(counts, calendar)
    torch.onnx.export(
        DayHead(model),
        (
            encoded,
            torch.arange(1, n + 1),
            torch.rand(n, 4) * 2 - 1,
            torch.arange(1, n + 1, dtype=torch.float32),
        ),
        output / "head.onnx",
        input_names=["encoded", "route_indices", "calendar", "lead"],
        output_names=["boardings"],
        dynamic_axes={
            name: {0: "requests"}
            for name in ("encoded", "route_indices", "calendar", "lead", "boardings")
        },
        dynamo=False,
    )
    write_json(
        output / "model.json",
        {
            "model_kind": payload["model_kind"],
            "cutoff": payload["artifact_contract"]["end"],
            "forecast_days": payload["settings"]["training"]["forecast_days"],
            "routes": list(ROUTES),
            "parameters": sum(p.numel() for p in model.parameters()),
            "scale": payload["scale"],
            "checkpoint_sha256": hashlib.sha256(
                Path(checkpoint).read_bytes()
            ).hexdigest(),
        },
    )
    return output
