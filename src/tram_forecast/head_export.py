import hashlib
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import torch

from .checkpoint import read_checkpoint
from .config import ROUTES
from .io import write_json
from .model import ForecastNetwork
from .data import calendar, label_grid, request_calendar


def export_head(checkpoint, labels, output):
    payload = read_checkpoint(checkpoint)
    model = ForecastNetwork(payload["scale"])
    model.load_state_dict(payload["model"])
    model.eval()
    cutoff = date.fromisoformat(payload["artifact_contract"]["end"])
    first = cutoff - timedelta(days=21)
    counts, present = label_grid(labels, ROUTES, first, cutoff)
    if not present.any(axis=1).all():
        raise ValueError(f"labels must cover {first}..{cutoff} for every route")
    days = payload["settings"]["training"]["forecast_days"]
    n = len(ROUTES)
    start = datetime.combine(first, time.min)
    hours = calendar(start + timedelta(hours=i) for i in range(504))
    hidden, out = model.head[0], model.head[3]
    with torch.inference_mode():
        encoded = model.encode_history(
            {
                "counts": torch.as_tensor(counts[:, None]),
                "calendar": torch.as_tensor(np.repeat(hours[None], n, axis=0)),
            }
        )
        e = encoded.shape[1]
        c = e + model.route.embedding_dim
        base = (
            encoded @ hidden.weight[:, :e].T
            + model.route.weight[1 : n + 1] @ hidden.weight[:, e:c].T
            + hidden.bias
        )
        check = []
        for i, route in enumerate(ROUTES):
            for lead in sorted({1, (days + 1) // 2, days}):
                day = cutoff + timedelta(days=lead - 1)
                values = model.predict_day(
                    encoded[i : i + 1],
                    torch.tensor([i + 1]),
                    torch.as_tensor(request_calendar([day])),
                    torch.tensor([float(lead)]),
                )
                check.append(
                    {"route": route, "date": str(day), "values": values[0].tolist()}
                )
        write_json(
            output,
            {
                "model_kind": payload["model_kind"],
                "cutoff": str(cutoff),
                "forecast_days": days,
                "routes": list(ROUTES),
                "parameters": sum(p.numel() for p in model.parameters()),
                "scale": payload["scale"],
                "checkpoint_sha256": hashlib.sha256(
                    Path(checkpoint).read_bytes()
                ).hexdigest(),
                "history": counts.tolist(),
                "hidden_base": base.tolist(),
                "hidden_calendar": hidden.weight[:, c : c + 4].tolist(),
                "hidden_lead": hidden.weight[:, c + 4].tolist(),
                "output_weight": out.weight.tolist(),
                "output_bias": out.bias.tolist(),
                "check": check,
            },
        )
    return Path(output)
