"""Explicit resource checks; backward-only, never an optimizer/training run."""

import resource
import sys
import time
from datetime import timedelta

import numpy as np
import torch

from .checkpoint import seed_all
from .dataset import collate_samples, history_sample
from .model import ForecastNetwork
from .schema import SampleIdentity, request_calendar
from .storage import Store


def smoke(settings, artifact, model_kind="full"):
    seed_all(settings.model.seed)
    store = Store(artifact, events=model_kind == "full")
    model = ForecastNetwork(
        store.metadata["vocab_sizes"],
        store.metadata["scale"],
        settings.model,
        model_kind,
    ).to(settings.training.device)
    counts = store.offsets[:, :, 1] - store.offsets[:, :, 0]
    candidates = []
    for r, route in enumerate(store.routes):
        prefix = np.concatenate(([0], np.cumsum(counts[r], dtype=np.int64)))
        for day in range(21, (store.end - store.start).days + 1):
            stop = day * 24
            candidates.append((int(prefix[stop] - prefix[stop - 504]), route, day))
    candidates.sort()
    if not candidates:
        raise ValueError("no complete histories for smoke check")
    results = []
    for label, (_, route, day) in zip(
        ("typical", "busiest"), (candidates[len(candidates) // 2], candidates[-1])
    ):
        cutoff = store.start + timedelta(days=day)
        sample = history_sample(
            store, SampleIdentity(route, cutoff, cutoff), model_kind == "full"
        )
        days = settings.training.forecast_days
        sample["lead"] = np.arange(1, days + 1, dtype=np.float32)
        sample["request_calendar"] = request_calendar(
            [cutoff + timedelta(days=i) for i in range(days)]
        )
        inputs, request, _ = collate_samples([sample], settings.training.device)
        if settings.training.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        started = time.monotonic()
        output = model(inputs, request)
        output.mean().backward()
        if not torch.isfinite(output).all() or any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("nonfinite forward/backward result")
        if settings.training.device.startswith("cuda"):
            torch.cuda.synchronize()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        results.append(
            {
                "case": label,
                "route": route,
                "cutoff": str(cutoff),
                "seconds": time.monotonic() - started,
                "process_peak_rss_bytes": peak,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if settings.training.device.startswith("cuda")
                else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                if settings.training.device.startswith("cuda")
                else None,
            }
        )
    return {
        "checks": results,
        "parameters": model.parameter_report(),
        "artifact": store.contract_hash,
        "optimizer_steps": 0,
        "batch_size_tested": 1,
        "forecast_days": settings.training.forecast_days,
    }
