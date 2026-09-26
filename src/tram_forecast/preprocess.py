"""Bounded disk preparation. No exploratory full-data profiling or training."""

import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

from .io import digest, source_fingerprints, write_json
from .schema import count_scale, label_grid

SCHEMA_VERSION = 2


def preparation_identity(settings, regime):
    if regime not in ("validation", "final"):
        raise ValueError("unknown preparation regime")
    data = settings.data
    end = data.validation_cutoff if regime == "validation" else data.final_cutoff
    return {
        "schema_version": SCHEMA_VERSION,
        "regime": regime,
        "start": data.start,
        "end": end,
        "data": settings.to_dict()["data"],
        "sources": source_fingerprints(data.label_paths),
    }


def prepare(settings, regime, output=None):
    """Publish immutable artifacts atomically; refuse to overwrite incompatible data."""
    identity = preparation_identity(settings, regime)
    fingerprint = digest(identity)
    output = Path(output or Path(settings.data.output_root) / regime)
    if output.exists():
        from .storage import Store

        existing = Store(output)
        if existing.metadata["fingerprint"] != fingerprint:
            raise ValueError(
                f"{output}: incompatible existing artifact; use a new output directory"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < settings.data.min_free_disk_bytes:
        raise OSError(f"insufficient free space in {output.parent}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.monotonic()
    try:
        _build(settings, identity, fingerprint, stage)
        if (
            source_fingerprints(settings.data.label_paths)
            != identity["sources"]
        ):
            raise ValueError("source files changed during preparation")
        stage.rename(output)
        print(f"Prepared {output} in {time.monotonic() - started:.1f}s", flush=True)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return output


def _build(settings, identity, fingerprint, stage):
    data = settings.data
    routes = tuple(data.routes)
    start = date.fromisoformat(identity["start"])
    end = date.fromisoformat(identity["end"])
    # Route 5 is deliberately checked even though it has no neural index.
    counts, presence = label_grid(data.label_paths, (*routes, 5), start, end)
    if presence[-1].any():
        raise ValueError(
            "route 5 has labels: conflicts with the no-history fallback policy"
        )
    counts = counts[:-1]
    presence = presence[:-1]
    if not presence.any(axis=1).all():
        raise ValueError("a configured neural route has no fitting-period labels")
    np.save(stage / "boardings.npy", counts)
    np.save(stage / "label_present.npy", presence)
    scale = count_scale(counts)
    write_json(
        stage / "scaling.json", {"scale": scale, "start": str(start), "end": str(end)}
    )
    eval_end = date.fromisoformat(
        data.final_cutoff if identity["regime"] == "validation" else data.forecast_end
    )
    if identity["regime"] == "validation":
        targets, target_present = label_grid(
            data.label_paths, (*routes, 5), end, eval_end
        )
        np.save(stage / "evaluation.npy", targets)
        np.save(stage / "evaluation_present.npy", target_present)
    write_json(stage / "metadata.json", {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "regime": identity["regime"],
        "start": str(start),
        "end": str(end),
        "evaluation_end": str(eval_end),
        "routes": routes,
        "label_rows_by_route": presence.sum(axis=1).tolist(),
        "scale": scale,
    })
