# Boarding-only implementation contracts

The executable architecture lives in `src/tram_forecast/config.py` and
`src/tram_forecast/model/{blocks,network}.py`. See the [manifest](architecture_manifest.md)
for the graph and default dimensions.

## Preparation and storage

Only `data.label_paths` are required. Rows have `route;date;hour;boardings`.
The parser rejects malformed dates, duplicate keys, negative counts and hours
outside 0–23. Missing route-hours remain zero, with source presence recorded
separately. Each configured neural route must have fitting-period observations;
route 5 observations conflict with the zero-fallback policy and are rejected.

Artifact schema version 2 contains:

- `boardings.npy`: float32 `[routes, fitting_hours]`.
- `label_present.npy`: boolean source-presence grid.
- `scaling.json`: scale and fitting boundaries.
- `metadata.json`: schema, source fingerprints, routes, intervals and coverage.
- Validation only: `evaluation.npy` and `evaluation_present.npy`, including route 5.

Preparation publishes atomically from a staging directory and refuses incompatible
reuse. Source fingerprints use path, size and modification time. History reads are
bounded to the fitting interval and copied from the memory-mapped boarding array.
There are no event arrays, offsets, vocabulary artifacts or DuckDB dependency.

## Model and batching API

`ForecastNetwork(scale, config=None, model_kind="boarding_only")` takes history
`counts [B,1,T]` and `calendar [B,6,T]`, where `T=24*history_days`.
`ForecastDataset` stores one index per route/cutoff and returns all eligible target
days. `collate_samples` creates R flattened requests with context indices selecting
one of the B histories. Outputs and targets are `[R,24]` in boarding units.
`Store.history(route, cutoff, history_days=21)` returns counts only.

`encode_history` and `predict_day` support reuse across future days. No prediction
is fed back as observed history. The `boarding_only` identifier is retained for
run directories, reporting and existing deployment integration; `full` is rejected.

## Training and evaluation

Defaults: seven-day horizon, original-unit MAE, AdamW (learning rate 0.001,
weight decay 0.0001), batch one context, accumulation eight, gradient norm limit
one, up to 30 epochs and patience five on validation WAPE. Partial horizons at
fitting boundaries remain included. Loss and gradient accumulation weight every
requested day equally. Contexts shuffle without replacement using seed + epoch.

Validation fits January–August and freezes history at September 1. Final refit
fits January–October from fresh weights and freezes history at November 1.
Horizon is configurable from 1 to 61 days and persisted in the checkpoint.
Weekly-profile baselines and route-5 zero predictions remain available.

## Checkpoints and migration

Version-2 checkpoints include weights, optimizer, resolved settings, boarding
artifact contract, scaling, RNG and epoch progress. Epoch-boundary resume requires
the same artifact and compatible model/training settings. Only the epoch limit
and output-root setting may change when resuming.

Rebuild old prepared artifacts in a new directory; existing artifacts are not
modified. Version-1 boarding-only checkpoints can still be read for head export
and fresh refits. Loading discards obsolete event settings without changing
weights. Their original artifact hashes do not match version-2 artifacts, so
cross-format resume/evaluation is rejected. Event-stream checkpoints are unsupported.

The Go service continues to consume exported boarding-only heads. Head export
folds the encoded history and route embedding into the first head-layer bias and
includes reference predictions for runtime verification.
