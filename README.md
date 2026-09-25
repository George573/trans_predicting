# Tram ridership forecasting

A PyTorch package that predicts one requested day's 24 hourly boarding counts
from 21 days of observed route history. Every convolutional stage uses parallel
kernel lengths and dilations. The full model combines raw events and boarding
history; a boarding-only model and weekly-profile baseline are also provided.

## Repository layout

| Path | Contents |
|---|---|
| `src/tram_forecast/` | Preprocessing, storage, dataset, model, training, evaluation, inference and CLI |
| `configs/default.json` | Model, data and training settings |
| `tests/` | Synthetic fixtures and correctness tests; no optimization runs |
| `docs/architecture/` | Architecture manifest and detailed implementation specification |
| `docs/development/` | Implementation checklist and progress/evidence tracker |
| `experiments/legacy/` | Earlier one-hour window utilities, retained outside the production package |
| `tools/` | Optional standalone profiling tools; not required by the package |
| `dataset/README.md` | Dataset rules; actual data remains local and ignored |
| `outputs/` | Ignored preparation artifacts, checkpoints, metrics and submissions |

[Implementation specification](docs/architecture/architecture_implementation_plan.md)
· [Steps](docs/development/implementation_steps.md)
· [Progress](docs/development/implementation_progress.md)

## Install and verify

Python 3.10+; verified locally with Python 3.14, PyTorch 2.14.0+cpu and DuckDB 1.5.5.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m tram_forecast check-config --config configs/default.json
.venv/bin/python -m pytest -q
```

For CPU-only PyTorch, install `torch` from the official
`https://download.pytorch.org/whl/cpu` index before the package. For a CUDA host,
install the appropriate official PyTorch wheel. Set `training.device` or pass
`--device cuda`. GPU execution has not been validated here.

All commands below assume the environment is activated. Commands that perform
training are implemented but were **not run** during this code-only delivery.
Tests use temporary synthetic CSVs and untrained weights; backward checks do not
perform optimizer steps. No full-data preparation or real forecast was run.

## Prepare data

Expected local files: `dataset/train.csv`, `dataset/test.csv`,
`dataset/labels/labels_day_train.csv`, `dataset/labels/labels_day_test.csv`.
Configure alternative paths in `configs/default.json`.

```bash
python -m tram_forecast prepare --config configs/default.json --regime validation
python -m tram_forecast inspect --artifact outputs/prepared/validation
```

Preparation assigns events by timestamp, validates input schemas, fits capped
vocabularies on January–August only, sorts events on disk, and atomically publishes
memory-mapped arrays. Validation labels are separate from history storage. File
path/size/mtime and preprocessing settings identify reusable artifacts. A changed
source or incompatible configuration requires a new artifact directory, supplied
with `--artifact PATH`. These fingerprints are compatibility checks, not content
hashes. Do not modify artifacts after publication.

DuckDB memory, threads, temporary directory, fetch batch size and minimum free
space are configurable. The free-space check is a threshold, not a guarantee
that a large sort will fit. Preparation reports progress by input file and basic
coverage/reconciliation diagnostics. It does not run the optional full profiler.

## Resource smoke check

```bash
python -m tram_forecast smoke --artifact outputs/prepared/validation --model full --output outputs/smoke.json
```

This explicitly requested command checks typical and busiest histories with a
forward/backward pass and no optimizer step. It reports CPU process high-water
RSS and CUDA peaks where available. It tests batch size one; larger training
batches still require checking. If an OOM occurs, lower the event chunk budget or
batch size and rerun. No automatic truncation or architecture changes occur.

## Training and evaluation

```bash
python -m tram_forecast train --artifact outputs/prepared/validation --model boarding_only
python -m tram_forecast train --artifact outputs/prepared/validation --model full
python -m tram_forecast evaluate --checkpoint outputs/runs/validation/boarding_only/best.pt --artifact outputs/prepared/validation --output outputs/boarding_metrics.json
python -m tram_forecast evaluate --checkpoint outputs/runs/validation/full/best.pt --artifact outputs/prepared/validation --output outputs/full_metrics.json
python -m tram_forecast compare --reports outputs/boarding_metrics.json outputs/full_metrics.json --output outputs/comparison.json
```

Training defaults to original-unit MAE, AdamW, batch size 1 with accumulation 8,
30 maximum epochs and patience 5. Validation freezes history at September 1 and
predicts all September–October days directly. Reports include WAPE/MAE by route,
lead and lead group, and the weekly-profile baseline. No validation observation
updates the history. Route 5 uses a separate zero fallback.

Checkpoints include model, optimizer, settings, fitted vocabulary/scaling contract,
RNG state and epoch progress. Resume from the latest checkpoint in the same run
directory; only increasing the epoch limit or changing the configured output root
is accepted without a fresh run:

```bash
python -m tram_forecast train --artifact outputs/prepared/validation --model full --resume outputs/runs/validation/full/latest.pt
```

Epoch-boundary resume is supported. An interrupted partial epoch is rerun. Keep
`best.pt`, `latest.pt` and `history.json` together. A fresh run refuses to overwrite
an existing checkpoint directory; use `--output NEW_DIRECTORY`.

## Final refit and submission

After choosing a validation model, execute these commands explicitly:

```bash
python -m tram_forecast prepare --regime final
python -m tram_forecast refit --selected-checkpoint outputs/runs/validation/full/best.pt --artifact outputs/prepared/final
python -m tram_forecast predict --checkpoint outputs/runs/final/full/final.pt --artifact outputs/prepared/final --template dataset/test_submission.csv --output outputs/submission.csv
```

Refit rebuilds mappings/scaling for January–October and starts fresh weights for
the selected epoch count. Submission reuses each route's October 11–31 encoding
for all 61 requested days. It preserves template order, validates all 14,640
unique keys, writes floating-point nonnegative predictions and records provenance.
November–December hidden labels are never used.

## Model API

`ForecastNetwork` receives original-unit `counts [B,1,504]`, historical
`calendar [B,6,504]`, and (full model only) `hours`: `B*504` chronological int64
`[events,5]` tensors. Empty hours use `[0,5]`. Ragged IDs may remain on CPU;
floating inputs/request tensors must share the model device. Request fields are
`route_indices [B]`, `calendar [B,4]`, `lead [B]`. Outputs are `[B,24]` in original
boarding units. Use `ForecastDataset` and `collate_samples` to build these inputs.

Learned event vectors are not cached during training. Inference uses
`encode_history` once per route/cutoff and `predict_day` for each requested day.
The default full network has 128,704 nonembedding parameters and at most 199,268
including capped embeddings.

Raw excerpts, the exploratory notebook and old PDFs remain local under ignored
paths (`samples/`, `experiments/references/`). Source datasets were not deleted.

## Scientific architecture PDF

The five-page vector architecture note can be regenerated from the default model:

```bash
.venv/bin/python -m pip install -e '.[docs]'
.venv/bin/python tools/docs/build_architecture_pdf.py
```

Output: `output/pdf/tram_cnn_scientific_architecture.pdf` (ignored generated artifact).
The builder checks the full-model parameter ceiling and includes the source Git
revision. It documents the full computation graph, every multiscale path, temporal
support, exact event tiling, embeddings and the fixed-cutoff evaluation protocol.
The user-supplied TFT visual reference is kept locally in
`experiments/references/tft_diagram_reference.png`.
