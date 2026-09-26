# Tram ridership forecasting

A PyTorch package that encodes 21 days of observed route history once and predicts
24 hourly boarding counts for each requested future day. The initial horizon is
seven days for both neural models and the weekly-profile baseline. Every convolutional stage uses parallel
kernel lengths and dilations. The full model combines raw events and boarding
history; a boarding-only model and weekly-profile baseline are also provided.

## Repository layout

| Path | Contents |
|---|---|
| `src/tram_forecast/` | Preprocessing, storage, dataset, model, training, evaluation, inference and CLI |
| `configs/default.json` | Model, data and training settings |
| `tests/` | Synthetic fixtures and correctness tests; no optimization runs |
| `notebooks/train.ipynb` | Interactive training, resume, baseline comparison and forecast plots |
| `docs/architecture/` | Architecture manifest and detailed implementation specification |
| `docs/development/` | Implementation checklist and progress/evidence tracker |
| `experiments/legacy/` | Earlier one-hour window utilities, retained outside the production package |
| `tools/` | Optional standalone profiling tools; not required by the package |
| `dataset/README.md` | Dataset rules; actual data remains local and ignored |
| `outputs/` | Ignored preparation artifacts, checkpoints, metrics and submissions |
| `bench/go-inference-stand/` | Go API and dispatcher UI that query the ONNX runner |
| `Dockerfile`, `docker-compose.yml` | Runner and UI images; the checkpoint is exported to ONNX at build time |

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

For a **GTX 1080 (Pascal)**, use the legacy CUDA installer in your virtual environment:

```bash
.venv/bin/python tools/install.py --legacy-cuda --notebook --test
```

On Windows, use `.venv\Scripts\python.exe` instead. The installer pins
`torch==2.14.0+cu126` from the official CUDA 12.6 wheel index, then installs the
project while retaining that exact build. Your NVIDIA 580.178.04 driver can run
this build. The CUDA 13.0 label in `nvidia-smi` describes driver capability;
the wheel must also support the GPU architecture. See the
[PyTorch Pascal support notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).

`--notebook` and `--test` are optional. Add `--dry-run` to preview the commands.
Without `--legacy-cuda`, the script installs the project using normal pip
dependency selection. The flag belongs to this script, not to pip. Use the
legacy flag again when installing additional extras through this script.

All commands below assume the environment is activated. Commands that perform
training are implemented but were **not run** during this code-only delivery.
Tests use temporary synthetic CSVs and untrained weights; backward checks do not
perform optimizer steps. No full-data preparation or real forecast was run.

## Training notebook

Open [notebooks/train.ipynb](notebooks/train.ipynb) for the interactive workflow.
Install notebook dependencies into the same environment as the training package:

```bash
.venv/bin/python -m pip install -e '.[notebook]'
.venv/bin/python -m jupyterlab notebooks/train.ipynb
```

In VS Code, select `.venv/bin/python` as the notebook kernel. The configuration
cell controls model kind, forecast horizon, device, batch size, accumulation,
epochs, output directory and resume checkpoint. The default is boarding-only
with a seven-day horizon; CUDA is selected when available. Run cells in order
to prepare data, inspect the weekly baseline, check memory, train, and plot
learning curves and forecasts. The Train cell performs actual optimization.
Change the run name for a fresh experiment, or set `RESUME` to `latest.pt` to
continue the same run. The notebook is saved without outputs or trained weights.

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

Training defaults to original-unit MAE, AdamW, one context per batch with
accumulation 8, 30 maximum epochs and patience 5. Each context is a route/cutoff
pair with a 21-day history and all eligible targets at leads 1–7. It is encoded
once per forward pass; the head predicts each requested day using that shared
representation. Contexts are shuffled without replacement. Near the fitting
boundary, shorter target groups are retained; loss and gradient accumulation
weight every requested day equally.

Validation freezes history at September 1 and predicts September 1–7 directly.
Reports include WAPE/MAE by route,
lead and lead group, and the weekly-profile baseline. No validation observation
updates the history. Route 5 uses a separate zero fallback.

Set `training.forecast_days` in `configs/default.json` to expand the horizon
(integer 1–61). Start a fresh run for each horizon; prepared artifacts can be
reused. Evaluation reads the horizon from the checkpoint, and comparison rejects
reports with different horizons. The evaluation period must cover the requested
horizon. Lead conditioning retains `(lead-1)/60`, so expansion does not change
network geometry or parameter count.

Checkpoints include model, optimizer, settings, fitted vocabulary/scaling contract,
RNG state and epoch progress. Resume from the latest checkpoint in the same run
directory; only changing the epoch limit or the configured output root
is accepted without a fresh run:

```bash
python -m tram_forecast train --artifact outputs/prepared/validation --model full --resume outputs/runs/validation/full/latest.pt
```

Epoch-boundary resume is supported. An interrupted partial epoch is rerun. Keep
`best.pt`, `latest.pt` and `history.json` together. A fresh run refuses to overwrite
an existing checkpoint directory; use `--output NEW_DIRECTORY`.
Checkpoints from the earlier individual-request training layout remain loadable
for inference with their original 61-day horizon, but cannot resume grouped-context
training.

## Final refit and forecast

After choosing a validation model, execute these commands explicitly:

```bash
python -m tram_forecast prepare --regime final
python -m tram_forecast refit --selected-checkpoint outputs/runs/validation/full/best.pt --artifact outputs/prepared/final
python -m tram_forecast predict --checkpoint outputs/runs/final/full/final.pt --artifact outputs/prepared/final --template outputs/template_7days.csv --output outputs/forecast_7days.csv
```

Refit rebuilds mappings/scaling for January–October and starts fresh weights.
By default it uses the best validation epoch; pass `--epochs N` to use a chosen
count, such as the total completed validation epochs before early stopping.
Forecasting reuses each route's
October 11–31 encoding for all requested days. For the default seven-day run,
create `outputs/template_7days.csv` by filtering the supplied template to
November 1–7. It must contain all ten routes and 24 hours (1,680 unique keys),
with the original `route;date;hour;prediction` header. Template order is preserved;
predictions are nonnegative floating-point values with recorded provenance.

The competition requires all 61 days (14,640 keys). To produce that submission,
first train/select with `training.forecast_days=61`, refit that checkpoint, then
use the full `dataset/test_submission.csv` template. A seven-day output is an
initial experiment, not a complete competition submission.
November–December hidden labels are never used.

## Model API

`ForecastNetwork` receives original-unit `counts [B,1,504]`, historical
`calendar [B,6,504]`, and (full model only) `hours`: `B*504` chronological int64
`[events,5]` tensors. Here B counts contexts, not requested days. Empty hours use
`[0,5]`. Ragged IDs may remain on CPU; floating inputs/request tensors must share
the model device. For R total requested days, request fields are
`context_indices [R]`, `route_indices [R]`, `calendar [R,4]`, `lead [R]`.
Context indices select from the B encoded histories. Outputs and flattened targets
are `[R,24]` in original boarding units. `ForecastDataset(..., forecast_days=7)`
and `collate_samples` build these grouped inputs. Omitting `context_indices`
preserves the one-request-per-context API.

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

## Forecast service

A boarding-only checkpoint is served by two containers. `runner` executes the network
with ONNX Runtime, without PyTorch; `server` is the Go API with the dispatcher UI.

```bash
docker compose up -d --build   # http://127.0.0.1:8090
```

The build expects `outputs/boarding_only_61days_v5_final/final.pt` (override with
`--build-arg CHECKPOINT=...`) and `dataset/labels/labels_day_test.csv`, the source of the
October 11-31 history. The Docker build runs `python -m tram_forecast export-onnx`; the
runner image starts `python -m tram_forecast serve`. Set `STAND_AUTH=user:pass` to enable
basic auth. Details: [bench/go-inference-stand/README.md](bench/go-inference-stand/README.md).
