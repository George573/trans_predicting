# Tram ridership forecasting

A PyTorch package that encodes 21 days of observed route history once and predicts
24 hourly boarding counts for each requested future day. The initial horizon is
seven days for the neural model and the weekly-profile baseline. Every convolutional stage uses parallel
kernel lengths and dilations. The model uses boarding history and calendar features, with route and forecast-lead
conditioning. A weekly-profile baseline is also provided.

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
| `bench/go-inference-stand/` | Go API and dispatcher UI that compute the CNN head in-process |
| `Dockerfile`, `docker-compose.yml` | Stand image; the checkpoint head is exported for Go at build time |

[Implementation specification](docs/architecture/architecture_implementation_plan.md)
· [Steps](docs/development/implementation_steps.md)
· [Progress](docs/development/implementation_progress.md)

## Install and verify

Python 3.10+; verified locally with Python 3.14, PyTorch 2.14.0+cpu.

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
epochs, output directory and resume checkpoint. The notebook uses boarding-only inputs and currently selects a 61-day horizon;
the CLI defaults to seven days. CUDA is selected when available. Run cells in order
to prepare data, inspect the weekly baseline, check memory, train, and plot
learning curves and forecasts. The Train cell performs actual optimization. Training shows a live `tqdm` batch progress
bar with running MAE, optimizer steps, speed and epoch ETA, followed by
validation and checkpoint status. Epoch summaries show current and best WAPE
and early-stopping patience. The same feedback appears in CLI training and final
refits. Notebook installs include `ipywidgets` for the widget progress bar. Interrupt with Ctrl+C or the notebook stop button, then
resume from `latest.pt`; the incomplete epoch is rerun.
Change the run name for a fresh experiment, or set `RESUME` to `latest.pt` to
continue the same run. The notebook is saved without outputs or trained weights.

## Prepare data

Required local files: `dataset/labels/labels_day_train.csv` and
`dataset/labels/labels_day_test.csv`. Raw transaction CSVs are not required.
Configure alternative paths in `configs/default.json`.

```bash
python -m tram_forecast prepare --config configs/default.json --regime validation
python -m tram_forecast inspect --artifact outputs/prepared/validation
```

Preparation validates hourly boarding labels, fits count scaling on the fitting
period only, and atomically publishes memory-mapped count arrays. Validation
labels remain separate from history storage. Path/size/mtime and preprocessing
settings identify reusable artifacts; these are compatibility checks, not content
hashes. Changed sources require a new artifact directory (`--artifact PATH`).

The boarding-only artifact format is version 2. Rebuild old event-based artifacts
in a new directory, for example `--artifact outputs/prepared/validation_boardings`,
and pass that directory to training/evaluation. Existing artifacts are not modified.

## Resource smoke check

```bash
python -m tram_forecast smoke --artifact outputs/prepared/validation --model boarding_only --output outputs/smoke.json
```

This explicitly requested command checks typical and busiest histories with a
forward/backward pass and no optimizer step. It reports CPU process high-water
RSS and CUDA peaks where available. It tests batch size one; larger training
batches still require checking. If an OOM occurs, lower batch size and rerun. Histories have fixed length.

## Training and evaluation

```bash
python -m tram_forecast train --artifact outputs/prepared/validation --model boarding_only
python -m tram_forecast evaluate --checkpoint outputs/runs/validation/boarding_only/best.pt --artifact outputs/prepared/validation --output outputs/boarding_metrics.json
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

Checkpoints include model, optimizer, settings, boarding artifact/scaling contract,
RNG state and epoch progress. Resume from the latest checkpoint in the same run
directory; only changing the epoch limit or the configured output root
is accepted without a fresh run:

```bash
python -m tram_forecast train --artifact outputs/prepared/validation --model boarding_only --resume outputs/runs/validation/boarding_only/latest.pt
```

Epoch-boundary resume is supported. An interrupted partial epoch is rerun. Keep
`best.pt`, `latest.pt` and `history.json` together. A fresh run refuses to overwrite
an existing checkpoint directory; use `--output NEW_DIRECTORY`.
New checkpoints use format version 2. Existing boarding-only version-1 weights
remain usable for `export-head` and fresh `refit`; event settings are discarded
when reading them. Their old artifact contracts cannot resume or evaluate against
new artifacts. Full/event-stream checkpoints are rejected. No checkpoint files
are rewritten automatically.

## Final refit and forecast

After choosing a validation model, execute these commands explicitly:

```bash
python -m tram_forecast prepare --regime final
python -m tram_forecast refit --selected-checkpoint outputs/runs/validation/boarding_only/best.pt --artifact outputs/prepared/final
python -m tram_forecast predict --checkpoint outputs/runs/final/boarding_only/final.pt --artifact outputs/prepared/final --template outputs/template_7days.csv --output outputs/forecast_7days.csv
```

Refit rebuilds scaling for January–October and starts fresh weights.
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

`ForecastNetwork(scale, config=None)` receives original-unit
`counts [B,1,T]` and historical `calendar [B,6,T]`, where
`T = history_days * 24` (504 by default). Calendar inputs retain hour, weekday and
day-of-month sine/cosine features. There is no event-list input.

For R requested days, request fields are `context_indices [R]`,
`route_indices [R]`, `calendar [R,4]`, and `lead [R]`. Outputs and targets are
`[R,24]` in original boarding units. Histories are encoded once per context;
`predict_day` reuses them. Omitting `context_indices` keeps one request per context.
The default model has **195,868 parameters**, including the route embedding.

`boarding_only` remains the model identifier in run paths, checkpoints, experiment
reports and the optional CLI `--model` argument. It is the only supported model.

## Scientific architecture PDF

```bash
.venv/bin/python -m pip install -e '.[docs]'
.venv/bin/python tools/docs/build_architecture_pdf.py
```

Output: `output/pdf/tram_cnn_scientific_architecture.pdf`. The vector note describes
the boarding-only graph, convolution paths, request conditioning and data contracts.

## Forecast service

A boarding-only checkpoint is served by one container: the Go API with the dispatcher
UI, which computes the CNN head itself on every request. PyTorch runs only during the
image build.

```bash
docker compose up -d --build   # http://127.0.0.1:8090
```

The build expects `outputs/boarding_only_61days_v5_final/final.pt` (override with
`--build-arg CHECKPOINT=...`) and `dataset/labels/labels_day_test.csv`, the source of the
October 11-31 history. The Docker build runs `python -m tram_forecast export-head`, which
encodes that history, folds it with the route embedding into the first head layer and
writes `head.json` with the remaining weights and 27 reference days predicted by torch.
The server refuses to start if its own output diverges from them. Set
`STAND_AUTH=user:pass` to enable basic auth. Details: [bench/go-inference-stand/README.md](bench/go-inference-stand/README.md).
