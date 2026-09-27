# Tram ridership forecasting

A boarding-only PyTorch CNN that encodes a route's hourly history once and predicts
24 boarding counts for each requested future day. Calendar features, route identity
and forecast lead condition the predictions. The model uses 14 days of history and the
training forecast horizon is seven days, configurable up to 61. The Python
inference API can predict later dates directly or advance the CNN one day at
a time with generated counts; see [the runtime example](docs/inference_example.md).

Calendar inputs include three binary Russian production-calendar flags:
`is_holiday` (public holidays and transferred days off), `is_day_off` (including
ordinary weekends, with working-Saturday overrides), and `is_short_working_day`.
The XML source is bundled in `src/tram_forecast/calendars/2025.xml`, based on the
example calendar. Additional years can be supported by adding a validated
`calendars/<year>.xml` file. Missing years raise an error; regional holidays are
excluded. XML files are cached per year and included in installed packages.
All three flags apply to hourly history and every requested forecast day.
Models trained with the previous calendar inputs must be retrained.

Load training data with the scheduled-event source:

```python
boardings = Boardings.load(
    labels_path,
    events_path="dataset/parking_events_hourly.csv",
    start=start,
    end=end,
)
```

Sampling automatically combines calendar features with the advance-known flags
`extended_night_service`, `event_near_route`, and `is_citywide_event`. History gets
three hourly channels; each target day gets three binary flags, reduced with
maximum over its 24 hours (active at any hour means 1).
Route-change and parking columns are excluded entirely. The full event schedule
is kept independently of label dates and cached across loads. Missing schedule rows
raise errors. Forecasting does not require future boarding labels. The training
notebook supplies the source path automatically. Retrain using a fresh checkpoint
name after adding these inputs.

## Project layout

For the current model's context/day inference API, see
[the Python runner example](docs/inference_example.md) and
[executable demonstration](examples/inference.py).

| Path | Purpose |
|---|---|
| `src/tram_forecast/config.py` | Data and training settings |
| `src/tram_forecast/data.py` | Label parsing, artifact preparation, history storage and batches |
| `src/tram_forecast/model.py` | Fixed three-layer Conv1d encoder and prediction head |
| `src/tram_forecast/train.py` | Training, refitting, loss and progress display |
| `src/tram_forecast/evaluate.py`, `predict.py` | Metrics, weekly baseline and forecast CSVs |
| `src/tram_forecast/checkpoint.py`, `head_export.py` | Checkpoint loading/saving and Go head export |
| `src/tram_forecast/cli.py`, `smoke.py` | Command-line workflows and resource checks |
| `src/tram_forecast/io.py` | Shared atomic file writing and fingerprints |
| `configs/` | One run configuration |
| `notebooks/train.ipynb` | Interactive training, evaluation and final refit |
| `service/` | Go forecast API, dispatcher UI and load tool |
| `tests/` | Synthetic correctness and workflow tests |
| `tools/` | Optional CUDA installer |
| `docs/architecture.md` | Model graph, data contracts and checkpoint compatibility |
| `dataset/`, `outputs/` | Local data and generated artifacts; ignored by Git |

The model architecture is explicit in `model.py`: three Conv1d/GELU/pooling stages,
then a prediction head. There is no architecture configuration or experiment suite.
Calendar, route and lead inputs are retained. Tests cover the core data and forecast
workflow rather than combinations of architecture options.

## Setup

Python 3.10+:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,notebook]'
.venv/bin/python -m pytest -q
```

For the GTX 1080/Pascal environment, the existing installer selects the CUDA 12.6
wheel before installing the package:

```bash
.venv/bin/python tools/install.py --legacy-cuda --notebook --test
```

Use `.venv\Scripts\python.exe` on Windows. The examples below assume the virtual
environment is activated. Set `training.device` or pass `--device cuda` to use a GPU.

## Train and forecast

Only these source files are required:

- `dataset/labels/labels_day_train.csv`
- `dataset/labels/labels_day_test.csv`

They contain `route;date;hour;boardings`. Set other paths in `configs/default.json`.
Raw transaction CSVs are not required. [Dataset rules](dataset/README.md) describe
the competition data and submission format.

```bash
python -m tram_forecast check-config --config configs/default.json
python -m tram_forecast prepare --regime validation --artifact outputs/prepared/validation_boardings
python -m tram_forecast train --artifact outputs/prepared/validation_boardings
python -m tram_forecast evaluate --checkpoint outputs/runs/validation/boarding_only/best.pt --artifact outputs/prepared/validation_boardings --output outputs/metrics.json
```

Or open [notebooks/train.ipynb](notebooks/train.ipynb). Its configuration cell
currently selects a 61-day horizon; the CLI defaults to seven days. Training and
final-refit cells perform real optimization. Use a fresh output directory for each
experiment, or `--resume PATH/latest.pt` to resume at an epoch boundary.

Each route/cutoff context is encoded once for all requested days. Training uses
original-unit MAE, AdamW, requested-day-weighted accumulation and validation WAPE
for early stopping. Validation freezes history at September 1. Route 5 uses a zero
fallback. The default direct forecast does not feed predictions back into
history. The optional autoregressive prediction function uses generated history.

After selecting a validation checkpoint:

```bash
python -m tram_forecast prepare --regime final --artifact outputs/prepared/final_boardings
python -m tram_forecast refit --selected-checkpoint outputs/runs/validation/boarding_only/best.pt --artifact outputs/prepared/final_boardings
python -m tram_forecast predict --checkpoint outputs/runs/final/boarding_only/final.pt --artifact outputs/prepared/final_boardings --template outputs/template_7days.csv --output outputs/forecast.csv
```

The seven-day template must contain November 1–7 for all ten routes and 24 hours.
For the complete 14,640-row competition submission, train/select with
`training.forecast_days=61`, refit, then use `dataset/test_submission.csv`.
November–December labels are unavailable and never used locally.

Prepared artifacts are reused only when their source fingerprints and settings
match. Rebuild old event-based artifacts in a new directory. The simplified CNN uses checkpoint format 3 and requires fresh training;
previous multiscale checkpoints cannot be resumed or exported by this version. See [architecture and compatibility](docs/architecture.md).

To check forward/backward memory without training:

```bash
python -m tram_forecast smoke --artifact outputs/prepared/validation_boardings --output outputs/smoke.json
```

## Forecast service

```bash
docker compose up -d --build   # http://127.0.0.1:8090
```

The image build exports a boarding-only checkpoint into a Go prediction head.
The default checkpoint is `outputs/runs/final/boarding_only/final.pt`; override
with `--build-arg CHECKPOINT=...`. The Go server validates reference predictions
before serving. Set `STAND_AUTH=user:pass` for basic auth.
See [service/README.md](service/README.md) for details.
