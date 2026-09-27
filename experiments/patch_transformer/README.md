# Seasonal patch Transformer experiment

This branch adds a daily-patch Transformer and a runnable training, comparison,
selection and final-refit workflow. It uses the existing labels, Russian calendar
and advance-known event schedule. It does not require additional ML dependencies.

## Measured result

History is frozen at **September 1, 2025** for every one of the following
61 forecast days. January–August supplies all training targets; September–October
is used only for checkpoint and blend selection. Score is `max(0, 1 - WAPE)`.

| Candidate | Validation score |
|---|---:|
| Two-week seasonal median | 0.8133 |
| Existing CNN, retrained under the same evaluation protocol | 0.8175 |
| New 28-day patch Transformer, seed 123 | 0.8340 |
| 75% patch Transformer + 25% CNN | **0.8366** |

The selected blend reduces WAPE by approximately **10.5%** relative to this CNN
run. These are model-selection results on one held-out period, not an unbiased
estimate after tuning and not the hidden November–December competition score.
Seed variation was material: the other 28-day Transformer seed scored 0.8046.
An equal-weight seed ensemble was worse than the best single seed. Shortening
the context to 14 days scored at most 0.8067, so the 28-day model was retained.
The checked-in [results.json](results.json) records both comparisons and data
fingerprints; full per-route breakdowns are in each output run's `results.json`.

## Architecture

`src/tram_forecast/patch_model.py` implements `PatchForecastNetwork`:

- One token per day contains its 24 hourly counts and calendar/event summaries.
- History-only local scaling handles differences in route volume. Two pre-norm
  Transformer layers process 28 tokens of width 48 with four attention heads.
- Route, requested-day calendar/events and forecast lead form a cross-attention
  query. The model directly predicts all 24 hours of that day.
- A same-weekday median profile across the observed weeks supplies the starting
  prediction. A zero-initialized residual head learns bounded corrections in
  normalized `log1p` space, with nonnegative outputs.
- Contexts are encoded once and reused across requested days. No predictions or
  future observations are fed back into history.

This is a compact PatchTST-inspired model with a calendar-conditioned decoder,
not an implementation of TFT or a pretrained foundation model. Its small size
is intentional for ten months of data and the available CPU: 65,720 parameters
versus 531,804 in the existing CNN.

## Reproduce

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python experiments/patch_transformer/run.py \
  --output outputs/patch_transformer_reproduction \
  --epochs 18 --patience 5 --seeds 67 123 --refit
```

The script caches features, samples up to 12 distinct future days per training
context, includes partial horizons through the last available training day,
and minimizes original-count MAE (divided by a constant for stable gradients).
It selects epochs using aggregate fixed-cutoff WAPE. AdamW uses learning rate
0.001, weight decay 0.01, gradient clipping 1, and batch size 32. Seeds are set for
Python, NumPy and PyTorch. CPU training uses four threads; `--device cuda` is also
supported when a suitable PyTorch build is installed.

The comparison includes two Transformer seeds, one CNN seed, 2/4/8-week seasonal
medians, and a small grid of blend weights. The best candidate is refitted from
scratch on January–October for its selected epoch count. Blend weights and epoch
counts remain fixed during refit. Route 5 receives zero; full-grid scores include
its actual counts. The 14,640-row submission covers all ten routes and every hour
of November–December. Output directories must be fresh to prevent accidental
mixing of runs.

To regenerate selection and refit from this session's saved validation checkpoints:

```bash
.venv/bin/python experiments/patch_transformer/run.py \
  --output outputs/patch_transformer_trial \
  --epochs 18 --patience 5 --seeds 67 123 --reuse-checkpoints --refit
```

Saved artifacts include configuration, best validation checkpoints, per-epoch
training logs, validation predictions, metrics, final checkpoints,
`submission_metadata.json` with the exact blend recipe, and `submission.csv`.
Large artifacts remain under the repository's ignored `outputs/` directory.

## Python inference

The existing `load_model` and `InferenceRunner` support the new checkpoint and
read its history length automatically:

```python
from datetime import date
from tram_forecast.train import load_model
from tram_forecast.runner import InferenceRunner

model = load_model("outputs/patch_transformer_trial/final_patch_123.pt")
runner = InferenceRunner(model)
# boardings must contain observed January–October counts and the event schedule.
context = runner.apply_context(boardings, route=1, cutoff=date(2025, 11, 1))
hourly_counts = runner.predict_day(context, date(2025, 12, 31))
```

This example predicts the Transformer component. The submitted forecast also
includes the CNN with the weights recorded in `submission_metadata.json`.
The legacy Go head exporter does not support attention checkpoints.

## Verification and existing repository limitations

```bash
.venv/bin/python -m pytest -q --confcutdir=experiments experiments/patch_transformer
.venv/bin/python -m ruff check src/tram_forecast/patch_model.py experiments/patch_transformer
```

Checks cover cached-feature equivalence, partial target horizons, the exact
seasonal initialization, grouped inference, attention gradients, finite positive
outputs, training cutoff boundaries, independence from held-out labels,
checkpoint round-trips and the existing inference runner.

The repository's pre-existing `tests/conftest.py` imports missing
`tram_forecast.config` and `prepare` APIs, so the original test suite cannot
collect. Several README CLI commands also refer to modules absent in this
checkout. Use the standalone experiment command above; the new tests are
isolated from the stale conftest. The original CNN notebook/training API is
retained, including its existing rolling-validation behavior; the corrected
fixed-cutoff selection is implemented in this experiment.
