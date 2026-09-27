# Python inference runner example

This example runs the current PyTorch checkpoint directly. It illustrates a
context/day API; it is not an HTTP service or a replacement for the legacy Go
runner. It does not use the legacy head exporter.

```python
from datetime import date
from tram_forecast.data import Boardings
from tram_forecast.predict import predict_autoregressive
from tram_forecast.runner import InferenceRunner
from tram_forecast.train import load_model

boardings = Boardings.load(
    "dataset/labels/labels_day_test.csv",
    events_path="dataset/parking_events_hourly.csv",
    routes=(7,),
    start=date(2025, 10, 18),
    end=date(2025, 11, 1),
)
runner = InferenceRunner(load_model("outputs/model.pt", device="cpu"))

context = runner.apply_context(boardings, route=7, cutoff=date(2025, 11, 1))
november_1 = runner.predict_day(context, date(2025, 11, 1))
november_10 = runner.predict_day(context, date(2025, 11, 10))
december_31_recursive = predict_autoregressive(runner, context, date(2025, 12, 31))

# Advance the context manually with a completed day's counts.
next_context = runner.update_context(context, date(2025, 11, 1), november_1)
november_2 = runner.predict_day(next_context, date(2025, 11, 2))

# Equivalent single call, encoding a fresh context:
november_10 = runner.predict(
    boardings, route=7, cutoff=date(2025, 11, 1), day=date(2025, 11, 10),
)
```

| Method | Input | Output |
|---|---|---|
| `apply_context(boardings, route, cutoff)` | Data store, route ID, first forecast date | Reusable `ForecastContext` |
| `predict_day(context, day)` | Context and target date | NumPy `float32[24]` |
| `predict(boardings, route, cutoff, day)` | Context inputs and target date | NumPy `float32[24]` |
| `update_context(context, day, counts)` | Context, day at its cutoff, 24 completed counts | New `ForecastContext` |
| `predict_autoregressive(runner, context, day)` | Runner, context and distant target date | NumPy `float32[24]` |

`apply_context` encodes exactly 14 days of history, ending immediately before the
cutoff. It does not require future labels or future event rows. The context retains
the supplied event schedule; work-calendar flags are read from the bundled calendar
when each day is requested. It can be reused for multiple days, even after another
context is created. It belongs to the runner that created it.
It rejects non-finite or negative counts in the initial history.

`update_context` adds one complete day of observed or predicted counts at the
context cutoff, drops the oldest day, and returns a new context for the next
day. The original context remains usable. It rejects another runner's context,
a day other than the cutoff, and counts that are not 24 finite nonnegative values.

`predict_day` supports any date on or after the cutoff, including dates in later
years. It prepares the target date's ten features: seven calendar values and
three event flags reduced with maximum over that day's 24 hours. Known production
calendar and scheduled-event values are used. Missing year calendars or missing
event hours, dates, routes and files contribute zero to those binary features;
the date's cyclic weekday and month-day features are still computed. Predictions
are in boarding-count units, indexed by hour 0 through 23.
The checkpoint's scale is used; no caller normalization is needed. Predictions
beyond the 61-day training horizon are supported by the API but have not been
validated for accuracy.

`predict_autoregressive` lives in `tram_forecast.predict` and starts from a
context made by `apply_context`. It calls
the same `predict_day` method for each day through the requested date, then
uses `update_context` to advance the rolling 14-day history. It returns the final
day. The original context and boarding data stay unchanged. It uses zero binary
flags where calendar or event information is missing. It never reads future
boarding labels. Calling it repeatedly for many distant
dates repeats the intervening computation.

Use the direct `predict_day` call as the usual starting point. Autoregression is
optional for longer sequences. It gave a small improvement in a two-month
blended forecast, but the CNN alone did not improve on every historical period.
The reported hidden-score gain was about +0.003, with no exact comparison score
available. Forecasts **longer than three months have not been evaluated** because
there was no suitable observed period for that test. At those lengths, each new
prediction depends on a growing chain of earlier predictions, so prefer a
backtest for the intended horizon before relying on the result.

Supported route IDs are 1, 7, 11, 12, 17, 25, 26, 28, and 50. Operational parking
and route-change features are excluded. Use a checkpoint trained with the current
daily-event feature schema. Treat model weights and context tensors as read-only;
reload the runner after replacing a model. New event files require a new store
and context. The bundled production calendar currently covers 2025; later years
default to zero work-calendar flags unless their `calendars/<year>.xml` is added.
The training data preparation remains strict about missing calendar and event
inputs. The 14 days of boarding counts needed to initialize a context remain
required for inference.

Executable demonstration:

```bash
python examples/inference.py \
  --checkpoint outputs/model.pt \
  --labels dataset/labels/labels_day_test.csv \
  --events dataset/parking_events_hourly.csv \
  --route 7 --cutoff 2025-11-01 --day 2025-11-10
```

Add `--autoregressive` to compare the recursive forecast with the normal direct
prediction for the requested day. `--events` is optional for inference; omit it
to use zero event flags throughout.
