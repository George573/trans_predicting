# Python inference runner example

This example runs the current PyTorch checkpoint directly. It illustrates a
context/day API; it is not an HTTP service or a replacement for the legacy Go
runner. It does not use the legacy head exporter.

```python
from datetime import date
from tram_forecast.data import Boardings
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

`apply_context` encodes exactly 14 days of history, ending immediately before the
cutoff. It does not require future labels or future event rows. All input
preparation uses the same helpers as training samples. The context retains the
event schedule from its source store and can be reused for multiple days, even
after another context is created. It belongs to the runner that created it.

`predict_day` supports cutoff through cutoff + 60 days, inclusive. It prepares the
target date's ten features: seven calendar values and three event flags reduced
with maximum over that day's 24 hours. Missing schedule/calendar coverage raises
an error. Predictions are in boarding-count units, indexed by hour 0 through 23.
The checkpoint's scale is used; no caller normalization is needed.

Supported route IDs are 1, 7, 11, 12, 17, 25, 26, 28, and 50. Operational parking
and route-change features are excluded. Use a checkpoint trained with the current
daily-event feature schema. Treat model weights and context tensors as read-only;
reload the runner after replacing a model. New event files require a new store
and context. A 61-day interface does not guarantee accuracy beyond the training
horizon. The bundled production calendar currently covers 2025.

Executable demonstration:

```bash
python examples/inference.py \
  --checkpoint outputs/model.pt \
  --labels dataset/labels/labels_day_test.csv \
  --events dataset/parking_events_hourly.csv \
  --route 7 --cutoff 2025-11-01 --day 2025-11-10
```
