# Boarding CNN with calendar and scheduled events

The architecture is fixed and written directly in `src/tram_forecast/model.py`.
There are no parallel branches, dilation choices, configurable stage counts or
adaptive-pooling options.

| Layer | Output per context |
|---|---|
| Boarding counts / scale + nine calendar and three event channels | 13 × 336 |
| Conv1d(13, 40, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 40 × 168 |
| Conv1d(40, 40, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 40 × 84 |
| Conv1d(40, 20, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 20 × 42 |
| Flatten | 840 |
| Append route embedding (8), target calendar and daily events (10), lead (1) | 859 |
| Linear(859, 320), GELU, dropout(0.1), Linear(320, 320), GELU, dropout(0.1), Linear(320, 24) | 24 |
| Softplus × count scale | 24 nonnegative boarding predictions |

Convolutions use padding to preserve length before pooling. History is always
14 days. Its calendar channels encode sine/cosine of hour, weekday and day of month,
followed by binary `is_holiday`, `is_day_off`, and `is_short_working_day`.
Target calendar omits the two hour channels and retains all three flags, giving
seven features per requested day. Public holidays and transferred days off are
holidays; ordinary weekends are day-off dates only. Explicit XML overrides handle
working Saturdays and shortened working days before the usual weekend rule.

Calendar overrides come from bundled `calendars/<year>.xml` files, cached by year.
The supplied 2025 XML follows Resolution No. 1335 of 4 October 2024
(https://government.ru/docs/52895/). Additional years require their XML files;
training with missing calendars, malformed overrides, duplicate dates, or mismatched
years raises errors. During inference, a missing year uses zero work-calendar
flags. Regional holidays are excluded. Calendar feature names are defined
in `data.py`; the model derives its calendar input sizes from those names.
Lead is `(lead−1)/60`; training uses integer leads 1–61. Direct inference also
accepts later dates, although those leads are outside the training horizon.
Count scale is `max(1, mean fitting-period counts)`.

`ForecastNetwork(scale)` accepts `counts [B,1,336]` and `calendar [B,12,336]`.
A batch contains B route/cutoff contexts and R requested days. Request
`context_indices [R]` select their shared encoded histories; route indices, target
calendar and leads condition the head. Output and targets are `[R,24]`.

Boarding counts come from label CSVs. `Boardings.load(..., events_path=...)` also
loads `parking_events_hourly.csv`. Its full schedule is retained independently of
label splits and cached across stores. `sample()` assembles all inputs automatically:
three event channels follow the nine history-calendar channels; the seven daily
request-calendar features are followed by three daily event flags, each reduced
with maximum over the target day's 24 hours. A flag is 1 if active at any hour.
The event order is `extended_night_service`, `event_near_route`, `is_citywide_event`.
Operational columns `has_route_change`, `parking_pct_route`, and `has_parking` are
ignored entirely. Training rejects missing rows, invalid binary flags, and
duplicate keys. The Python inference runner uses zero flags for missing event
rows or schedules and never reads future boarding labels. The model weights and
architecture are unchanged.

Validation history ends September 1;
final-refit history ends November 1. Future labels never enter history. Route 5
uses an external zero fallback. The direct forecast never feeds predictions back
as observations; the optional autoregressive function uses generated counts.

## Autoregressive inference

`InferenceRunner.apply_context` encodes the last 14 days of observed counts.
For each next day, `predict_autoregressive` in `predict.py` calls the normal
`predict_day` handle, then passes those 24 predicted counts to
`InferenceRunner.update_context`. The update drops the oldest day and returns a
new context. After 14 updates, the entire history is generated rather than
observed. The original context remains unchanged. The target calendar and event
flags are obtained for each date; missing binary flags are zero during inference.
See [the inference guide](inference_example.md) for a code example.

This can help on longer forecasts by giving the CNN a recent rolling pattern
instead of asking it to make every prediction from the same 14 observed days.
The measured benefit was **mild and inconsistent** in two 61-day backtests:

| Forecast | July–August score | September–October score | Pooled score |
|---|---:|---:|---:|
| Direct CNN | 0.7955 | 0.8175 | — |
| Recursive CNN | 0.7672 | 0.8553 | — |
| Previous direct Transformer/CNN blend | 0.8373 | 0.8366 | 0.8369 |
| 50/50 direct Transformer/recursive CNN blend | 0.8297 | 0.8543 | 0.8432 |

The blended pooled score improved by 0.0063, while pure CNN recursion helped
September–October and hurt July–August. The user also reported only about
**+0.003** on the hidden two-month submission; the exact score and comparison
submission were not supplied. These results support offering autoregression as
an option, not making it a required default.

For a new route or period, compare direct and recursive forecasts on a held-out
window if labels are available. Recursive errors can feed into later days, so a
better 61-day backtest does not guarantee a better forecast elsewhere. Accuracy
for **sequences longer than three months is untested**: no suitable observed
long-horizon data was available for that experiment. The API can run longer,
but the score at those lengths is unknown.

Training retains MAE, AdamW, requested-day-weighted accumulation, validation WAPE
and epoch-boundary resume. Run settings live in `configs/default.json`; layer
changes belong in `model.py`, not in configuration.

Adding calendar and event features changes the first convolution and the head input
dimensions. Checkpoints trained without it need fresh training. The legacy Go
service and head exporter still assume four target-calendar features and need
updating before they can serve models with this feature.
