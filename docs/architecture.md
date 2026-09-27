# Boarding CNN with calendar and scheduled events

This page describes the original direct CNN. For the added daily-patch Transformer
and the selected direct/recursive blend, see
[Seasonal patch Transformer and recursive CNN blend](patch_transformer.md).
The no-feedback statements below apply to the direct CNN path only.

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
missing calendars, malformed overrides, duplicate dates, or mismatched years
raise errors. Regional holidays are excluded. Calendar feature names are defined
in `data.py`; the model derives its calendar input sizes from those names.
Lead is `(lead−1)/60`, with integer leads 1–61.
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
ignored entirely. Missing rows, invalid binary flags, and duplicate keys raise
errors. Inference uses `sample(..., include_target=False)` and never reads future
boarding labels. The model API and forecast calling pattern are unchanged.

 Validation history ends September 1;
final-refit history ends November 1. Future labels never enter history. Route 5
uses an external zero fallback. Forecasts are never fed back as observations.

Training retains MAE, AdamW, requested-day-weighted accumulation, validation WAPE
and epoch-boundary resume. Run settings live in `configs/default.json`; layer
changes belong in `model.py`, not in configuration.

Adding calendar and event features changes the first convolution and the head input
dimensions. Checkpoints trained without it need fresh training. The legacy Go
service and head exporter still assume four target-calendar features and need
updating before they can serve models with this feature.
