# Boarding-only CNN

The architecture is fixed and written directly in `src/tram_forecast/model.py`.
There are no parallel branches, dilation choices, configurable stage counts or
adaptive-pooling options.

| Layer | Output per context |
|---|---|
| Boarding counts / scale + seven calendar channels | 8 × 336 |
| Conv1d(8, 40, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 40 × 168 |
| Conv1d(40, 40, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 40 × 84 |
| Conv1d(40, 20, kernel=50), GELU, MaxPool1d(2), dropout(0.2) | 20 × 42 |
| Flatten | 840 |
| Append route embedding (8), target calendar (5), lead (1) | 854 |
| Linear(854, 320), GELU, dropout(0.1), Linear(320, 320), GELU, dropout(0.1), Linear(320, 24) | 24 |
| Softplus × count scale | 24 nonnegative boarding predictions |

Convolutions use padding to preserve length before pooling. History is always
14 days. Its calendar channels encode sine/cosine of hour, weekday and day of month,
followed by binary `is_holiday`. Target calendar omits the two hour channels and
retains the holiday flag. Russian federal public holidays and transferred days off
are marked 1; ordinary weekends are not. The calendar covers 2025, including
transfers under Resolution No. 1335 of 4 October 2024
(https://government.ru/docs/52895/). Unsupported years raise an error; regional
holidays are excluded. Lead is `(lead−1)/60`, with integer leads 1–61.
Count scale is `max(1, mean fitting-period counts)`.

`ForecastNetwork(scale)` accepts `counts [B,1,336]` and `calendar [B,7,336]`.
A batch contains B route/cutoff contexts and R requested days. Request
`context_indices [R]` select their shared encoded histories; route indices, target
calendar and leads condition the head. Output and targets are `[R,24]`.

Data comes only from hourly boarding label CSVs. Validation history ends September 1;
final-refit history ends November 1. Future labels never enter history. Route 5
uses an external zero fallback. Forecasts are never fed back as observations.

Training retains MAE, AdamW, requested-day-weighted accumulation, validation WAPE
and epoch-boundary resume. Run settings live in `configs/default.json`; layer
changes belong in `model.py`, not in configuration.

Adding the holiday feature changes the first convolution and the head input
dimensions. Checkpoints trained without it need fresh training. The legacy Go
service and head exporter still assume four target-calendar features and need
updating before they can serve models with this feature.
