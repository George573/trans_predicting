# Boarding-only CNN

The architecture is fixed and written directly in `src/tram_forecast/model.py`.
There are no parallel branches, dilation choices, configurable stage counts or
adaptive-pooling options.

| Layer | Output per context |
|---|---|
| Boarding counts / scale + six calendar channels | 7 × 504 |
| Conv1d(7, 32, kernel=5), GELU, MaxPool1d(2) | 32 × 252 |
| Conv1d(32, 32, kernel=5), GELU, MaxPool1d(2) | 32 × 126 |
| Conv1d(32, 16, kernel=3), GELU, MaxPool1d(3) | 16 × 42 |
| Flatten | 672 |
| Append route embedding (8), target calendar (4), lead (1) | 685 |
| Linear(685, 250), GELU, dropout(0.1), Linear(250, 24) | 24 |
| Softplus × count scale | 24 nonnegative boarding predictions |

Convolutions use padding to preserve length before pooling. History is always
21 days. Its calendar channels encode sine/cosine of hour, weekday and day of month.
Target calendar omits hour. Lead is `(lead−1)/60`, with integer leads 1–61.
Count scale is `max(1, mean fitting-period counts)`.

`ForecastNetwork(scale)` accepts `counts [B,1,504]` and `calendar [B,6,504]`.
A batch contains B route/cutoff contexts and R requested days. Request
`context_indices [R]` select their shared encoded histories; route indices, target
calendar and leads condition the head. Output and targets are `[R,24]`.

Data comes only from hourly boarding label CSVs. Validation history ends September 1;
final-refit history ends November 1. Future labels never enter history. Route 5
uses an external zero fallback. Forecasts are never fed back as observations.

Training retains MAE, AdamW, requested-day-weighted accumulation, validation WAPE
and epoch-boundary resume. Run settings live in `configs/default.json`; layer
changes belong in `model.py`, not in configuration.

The fixed CNN uses checkpoint format 3 and needs fresh training. Previous
multiscale checkpoints cannot be loaded, resumed or exported by this version.
Prepared boarding artifacts remain format 2 and can be reused. The Go service
uses a newly exported head; its request-time computation is unchanged.
