# trans_predicting
Mega ai transport planning solution 67

## Streaming boarding windows

`hourly_windows.py` samples fixed length route-hour histories from the provided
hourly labels without loading either CSV into memory. It reads rows in their
existing route/date/hour order, merges train and test when both are supplied,
and fills absent hours with zero. Each filled hour has `observed=False` so it
can be distinguished from a row present in the source. A window never crosses
from one route into another.

To inspect two random training windows with eight history hours and a target
one hour ahead:

```bash
python hourly_windows.py dataset/labels/labels_day_train.csv \
  --history-hours 8 --horizon-hours 1 --sample-size 2
```

To inspect a September target whose history can extend into August:

```bash
python hourly_windows.py \
  dataset/labels/labels_day_train.csv \
  dataset/labels/labels_day_test.csv \
  --series-end 2025-10-31 --target-start 2025-09-01 \
  --target-end 2025-09-30 --history-hours 64 --sample-size 1
```

The Python API exposes `iter_boarding_windows(...)` for sequential training
and `sample_boarding_windows(..., sample_size=..., seed=...)` for uniform random
sampling in one pass. Memory use is proportional to the history and horizon
lengths, plus the number of retained samples. `horizon_hours=1` trains a
one-hour-ahead target. Longer horizons are available, but validation must match
the intended November–December forecasting setup: using observed September
history to predict the next September hour measures one-step forecasting, not
an uninterrupted two-month forecast.

Route 5 has no rows in the supplied labels, so the pipeline does not create
training windows for it. The current pipeline uses the aggregated labels;
the much larger raw validation CSVs can be streamed separately when additional
event-level features are needed.

## Full data profile

The profiler reads the full raw transaction files, hourly labels, and
submission template. It writes `outputs/data_profile/profile.json`, a concise
`summary.md`, and compact CSV detail tables. It does not train a model.
In a terminal it shows a progress bar for each raw file, using file bytes for
a full scan. Without a terminal it prints periodic bar updates as log lines.

```bash
python -u tools/profile_dataset.py \
  --raw-train dataset/train.csv \
  --raw-validation dataset/test.csv \
  --labels-train dataset/labels/labels_day_train.csv \
  --labels-validation dataset/labels/labels_day_test.csv \
  --submission dataset/test_submission.csv \
  --output-dir outputs/data_profile \
  --temp-dir outputs/data_profile/tmp \
  --memory-limit-gb 12 --sort-buffer 2G --seed 67
```

For a quick inspection of only the first 100,000 raw rows from **each** file:

```bash
python -u tools/profile_dataset.py \
  --max-raw-rows-per-file 100000 \
  --progress-every 10000 \
  --skip-exact-duplicates --skip-sequence-sort \
  --output-dir outputs/data_profile_quick \
  --temp-dir outputs/data_profile_quick/tmp
```

The row limit is a deterministic file prefix, not a random sample. The hourly
labels are still scanned in full. The quick report marks raw statistics as
partial and does not interpret raw-to-label reconciliation as a full-data
result. Increase the limit to inspect a larger prefix. Use a separate output
directory for each run so results do not overwrite one another.

Input paths, period boundaries, reservoir size, memory limit, temporary
directory, sort buffer, random seed, and progress interval are configurable;
see `python tools/profile_dataset.py --help`. The raw scan keeps only compact
route-hour aggregates and categorical counters in memory. Exact row-duplicate
and sorted event-order checks use external sort and need several gigabytes of
working disk. Use `--skip-exact-duplicates` or `--skip-sequence-sort` if that
disk work is unavailable; the report will mark those checks as skipped.

Missing label rows are zero-filled on the hourly grid, with their original
presence recorded separately. Numerical grid distributions are exact. Delay
and inter-event-gap quantiles use fixed-size reservoir samples and linear
interpolation; the random seed and sample size are recorded in the report.

Check a finished report and the headers of its detail tables with:

```bash
python tools/check_profile.py outputs/data_profile/profile.json
```

## Forecast package: implemented core

The `src/tram_forecast` package currently implements validated model settings,
source parsing/calendar functions, daily sample indexing, the complete multiscale
network, and exact chunked/checkpointed event encoding. Every convolutional stage
has parallel kernels with different lengths and dilations. See
`implementation_progress.md` for completed and pending work.

Create an environment and install the package:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m tram_forecast check-config --config configs/default.json
.venv/bin/python -m pytest -q
```

For a CPU-only PyTorch installation, install `torch` from
`https://download.pytorch.org/whl/cpu` before installing the package. The core was
verified with Python 3.14 and PyTorch 2.14.0+cpu. GPU and real-data memory checks
have not been run. Preprocessing/training/evaluation/submission CLI commands
currently fail explicitly as pending integration; they do not run a pipeline.

`ForecastNetwork` accepts a history dictionary with original-unit `counts`
`[B,1,504]`, historical `calendar` `[B,6,504]`, and `hours`: a list of `B*504`
chronologically aligned int64 tensors `[number_of_events,5]`. Empty hours use
`[0,5]` tensors. Category IDs must come from training-only fitted vocabularies.
Request dictionaries contain `route_indices` `[B]` (1–9 in configured route order),
`calendar` `[B,4]`, and `lead` `[B]` (1–61). Floating inputs must share the model's
device. Outputs are nonnegative original-unit counts `[B,24]`.

Raw datasets/excerpts, generated outputs/checkpoints, the exploratory notebook,
and superseded PDF designs remain local and are ignored by Git. Dataset rules
are retained in `dataset/README.md`; the Markdown implementation plan is the
source of truth for the model.
