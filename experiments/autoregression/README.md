# Autoregression for a genuinely unseen 61-day interval

The experiment compares direct prediction with forecasts that advance the cutoff
using their own predictions. Every historical evaluation freezes actual boarding
history at its initial cutoff. Only calendar and scheduled-event inputs remain
available in the future. No held-out boarding observations are supplied during
rollout, including when those labels are present in the loaded data object.

Two forecast origins are used:

- July 2 → August 31, training only through July 1.
- September 1 → October 31, training only through August 31.

Both are 61 days. The earlier origin retrains each direct architecture from
scratch for the seven epochs selected in the original experiment. This is a
retrospective robustness check with no target leakage into model fitting;
architecture, epoch counts and blend weights have been tuned using available
2025 validation data. Neither origin is a pristine final test after selection.
The exact hidden November–December score remains unknown. After submission, the
user reported only **+0.003 score improvement**, without specifying the absolute
score or comparison submission. This feedback is distinct from the local gains
below; see [the architecture documentation](../../docs/patch_transformer.md) for
the model explanation and limitations.

## Results and recommendation

The selected forecast is **50% direct Transformer + 50% daily recursive CNN**.
It maximizes pooled WAPE-score across the two historical origins in the tested
blend grid. Pure CNN recursion is too sensitive to the forecast period.

| Strategy | July–August | September–October | Pooled score |
|---|---:|---:|---:|
| Previous direct blend | 0.8373 | 0.8366 | 0.8369 |
| Pure recursive CNN | 0.7672 | 0.8553 | 0.8155 |
| 75% direct Transformer / 25% recursive CNN | 0.8378 | 0.8458 | 0.8422 |
| **50% direct Transformer / 50% recursive CNN** | **0.8297** | **0.8543** | **0.8432** |
| 25% direct Transformer / 75% recursive CNN | 0.8044 | 0.8579 | 0.8337 |

The selected blend reduces pooled WAPE by **3.85%** and September–October WAPE
by **10.83%** relative to the previous blend, at the cost of a lower July–August
score. The 75/25 version is the conservative alternative; it improves both
historical periods, but has a slightly lower pooled score. Neither result
establishes that recursion always helps or guarantees a hidden-test improvement.

Selecting the CNN checkpoint by recursive WAPE retained epoch seven. All final
components therefore reuse the existing January–October refits; the new
submission changes forecast generation and blend weights, without fitting to
November–December labels. The [results snapshot](results.json) includes the
lead breakdowns, weaker candidates and data fingerprints.

## What was tested

1. Existing direct Transformer and CNN checkpoints, advanced by one day or one
   week at a time with generated counts fed back into history.
2. Anchored recursion: combine each newly generated block 50/50 with the original
   fixed-cutoff forecast before feeding it back.
3. New Transformers trained for one-day or seven-day predictions, selected by
   fully recursive 61-day validation WAPE.
4. A Transformer trained over four recursively generated weeks. Gradients pass
   through generated history; later training steps do not receive actual labels
   as inputs. MAE is computed at each step.
5. A seasonal linear ARX model with 1/7/14/21/28-day lag profiles, calendar and
   event features, and ridge penalties 1, 10 and 100.
6. CNN checkpoint selection using recursive 61-day WAPE instead of direct WAPE.
7. Coarse blends of the direct Transformer and recursive CNN, chosen across both
   forecast origins rather than solely on the latest period.

The specialized recursive Transformers and linear models do not improve on the
best direct/recursive combination in this search. This does not rule out other
AR architectures or training settings. Pure recursion hurts the Transformer,
while the CNN's recursive improvement varies substantially by forecast origin.
See the checked-in results for the exact comparison and selected recipe.

## Implementation

`src/tram_forecast/autoregressive.py` provides `recursive_forecast`. It creates a
private state from pre-cutoff observations, predicts a block, appends predictions,
and slides the context. Relative forecast leads restart at each block. The
function never modifies the caller's observed counts and raises on non-finite
or negative generated counts. It supports the existing CNN and patch Transformer.

The selected hybrid combines independently generated direct and recursive
trajectories. The direct Transformer contribution is **not** fed back into the
CNN. This differs from the separately tested anchored-recursion candidate.

```python
from datetime import date
from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.train import load_model

cnn = load_model("outputs/patch_transformer_trial/final_cnn_67.pt")
# boardings contains January–October observations and the known event schedule.
counts = recursive_forecast(cnn, boardings, date(2025, 11, 1), days=61, block_days=1)
```

## Reproduce

To regenerate the final November–December files from the already trained models:

```bash
.venv/bin/python experiments/autoregression/export_submission.py
```

This writes `outputs/submission_nov_dec.csv` in the notebook's official
`route;date;hour;prediction` format, preserving template order, and
`outputs/predicted_labels_nov_dec.csv` with the label-style `boardings` column.
Both files contain the same predicted counts. Source/checkpoint fingerprints
and the cutoff are recorded in `outputs/submission_nov_dec_metadata.json`.

Run from the repository root with the existing virtual environment. Each experiment command below
writes to a fresh output directory and refuses to overwrite an existing run.
The original direct checkpoints from the patch-Transformer experiment are
prerequisites for the probe and final export.

```bash
.venv/bin/python experiments/autoregression/probe.py
.venv/bin/python experiments/autoregression/backtest.py
.venv/bin/python experiments/autoregression/train_recursive.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/autoregression/linear.py
.venv/bin/python experiments/autoregression/tune_cnn.py
.venv/bin/python experiments/autoregression/select_and_export.py --include-tuned
```

The export chooses the highest **pooled WAPE-score** across both origins from its
small direct/recursive blend grid. It preserves the original direct submission.
It uses January–October refit checkpoints and writes the new complete 14,640-row
November–December submission to `outputs/autoregression_selected/submission.csv`.
The metadata records checkpoint paths and blend weights. Route 5 uses zero;
its actual boarding total is zero in both historical evaluation windows, so the
reported scores equal full-grid scores.

Two alternatives are saved for comparison: `submission_conservative.csv` chooses
from recipes at least as good as the previous blend on both origins;
`submission_latest_holdout.csv` optimizes only September–October and has greater
risk of being specific to that period. The main submission uses the pooled rule.

## Checks

```bash
.venv/bin/python -m pytest -q --confcutdir=experiments \
  experiments/autoregression experiments/patch_transformer
.venv/bin/python -m ruff check src/tram_forecast/autoregressive.py \
  experiments/autoregression experiments/patch_transformer/run.py
```

Tests verify actual prediction feedback, daily and weekly blocks, independence
from altered or physically absent future labels, no mutation of observations,
anchored feedback and argument validation. The existing patch-model and checkpoint
integration tests are also run. The pre-existing repository test suite still
cannot collect because it imports modules absent from this checkout.
