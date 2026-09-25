# Tram forecasting: executable architecture and implementation plan

Version 1.1 — 2026-09-25. Status: implementation specification; no model accuracy or resource measurements claimed.

## 1. Authority, scope, and deliverables

Implement this plan as a Python package in this repository. Read applicable AGENTS.md first. Preserve existing user files and source datasets. This plan resolves ambiguous implementation details in `architecture_manifest.md`; the PDFs are explanatory references, not authoritative layer specifications. Later explicit user instructions override this plan.

**Critical clarification: EVERY convolutional stage uses parallel convolutions with different kernel lengths AND dilations.** This includes both event stages, both hourly branches, and all three shared compression stages. A single Conv1d substituted for any of these blocks is incorrect.

Deliver a working preprocessing, dataset, model, training, evaluation, and submission pipeline; configuration, focused tests, and CLI documentation. First establish correctness and resource feasibility on fixtures and a real batch. Do not automatically launch exhaustive profiling, a hyperparameter sweep, or lengthy full training. Do not implement the model inside the existing exploratory notebook.

Defaults below are concrete implementation decisions within the agreed architecture. Expose them in configuration, persist resolved values, and implement them first. Do not reinterpret configurability as permission to silently redesign the model. No attention, recursive rollout, manually aggregated event features, or additional event fields in v1.

## 2. Forecast contract and time boundaries

One identity is `(route_id, cutoff_date, requested_date)`. All dates use the dataset's existing local calendar convention; do not silently timezone-convert naive timestamps. Retain full timestamps for sorting and boundaries.

- Cutoff `c` is midnight immediately after the last observed day.
- History is exactly `[c - 504 hours, c)` for one route, including zero-filled label hours.
- `lead = (requested_date - cutoff_date).days + 1`, restricted to 1..61.
- Requested target is that date's hours 0..23, ordered chronologically, shape `[24]`.
- All requested days are predicted directly from observed history. A forecast never enters another sample's history.
- Validation: cutoff 2025-09-01; history August 11–31; targets September 1–October 31.
- Submission: cutoff 2025-11-01; history October 11–31; targets November 1–December 31.
- Initial fitting interval: `[2025-01-01, 2025-09-01)`. Final refitting interval: `[2025-01-01, 2025-11-01)`.
- Every training history hour and full target day must lie inside the fitting interval. Do not demand a label row for every hour: absent rows follow the zero-fill convention.
- Fit vocabularies and count scaling using only the relevant fitting interval. Final refit rebuilds them using January–October and starts a fresh model.

Supported neural routes initially: `1,7,11,12,17,25,26,28,50`. Route 5 is excluded under the current no-label-history assumption and uses a separate zero submission fallback. During preprocessing check this assumption: if route 5 has supplied labels, report the conflict rather than silently changing route policy. Require each configured neural route to have some labels in its fitting period; otherwise fail with a diagnostic.

Generate every eligible identity once. With January 1–August 31 (243 days), complete 21-day histories, and all nine routes eligible, lead h has `223-h` cutoffs per route. Expect 11,712 identities per route, 105,408 total. These are structural counts, not independent observations. For final refit (304 days), expect 15,433 per route, 138,897 total. Assert counts on a synthetic complete grid; real eligibility conflicts must be reported.

Each epoch shuffles the complete identity index without replacement using seed `67 + epoch`; `drop_last=False`. Histories and target days may repeat across distinct identities. Do not balance leads through oversampling in v1. Report identity counts by lead because this scheme gives longer leads fewer examples.

## 3. Package and interfaces

Use `src/tram_forecast/` with:

| Module | Responsibility |
|---|---|
| `config.py` | Validated dataclasses, JSON configuration, defaults, resolved configuration hash |
| `schema.py` | Columns, route parsing, dates, sample identity and batch types |
| `preprocess.py` | Bounded CSV ingestion, vocabulary fit, external sort, artifact publication |
| `storage.py` | Read-only memory maps, route-hour offsets, compatibility checks |
| `dataset.py` | Compact identity generation and sample loading |
| `collate.py` | Ragged event collation, length bucketing, chunk descriptors |
| `model/blocks.py` | MultiscaleConv1d and masking |
| `model/events.py` | Embeddings, event encoder, exact event tiling |
| `model/network.py` | Hourly branches, fusion, history encoder, request head |
| `losses.py` | Original-unit MAE and scaling |
| `train.py` | Train loop, checkpointing, resume, early stopping |
| `evaluate.py` | Fixed-cutoff forecasts, WAPE, baseline comparison |
| `predict.py` | Final inference, route-5 fallback, submission checks |
| `cli.py`, `__main__.py` | Subcommands and entry point |

Add `pyproject.toml`, `configs/default.json`, focused tests under `tests/`, and README commands. Use Python, PyTorch, NumPy, standard-library CSV/JSON, and DuckDB for bounded staging/sorting. Pin/document dependencies compatible with the installed environment; do not introduce a training framework. Existing `hourly_windows.py` remains a separate one-hour-target utility; its horizon contract must not be reused as the new day-lead contract.

Required public model methods:

```python
encode_history(history_batch) -> Tensor[B, 672]
predict_day(encoded_history, route_indices, request_calendar, lead) -> Tensor[B, 24]
forward(history_batch, request_batch) -> Tensor[B, 24]
```

All returned predictions are in original boarding units. `forward` composes the other methods. The model input type contains no future target; training targets are a separate batch member.

## 4. Essential preprocessing and disk format

### Parsing and validation

Read semicolon-separated UTF-8 CSVs with BOM tolerance and quoted-field support. Use `tran_date_time` for event assignment, not filename or `input_date_time`. Normalize `ngpt_route` using a strict parser accepting numeric strings or `<integer> трамвай` after trimming. Reject unparseable routes/timestamps with file and row diagnostics. Validate labels: unique `(route,date,hour)`, integer hour 0..23, nonnegative integer count. Identical as well as conflicting duplicate label keys are errors; no implicit summation.

Use an explicit ordered input file list. Sort events by `(route, timestamp, source_file_index, source_row_number)`. Last two fields only break timestamp ties deterministically; they are not model inputs. Retain repeated events; do not infer deduplication from card/device matches. Strip outer whitespace in model categories, preserve case and internal content, and map empty strings to missing. Do not treat literal `"0"`, `"None"`, or `"nan"` as missing unless the source parser/documentation explicitly establishes that convention. Do not apply `normalize_pass_route.py` automatically: v1 treats `pass_route` as a whole categorical value.

Stage all supplied raw files into a disk-backed DuckDB database in bounded batches. Filter by event timestamp for each artifact's fitting/history interval. Keep selected categorical strings and sort keys only; card hashes and transaction IDs are not needed. Use SQL grouping and external sorting, not Python dictionaries of unbounded category counts. Defaults: DuckDB memory limit `2GB`, two threads, configurable temporary directory. Verify available temporary disk space before substantial sorting and report progress and disk failures clearly. Stream ordered results into the final arrays; do not fetch the full relation into RAM. Required preparation may scan files and sort them; do not add exhaustive duplicate profiling or exploratory statistics as a prerequisite.

### Categorical mapping

For each field reserve `0=PAD`, `1=MISSING`, `2=UNK`. Retain up to the following number of **nonmissing training categories**, requiring frequency >=5. Rank by descending frequency, breaking ties lexicographically. Assign retained IDs starting at 3 in that order. Rare and unseen nonmissing values map to UNK.

| Field, in concatenation order | Retained cap | Embedding width |
|---|---:|---:|
| validation_result | 32 | 4 |
| tran_type_id | 128 | 4 |
| good_type | 512 | 8 |
| place_id | 32 | 4 |
| pass_route | 8192 | 8 |

Total event width `D=28`. All embeddings use `padding_idx=0`; count the entire embedding weight matrix in the parameter budget. PAD is never used for a real event, even when every selected field is missing.

Route embedding: nine configured neural routes, sorted numerically, IDs 1..9; reserve 0 for unknown, table size 10, width 8. Reject unknown routes at inference instead of relying on an untrained unknown vector. Route 5 bypasses the network.

### Storage artifact

For each fitting regime create a versioned artifact directory, atomically published after verification:

- `events.npy`: C-contiguous int32 `[N_events,5]`, ordered route then hour then timestamp/tie keys. Write via NumPy open_memmap; never create overlapping-window copies.
- `offsets.npy`: int64 `[R,T,2]`, global half-open `[start,end)` row offsets into events for every route-hour. Empty ranges have start=end.
- `boardings.npy`: float32 `[R,T]`, complete hourly grid with absent rows zero-filled.
- `label_present.npy`: bool `[R,T]` for diagnostics, not a model feature.
- `vocab.json`, `metadata.json`: selected categories, mappings, route order, grid start/end, schema version, resolved preprocessing options, source file sizes/mtimes, row/error/out-of-period counts, and source hashes computed while reading where practical.
- `scaling.json`: global positive count scale and fitting interval.

Small label grids and offsets may fit in RAM; large events must stay disk-backed. Reopen memory maps in each worker if workers are enabled. Default loader `num_workers=0`, avoiding implicit prefetch replication. Validation target labels live in a separate evaluation artifact and cannot be loaded through a history sample's future range. Metadata interval/config mismatches are hard errors. Reuse artifacts only when source fingerprints, schema, fitting interval, and preprocessing configuration match.

Report category sizes/OOV rates, per-route event and label counts, maximum events/hour, and retained disk size as preparation diagnostics. Serious route-hour coverage discrepancies should be surfaced; do not declare raw-to-label equality based on excerpts.

## 5. Scaling and calendars

Compute `s = max(1, mean(boardings))` over ALL hourly grid positions of the configured neural routes in the fitting interval, including zero-filled hours. Use float64 accumulation. Store this single scalar. Historical count input is `boardings / s`; do not use route-specific scaling or log targets in v1.

Historical calendar vector for every hour, including empty hours, in this exact order:

1. `sin(2π hour/24)`, `cos(2π hour/24)`;
2. `sin(2π weekday/7)`, `cos(2π weekday/7)`, Monday=0;
3. `sin(2π (day_of_month-1)/31)`, `cos(2π (day_of_month-1)/31)`.

Shape `[B,6,504]`, float32. The fixed denominator 31 is intentional; it is not annual seasonality. No month, holiday, event-count, fare-mix, unique-card, or other additional inputs in v1.

Requested-day calendar uses the same weekday and day-of-month formulas: `[B,4]`. Normalize lead as `(lead-1)/60`, shape `[B,1]`. Request conditioning order is route embedding (8), requested calendar (4), lead (1), hence `Q=13`. Forecast hour is identified by the output position.

## 6. Universal multiscale block

Define `MS(Cin, [(Cout_i,k_i,d_i), ...])`:

1. Each path independently applies `Conv1d(Cin,Cout_i,k_i,stride=1,padding=d_i*(k_i-1)//2,dilation=d_i,bias=True)`.
2. Apply GELU (`approximate='none'`) to each path.
3. Concatenate path outputs in listed order along channel axis. Length is unchanged; total width is the sum of path widths.

All kernels are odd. No BatchNorm, LayerNorm, residual connection, attention, or extra projection in the default blocks. Distinct blocks and paths have independent weights. Event encoder weights are shared across all routes/hours.

For event tensors only: zero invalid inputs before convolution and zero invalid outputs after GELU in EVERY block. Multiplication by a broadcast bool/float mask is sufficient for finite activations. Never pass `-inf` through a convolution. Use `-inf` only immediately before masked max reduction. Empty hours bypass event encoding. Padding invariance must hold regardless of which other hours share a batch.

Event dilation is measured in event indices, not elapsed seconds. Hourly-block dilation is in hourly positions; after pooling it is measured in compressed positions. Symmetric temporal padding is allowed: every observed input is before the cutoff, even when a kernel accesses a later position within the history.

## 7. Exact default network

Notation in the paths column: `(output channels, kernel length, dilation)`.

| Stage | Input channels | Parallel paths | Output before pooling | Reduction |
|---|---:|---|---|---|
| Event E1 | 28 | (12,3,1), (12,5,1), (8,3,2) | `[H,32,Npad]` | None |
| Event E2 | 32 | (12,3,1), (12,5,2), (8,3,4) | `[H,32,Npad]` | Masked max over valid events |
| Raw hourly A | 38 | (12,3,1), (12,5,24), (8,3,168) | `[B,32,504]` | None |
| Boarding hourly B | 7 | (12,3,1), (12,5,24), (8,3,168) | `[B,32,504]` | None |
| Shared S1 | 64 | (24,3,1), (24,5,2), (16,3,12) | `[B,64,504]` | MaxPool1d(2,2) -> `[B,64,252]` |
| Shared S2 | 64 | (12,3,1), (12,5,2), (8,3,12) | `[B,32,252]` | MaxPool1d(2,2) -> `[B,32,126]` |
| Shared S3 | 32 | (6,3,1), (6,5,2), (4,3,6) | `[B,16,126]` | MaxPool1d(3,3) -> `[B,16,42]` |

`H` is the number of nonempty hours being processed in one chunk; `Npad` is its padded event length. Embedding tensors are transposed to channel-first before E1.

After E2, masked max produces `[H,32]`. Scatter vectors back to their exact `(batch,hour)` positions; empty hours get zero. Form raw input by concatenating `[pooled_events, historical_calendar]`, yielding `[B,38,504]`. Form boarding input by concatenating `[scaled_boardings, historical_calendar]`, yielding `[B,7,504]`. Calendar values must survive empty event hours.

Fuse `[raw_A,boarding_B]` in that order to `[B,64,504]`. Run S1→pool→S2→pool→S3→pool. Pool padding=0, dilation=1, ceil_mode=False. Flatten channel-first `[16,42]` in the standard contiguous PyTorch order to `[B,672]`.

Head: concatenate `[history,route_embedding,request_calendar,normalized_lead]` -> `[B,685]`; Linear(685,128) -> GELU -> Dropout(0.1) -> Linear(128,24) -> Softplus(beta=1,threshold=20) -> multiply by stored `s`. Return finite, nonnegative original-unit predictions `[B,24]`. Do not round during training or metric computation.

Use PyTorch default layer initialization except final Linear bias: initialize each bias to `log(expm1(1))`, so zero logits from the weight contribution correspond to a normalized level near 1. Register `s` as a model buffer. Seed before model construction.

### Parameter accounting

Count `Conv1d = Cout*(Cin*k+1)`, independent of dilation; Linear = out*(in+1). Sum EVERY path and embedding, including reserved rows. Default nonembedding counts:

| Component | Parameters |
|---|---:|
| E1 | 3,392 |
| E2 | 3,872 |
| Raw hourly A | 4,592 |
| Boarding hourly B | 872 |
| S1 | 15,424 |
| S2 | 7,712 |
| S3 | 1,936 |
| Dense head | 90,904 |
| Total excluding embeddings | **128,704** |

Embedding count = `Σ (retained_categories_i+3)*embedding_dim_i + 10*8`. At all listed caps, embeddings = 70,564 and total = **199,268**. Actual total is determined by fitted vocabularies and must equal an independent programmatic `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Default is necessarily within 100k–400k. Do not reuse the old PDF formula. Reject custom configurations outside the agreed range before full training; do not widen layers merely to fill it.

## 8. Ragged events and bounded training memory

Default batch size 1, accumulation 8. Default event chunk limits: at most 32 hour sequences and at most 32,768 padded event positions (`H*Npad`) per chunk. Sort nonempty hours by `(length, original_flat_hour_index)` for packing; greedily take hours while both limits hold, then restore original ordering through scatter. Do not reorder events inside an hour. Padded categorical IDs are zero; lengths/masks define validity.

Forward chunking alone is insufficient: default training MUST checkpoint each chunk's **embedding → E1 → E2 → masked max** function with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Pass compact integer IDs/lengths as checkpoint inputs, construct embeddings inside the checkpoint, and avoid retaining embedded tensors in Python lists. Store only pooled outputs and compact inputs for recomputation. Verify gradients reach embeddings; integer inputs requiring no gradients must not disable training. Checkpointing is disabled under inference/no_grad.

Do not silently truncate a single extremely busy hour. Implement exact tiling when one hour exceeds the padded-position budget:

- E1 maximum receptive-field radius is 2 events; E2 maximum radius is 4; combined radius is 6 events.
- Split an hour into disjoint core intervals of at most `32768-12` events; extend each core with up to six actual neighboring events on both sides.
- Encode the extended interval with E1/E2, retaining outputs only at core positions. Internal tile boundaries have real halo context; actual hour boundaries use normal zero padding.
- Take channelwise max over the retained core positions, then across tiles to obtain the exact hourly max. On exact ties, preserve the earliest chronological event (combine tile maxima in chronological order using strict `>` selection); this matches full-sequence `torch.max(dim=...)` tie behavior.
- Checkpoint each tile during training. Compute halo from configured kernels/dilations, not a magic constant, and ensure the chosen core is positive.
- Never allow convolution context to cross from one hour to another.

Test tiled and untiled outputs AND gradients. Packing and tiling are execution strategies, not model changes. They need no added count features. Compact CPU inputs still scale with the actual events in a batch; measure this and fail descriptively if resource limits are exceeded. Do not claim globally constant RAM or a guaranteed GPU fit.

Default precision float32. AMP is optional only after masked reductions/checkpointed backward are verified; no silent precision change. Measure full-step CUDA peak allocated/reserved memory when CUDA is used, and process RSS on CPU. Exercise both a typical real batch and the busiest indexed history. On OOM lower chunk limits/batch size, preserving all events and the model; record the revised settings. Do not automatically change architecture.

No persistent cache of learned event/history vectors during training: encoder weights change. Inference in eval mode may cache one `[1,672]` history vector per route/cutoff and reuse it for all 61 requests.

## 9. Training and evaluation defaults

- Seed 67 for Python/NumPy/PyTorch; record device/library versions. Request deterministic operations where available, report unsupported cases.
- Optimizer AdamW, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0001 on trainable parameters. No scheduler initially.
- Objective original-unit MAE: `abs(prediction-target).mean()` across sample/hour dimensions. This directly matches the numerator of global WAPE for a fixed evaluation set.
- Gradient accumulation: combine sample loss sums, normalize by the actual number of samples in the accumulation group (including the final partial group), then clip global gradient norm at 1.0 and step. Do not underweight the final partial group or overweight smaller batches.
- Maximum 30 complete epochs, validation once per epoch, early stopping after five consecutive epochs without strictly lower global neural-route validation WAPE. Best checkpoint is minimum WAPE; ties retain the earlier epoch. Configurable short smoke mode is not a full training result.
- Fixed-cutoff validation encodes each route once in eval mode and requests all 61 days. It must not update history with September/October observations.
- Report global WAPE by summing absolute errors and actual counts before division. Never average batch or route WAPEs to obtain global WAPE. If denominator is zero, return JSON null plus reason, and report absolute error separately.
- Report per-route, per-lead (1..61), and grouped leads 1–7, 8–14, 15–30, 31–61; include denominators. Report MAE too.
- Report full-grid score with route-5 fallback separately from neural-route selection metric. Keep the baseline/full comparison on identical keys. `WAPE-score=max(0,1-WAPE)` is optional additional reporting.

Baselines:

1. **Weekly profile:** for each requested weekday/hour, arithmetic mean of the three matching weekday/hour counts in the fixed 21-day history. No observed-history updates and no future inputs. Apply route-5 fallback separately.
2. **Boarding-only CNN:** remove event embedding/E1/E2/raw hourly branch. Keep boarding branch, request conditioning, head and pooling schedule; S1 input becomes 32 channels with the same listed output paths. Train independently on the same sample identities/splits/loss. Its parameter count may differ; report it. Raw events must not be loaded for this baseline.
3. **Full network:** exact architecture above.

No accuracy claim until these runs exist. Do not select settings using hidden November–December labels. After model selection, final refit uses January–October, new preprocessing/scaling/vocabularies, and a fresh initialization for the selected number of epochs (best validation epoch count). It has no November/December early stopping. Preserve the selected architecture/hyperparameters.

## 10. Persistence, inference, and submission

Save best and latest checkpoints containing model state, optimizer state, epoch/step, best metric, early-stopping counter, RNG states, complete resolved configuration, vocabulary/scaler artifacts or validated hashes, route mapping, fitting boundaries, and preprocessing artifact version. Support epoch-boundary resume; explicitly reject an incompatible checkpoint/artifact combination. Do not imply exact mid-epoch resume unless sampler position and accumulation state are implemented.

Inference runs eval/no_grad. Encode October 11–31 once per neural route; request each November/December day with correct lead. Generate route 5 as zero separately. Preserve the template key order and replace predictions; template baseline values are not labels or inputs.

Submission UTF-8 semicolon CSV header: `route;date;hour;prediction`. Exactly 14,640 unique keys, 10 listed routes × 61 dates × 24 hours; finite nonnegative numeric predictions, no index column. Write floating-point predictions without manual rounding; dataset scoring handles rounding. Reject missing/extra/duplicate template keys. Atomically publish outputs and save checkpoint/config provenance alongside the CSV.

## 11. Acceptance tests and implementation gates

Use small synthetic fixtures to check behavior, not merely duplicate formulas in tests. Preserve existing tests.

### Gate A — preparation and sample boundaries

- Unsorted transactions and tied timestamps produce deterministic route-hour sequences; raw file boundaries do not determine date eligibility.
- Empty fields, rare training categories and unseen evaluation categories map to their distinct intended IDs. Changes to held-out category values cannot change fitted training mappings/scaling.
- Label duplicates, invalid hours/counts/timestamps fail visibly.
- Lead 1/61, first valid cutoff, last valid target, and month boundaries satisfy exact intervals; September labels cannot enter training inputs/targets.
- Identity enumeration counts match Section 2; an epoch visits all identities exactly once.
- Both branches and calendar features reference identical route-hour keys, including zero-filled hours.

### Gate B — model correctness

- Batch output `[B,24]` is finite/nonnegative; every stage has expected channels/length.
- Every convolutional stage contains the specified parallel kernel/dilation paths; weights are independent where specified.
- An hour encoded alone versus padded beside a longer hour produces equivalent output/gradients within float32 tolerances (`atol=1e-5, rtol=1e-4`). Include lengths 1,2 and zero.
- All-empty raw hours yield zero event vectors without reducing an all-`-inf` tensor; historical calendars still enter both hourly branches.
- Tiled vs untiled events match pooled values and parameter gradients, including maxima near tile boundaries and ties. Chunked/checkpointed vs unchunked execution agrees with dropout disabled.
- Loss gradients reach both hourly branches and retained event embeddings for nonempty fixtures. No model input includes the requested target.
- Observations after cutoff cannot affect a fixed-cutoff prediction; changing the requested day changes conditioning without changing history tensors.
- Independently counted parameters agree with the component table and embedding formula and fit the budget.

### Gate C — training/inference integration

- A tiny fixture can be overfit enough to materially decrease loss; require finite backward gradients rather than an arbitrary accuracy threshold on real data.
- Save/reload in eval mode reproduces predictions; epoch-boundary resume restores optimizer/RNG and sampling semantics.
- Cached-history prediction equals full forward in eval mode.
- WAPE matches hand-calculated cases, including zero denominator and unequal route volumes.
- Submission fixture covers every key exactly once and keeps route 5 outside neural calls.
- Real typical and busiest-history forward/backward memory checks pass with recorded timings/peak memory before full training.

Stop optional testing once these gates pass. If a gate fails, fix the specific cause. Do not hide failures by reducing events, feeding validation history, adding features, or modifying the agreed topology.

## 12. Work order and CLI contract

See [implementation_steps.md](../development/implementation_steps.md) for the task-by-task checklist, deliverables, completion checks, and implementation-agent handoff. Steps 1–11 implement and verify the package; steps 12–13 are separately requested experiment and final-forecast execution phases.

Implement sequentially:

1. Configuration/schema and focused fixture generators.
2. Preprocessing/storage and identity dataset; pass Gate A.
3. Multiscale blocks and complete model; verify shapes and parameter report.
4. Ragged collation, checkpointing and exact tiling; pass Gate B.
5. Training/checkpoints, fixed-cutoff evaluation and baselines.
6. Inference/submission and Gate C; document completed and unrun checks.

Expose these commands (flag names form part of the requested interface):

```bash
python -m tram_forecast prepare --config configs/default.json --regime validation
python -m tram_forecast inspect --artifact outputs/prepared/validation
python -m tram_forecast smoke --config configs/default.json --artifact outputs/prepared/validation
python -m tram_forecast train --config configs/default.json --artifact outputs/prepared/validation --model boarding_only
python -m tram_forecast train --config configs/default.json --artifact outputs/prepared/validation --model full
python -m tram_forecast evaluate --checkpoint PATH --artifact outputs/prepared/validation
python -m tram_forecast prepare --config configs/default.json --regime final
python -m tram_forecast refit --selected-checkpoint PATH --artifact outputs/prepared/final
python -m tram_forecast predict --checkpoint PATH --artifact outputs/prepared/final --template dataset/test_submission.csv --output outputs/submission.csv
```

Default data paths: `dataset/train.csv`, `dataset/test.csv`, `dataset/labels/labels_day_train.csv`, `dataset/labels/labels_day_test.csv`. Verify existence rather than substituting excerpts. `prepare --regime validation` can read both raw files to assign timestamp tails correctly but must only fit/store training-period raw history; held-out labels are stored separately for evaluation. `prepare --regime final` uses the January–October timestamp interval. `inspect` reads artifact metadata/array shapes without a fresh full-data scan. `smoke` runs correctness/resource checks and a small number of training steps, not 30 epochs. Evaluation includes the weekly-profile baseline automatically.

Final implementation report: changed files, exact model counts, gates passed, memory/time measurements, commands to run next, and any concrete blocker. Clearly distinguish implemented code, fixture-verified behavior, real-data checks, and experiments not yet run.
