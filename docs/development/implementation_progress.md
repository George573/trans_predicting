> Historical implementation record for the original two-stream model.
> The current boarding-only architecture is documented in [the manifest](../architecture/architecture_manifest.md).
> Event processing and its associated implementation steps have been retired.

# Implementation progress tracker

Track execution of [implementation_steps.md](implementation_steps.md). Technical requirements remain in [architecture_implementation_plan.md](../architecture/architecture_implementation_plan.md).

## Training notebook and legacy installer — 2026-09-26

- **Notebook: Done.** Added `notebooks/train.ipynb` with editable run configuration,
  preparation/reuse, weekly baseline, resource checks, training/resume, learning
  curves, best-checkpoint evaluation and forecast plots. Added the `notebook`
  dependency extra and README launch instructions.
- **Verification:** 43 tests passed in 6.04s; notebook schema validated with
  nbformat. Fixture execution covers both model kinds using untrained checkpoint
  doubles for training; no actual optimizer step or real-data preparation ran.
- **Legacy installer: awaiting clarification.** The requested “CUDA 1.12” is
  ambiguous between CUDA runtime versions and PyTorch 1.12. Select the wheel and
  Python compatibility range once the CUDA version/GPU target is identified.

## Seven-day shared-context refactor — 2026-09-26

**Status:** Done

- User-approved scope: configurable forecast horizon, initially seven days for all models; encode each training context once for multiple requested days.
- Group route/cutoff contexts, retain partial horizons at the fitting boundary, and weight loss/accumulation by requested days.
- Persist the horizon in checkpoints, use it in evaluation/refit/export, and reject mixed-horizon report comparisons or incompatible resume.
- Update the current README/specification and verify shared-context predictions and gradients, boundary behavior, and seven-day export on fixtures.
- **Verification:** `.venv/bin/python -m pytest -q` — 40 passed in 4.14s on CPU. Shared-context values and parameter gradients match individual requests for both model variants, with event checkpointing enabled. Tests cover partial-horizon weighting, held-out targets beyond day 7, checkpoint-driven 7/14-day evaluation, incompatible resume, mixed-horizon comparison rejection, and 7/61-day exports.
- **Additional checks:** Default `check-config` and `git diff --check` passed. Installed project/test dependencies into the ignored local `.venv`; no actual optimizer training run was performed.
- Real-data preparation, full training and empirical forecast quality remain unmeasured.

The numbered completion records below describe the original implementation pass;
the refactor entry above supersedes its individual-request and 61-day defaults.

## Update rules

- Start each step with status **In progress**.
- Mark its checkbox and status **Done** only after its deliverables and completion checks pass.
- On completion, briefly record what changed, how it was implemented, and how it was verified. Include file paths and actual check results.
- If incomplete, use **Blocked** or **In progress**, leave its checkbox unchecked, and state the remaining work.
- Distinguish fixture checks, real-data checks, and experiments. Never mark unrun checks as passed.
- Update this tracker as part of finishing each step; do not wait until the whole project is complete.
- Status entries below distinguish implemented code and synthetic checks from real-data execution. The user requested code only: no actual model training, full-data preparation, or forecast generation during this pass.

## Checklist

- [x] 1. Package skeleton and configuration
- [x] 2. Parsing, hourly labels, and calendar features
- [x] 3. Disk-backed preprocessing and vocabulary artifacts
- [x] 4. Daily sample dataset and epoch sampler
- [x] 5. Universal multiscale block and event encoder
- [x] 6. Complete forecasting network
- [x] 7. Ragged collation, checkpointing, and exact tiling
- [x] 8. Fixed-cutoff evaluation and baselines
- [x] 9. Training loop, checkpointing, and resume
- [x] 10. Final refit and submission pipeline
- [ ] 11. Real-data preparation and resource smoke checks
- [ ] 12. Validation experiments — explicit execution phase
- [ ] 13. Final fit and forecast — explicit execution phase

## 1. Package skeleton and configuration

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Complete package layout, application/model configuration and functional CLI.
- **How implemented:** Nested validated model/data/training settings; explicit subcommands and atomic outputs; organized architecture/development docs and legacy experiments.
- **Verification:** Editable package install, configuration loading, CLI help/inspect and lint verified.
- **Files/artifacts:** pyproject.toml; configs/default.json; src/tram_forecast/settings.py; cli.py; README.md
- **Remaining work/blockers:** None for code delivery.

## 2. Parsing, hourly labels, and calendar features

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Strict parsing, labels, calendars and fitting-only scale.
- **How implemented:** Validated CSV readers and complete hourly grids; production raw staging validates schema and timestamps.
- **Verification:** Synthetic parsing/calendar/label and pipeline integration tests passed.
- **Files/artifacts:** src/tram_forecast/schema.py; preprocess.py; tests/test_contracts.py; test_pipeline_data.py
- **Remaining work/blockers:** No real-data scan requested.

## 3. Disk-backed preprocessing and vocabulary artifacts

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Immutable disk-backed event/label artifacts and training-only vocabularies.
- **How implemented:** DuckDB bounded staging, external ordering and category grouping; memory-mapped int32 event export, offsets, source identity checks, atomic publication.
- **Verification:** Fixture preparation/reuse, held-out category exclusion, final refit vocabulary growth and corrupt-scaler rejection passed.
- **Files/artifacts:** src/tram_forecast/preprocess.py; storage.py; io.py; settings.py
- **Remaining work/blockers:** Large-data runtime/disk capacity remain unmeasured.

## 4. Daily sample dataset and epoch sampler

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Storage-backed daily dataset and batch collation.
- **How implemented:** Compact integer identity array; route-hour slices; separate targets; boarding-only path avoids event storage.
- **Verification:** 105,408 structural identities verified; fixture histories/targets and both data modes passed.
- **Files/artifacts:** src/tram_forecast/dataset.py; tests/test_pipeline_data.py
- **Remaining work/blockers:** Real-data feasibility belongs to step 11.

## 5. Universal multiscale block and event encoder

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Independent multiscale event blocks, embeddings, masked pooling and empty-hour handling.
- **How implemented:** Exact configured parallel Conv1d/GELU paths with aligned concatenation and masking.
- **Verification:** 11 tests pass on CPU PyTorch 2.14.0; padding output/gradient equivalence, branch gradients, a prior tiny-fixture learning check (before the later no-training instruction), cache/reload equivalence and 199,268 maximum parameter count.
- **Files/artifacts:** src/tram_forecast/model/blocks.py, events.py, tests/test_model.py
- **Remaining work/blockers:** None for this component; real-data integration remains in later steps.

## 6. Complete forecasting network

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Full calendar-aware two-branch network, compressed head, cached-history API and parameter reporting.
- **How implemented:** Exact configured parallel Conv1d/GELU paths with aligned concatenation and masking.
- **Verification:** 11 tests pass on CPU PyTorch 2.14.0; padding output/gradient equivalence, branch gradients, a prior tiny-fixture learning check (before the later no-training instruction), cache/reload equivalence and 199,268 maximum parameter count.
- **Files/artifacts:** src/tram_forecast/model/network.py, tests/test_model.py
- **Remaining work/blockers:** None for this component; real-data integration remains in later steps.

## 7. Ragged collation, checkpointing, and exact tiling

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Length-bucketed ragged execution, checkpointed event encoding, and exact core/halo tiling.
- **How implemented:** Enforce padded-position/hour limits, derive halo from configured paths, restore chronological positions, preserve earliest maxima on ties.
- **Verification:** 14 CPU tests pass, including padded/chunked/tiled value and parameter-gradient equivalence, boundary context, ties, empty hours and full-network integration. Ruff passes; editable installation and config CLI verified.
- **Files/artifacts:** src/tram_forecast/collate.py, model/events.py, model/network.py, tests/test_event_chunks.py.
- **Remaining work/blockers:** No component blocker. Fixed-history held-out-target independence now has integration coverage. Real peak-memory feasibility is deferred under step 11.

## 8. Fixed-cutoff evaluation and baselines

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Fixed-cutoff evaluation, weekly profile and boarding-only network.
- **How implemented:** Encode each route once; sum absolute error and actual totals before WAPE; report route/lead groups and separate route-5 fallback.
- **Verification:** Untrained fixture evaluation, hand-calculated metrics, zero denominator, held-out target independence and event-free baseline passed.
- **Files/artifacts:** src/tram_forecast/evaluate.py; model/network.py; tests/test_execution.py
- **Remaining work/blockers:** No empirical trained-model comparison run.

## 9. Training loop, checkpointing, and resume

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Training, accumulation, early stopping, checkpointing and epoch resume code.
- **How implemented:** Original-unit MAE; AdamW; sample-weighted partial groups; RNG/optimizer/artifact persistence; strict resume compatibility.
- **Verification:** Inert model/loss/optimizer doubles test epoch control and early stopping; analytical gradients test partial accumulation. Untrained checkpoints/RNG roundtrip correctly. No actual optimizer training run.
- **Files/artifacts:** src/tram_forecast/train.py; losses.py; checkpoint.py; tests/test_execution.py
- **Remaining work/blockers:** Convergence and actual epoch-boundary resumed learning remain unmeasured by user request.

## 10. Final refit and submission pipeline

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Final refit and submission code.
- **How implemented:** Fresh final artifacts/model with selected epoch count; all-route inference and separate route-5 zeros; exact key validation, atomic CSV and checkpoint provenance.
- **Verification:** Mocked refit handoff preserves epoch selection; synthetic 14,640-key submission preserves template order and rejects nonfinite values.
- **Files/artifacts:** src/tram_forecast/predict.py; train.py; tests/test_execution.py
- **Remaining work/blockers:** No final training or real submission generated.

## 11. Real-data preparation and resource smoke checks

**Status:** Code complete; real-data execution deferred
**Completed:** —

- **What changed:** Resource-check and real-data command code complete.
- **How implemented:** Typical/busiest indexed history forward/backward with no optimizer; CPU RSS/CUDA peak reporting.
- **Verification:** Synthetic artifact smoke command verified with optimizer steps explicitly prohibited; package tests pass.
- **Files/artifacts:** src/tram_forecast/smoke.py; cli.py; README.md
- **Remaining work/blockers:** Real-data preparation, GPU tests and real peak-memory/runtime checks deliberately not run (code-only request).

## 12. Validation experiments — explicit execution phase

**Status:** Code complete; execution deferred
**Completed:** —

- **What changed:** Experiment and comparison commands implemented.
- **How implemented:** Train/evaluate both model kinds and compare identical artifact/cutoff reports with weekly baseline.
- **Verification:** Evaluation paths and comparison compatibility checks covered by code review/fixtures; no trained results claimed.
- **Files/artifacts:** src/tram_forecast/train.py; evaluate.py; cli.py
- **Remaining work/blockers:** Actual experiments excluded by the user.

## 13. Final fit and forecast — explicit execution phase

**Status:** Code complete; execution deferred
**Completed:** —

- **What changed:** Final fitting/forecast command path implemented.
- **How implemented:** Final preparation, fresh refit or epoch resume, cached-history predictions and validated submission export.
- **Verification:** Final vocabulary regeneration, mocked refit and full-key synthetic export tests.
- **Files/artifacts:** src/tram_forecast/preprocess.py; train.py; predict.py
- **Remaining work/blockers:** Actual final fitting and production forecast excluded by the user.
