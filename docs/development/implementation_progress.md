# Implementation progress tracker

Track execution of [implementation_steps.md](implementation_steps.md). Technical requirements remain in [architecture_implementation_plan.md](../architecture/architecture_implementation_plan.md).

## Update rules

- Start each step with status **In progress**.
- Mark its checkbox and status **Done** only after its deliverables and completion checks pass.
- On completion, briefly record what changed, how it was implemented, and how it was verified. Include file paths and actual check results.
- If incomplete, use **Blocked** or **In progress**, leave its checkbox unchecked, and state the remaining work.
- Distinguish fixture checks, real-data checks, and experiments. Never mark unrun checks as passed.
- Update this tracker as part of finishing each step; do not wait until the whole project is complete.
- Status entries below distinguish implemented code and synthetic checks from real-data execution. The user requested code only: no actual model training, full-data preparation, or forecast generation during this pass.

## Checklist

- [ ] 1. Package skeleton and configuration
- [x] 2. Parsing, hourly labels, and calendar features
- [x] 3. Disk-backed preprocessing and vocabulary artifacts
- [x] 4. Daily sample dataset and epoch sampler
- [x] 5. Universal multiscale block and event encoder
- [x] 6. Complete forecasting network
- [x] 7. Ragged collation, checkpointing, and exact tiling
- [ ] 8. Fixed-cutoff evaluation and baselines
- [ ] 9. Training loop, checkpointing, and resume
- [ ] 10. Final refit and submission pipeline
- [ ] 11. Real-data preparation and resource smoke checks
- [ ] 12. Validation experiments — explicit execution phase
- [ ] 13. Final fit and forecast — explicit execution phase

## 1. Package skeleton and configuration

**Status:** In progress
**Completed:** —

- **What changed:** Package skeleton, model configuration and explicit pending CLI commands.
- **How implemented:** Frozen validated configuration with deterministic hash; src-layout packaging.
- **Verification:** 8 tests pass in aggregate; configuration roundtrip and invalid path checks.
- **Files/artifacts:** pyproject.toml, configs/default.json, src/tram_forecast/config.py, cli.py
- **Remaining work/blockers:** Full preprocessing/training configuration and install integration remain.

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
- **Verification:** 11 tests pass on CPU PyTorch 2.14.0; padding output/gradient equivalence, branch gradients, tiny-fixture learning, cache/reload equivalence and 199,268 maximum parameter count.
- **Files/artifacts:** src/tram_forecast/model/blocks.py, events.py, tests/test_model.py
- **Remaining work/blockers:** None for this component; real-data integration remains in later steps.

## 6. Complete forecasting network

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Full calendar-aware two-branch network, compressed head, cached-history API and parameter reporting.
- **How implemented:** Exact configured parallel Conv1d/GELU paths with aligned concatenation and masking.
- **Verification:** 11 tests pass on CPU PyTorch 2.14.0; padding output/gradient equivalence, branch gradients, tiny-fixture learning, cache/reload equivalence and 199,268 maximum parameter count.
- **Files/artifacts:** src/tram_forecast/model/network.py, tests/test_model.py
- **Remaining work/blockers:** None for this component; real-data integration remains in later steps.

## 7. Ragged collation, checkpointing, and exact tiling

**Status:** Done
**Completed:** 2026-09-25

- **What changed:** Length-bucketed ragged execution, checkpointed event encoding, and exact core/halo tiling.
- **How implemented:** Enforce padded-position/hour limits, derive halo from configured paths, restore chronological positions, preserve earliest maxima on ties.
- **Verification:** 14 CPU tests pass, including padded/chunked/tiled value and parameter-gradient equivalence, boundary context, ties, empty hours and full-network integration. Ruff passes; editable installation and config CLI verified.
- **Files/artifacts:** src/tram_forecast/collate.py, model/events.py, model/network.py, tests/test_event_chunks.py.
- **Remaining work/blockers:** No component blocker. Peak memory and real busiest-history feasibility remain unmeasured until step 11; global Gate B data-leakage integration checks await storage/dataset implementation.

## 8. Fixed-cutoff evaluation and baselines

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 9. Training loop, checkpointing, and resume

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 10. Final refit and submission pipeline

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 11. Real-data preparation and resource smoke checks

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 12. Validation experiments — explicit execution phase

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 13. Final fit and forecast — explicit execution phase

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.
