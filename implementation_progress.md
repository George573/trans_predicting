# Implementation progress tracker

Track execution of [implementation_steps.md](implementation_steps.md). Technical requirements remain in [architecture_implementation_plan.md](architecture_implementation_plan.md).

## Update rules

- Start each step with status **In progress**.
- Mark its checkbox and status **Done** only after its deliverables and completion checks pass.
- On completion, briefly record what changed, how it was implemented, and how it was verified. Include file paths and actual check results.
- If incomplete, use **Blocked** or **In progress**, leave its checkbox unchecked, and state the remaining work.
- Distinguish fixture checks, real-data checks, and experiments. Never mark unrun checks as passed.
- Update this tracker as part of finishing each step; do not wait until the whole project is complete.
- Steps 5–7 are complete at component/fixture level; steps 1, 2 and 4 have partial foundations. Steps 3 and 8–13 remain pending. Steps 12–13 remain separately requested execution phases. A model-first increment was chosen to verify the difficult convolution/memory components before real-data integration; sequential data gates have not been claimed complete.

## Checklist

- [ ] 1. Package skeleton and configuration
- [ ] 2. Parsing, hourly labels, and calendar features
- [ ] 3. Disk-backed preprocessing and vocabulary artifacts
- [ ] 4. Daily sample dataset and epoch sampler
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

**Status:** In progress
**Completed:** —

- **What changed:** Strict CSV parsing, hourly label grid, count scale and calendar vectors.
- **How implemented:** Streaming readers, route parser, duplicate validation and NumPy calendar encoding.
- **Verification:** Fixture parsing, held-out scale exclusion, duplicate rejection and calendar tests pass.
- **Files/artifacts:** src/tram_forecast/schema.py, tests/test_contracts.py
- **Remaining work/blockers:** Broader malformed-input fixtures and preparation integration remain.

## 3. Disk-backed preprocessing and vocabulary artifacts

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 4. Daily sample dataset and epoch sampler

**Status:** In progress
**Completed:** —

- **What changed:** Compact daily sample identities and deterministic epoch permutation.
- **How implemented:** Enumerate valid cutoff/lead pairs; seed epoch shuffles without replacement.
- **Verification:** 105,408 unique identities; leads 1–61 and split boundaries verified.
- **Files/artifacts:** src/tram_forecast/dataset.py, tests/test_contracts.py
- **Remaining work/blockers:** Memory-mapped sample loading depends on step 3; full Gate A pending.

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
