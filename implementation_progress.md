# Implementation progress tracker

Track execution of [implementation_steps.md](implementation_steps.md). Technical requirements remain in [architecture_implementation_plan.md](architecture_implementation_plan.md).

## Update rules

- Start each step with status **In progress**.
- Mark its checkbox and status **Done** only after its deliverables and completion checks pass.
- On completion, briefly record what changed, how it was implemented, and how it was verified. Include file paths and actual check results.
- If incomplete, use **Blocked** or **In progress**, leave its checkbox unchecked, and state the remaining work.
- Distinguish fixture checks, real-data checks, and experiments. Never mark unrun checks as passed.
- Update this tracker as part of finishing each step; do not wait until the whole project is complete.
- Planning documents already exist, but no implementation step has been completed. Steps 12–13 remain separately requested execution phases.

## Checklist

- [ ] 1. Package skeleton and configuration
- [ ] 2. Parsing, hourly labels, and calendar features
- [ ] 3. Disk-backed preprocessing and vocabulary artifacts
- [ ] 4. Daily sample dataset and epoch sampler
- [ ] 5. Universal multiscale block and event encoder
- [ ] 6. Complete forecasting network
- [ ] 7. Ragged collation, checkpointing, and exact tiling
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

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 6. Complete forecasting network

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

## 7. Ragged collation, checkpointing, and exact tiling

**Status:** Not started
**Completed:** —

- **What changed:** —
- **How implemented:** —
- **Verification:** —
- **Files/artifacts:** —
- **Remaining work/blockers:** Not assessed.

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
