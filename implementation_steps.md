# Implementation steps

Follow `architecture_implementation_plan.md` v1.1 for all architecture, data, training, and CLI details. This checklist divides that specification into bounded tasks; it does not change its defaults. Execute steps in order. All steps are initially pending.

Record live status and completion summaries in [implementation_progress.md](implementation_progress.md). Update that tracker when starting or finishing each step.

For each step: read its referenced specification sections, implement the listed deliverables, run its completion checks, and record changed files, checks run, results, and remaining blockers in the progress tracker. Preserve existing user work. Do not mark fixture tests as real-data verification. Do not start full training during implementation steps 1–11.

## 1. Package skeleton and configuration

**Specification:** sections 1–3, 12.

- Create `pyproject.toml`, `src/tram_forecast/`, module entry point, and `configs/default.json`.
- Implement validated configuration dataclasses, JSON loading, resolved configuration serialization/hash, schema constants, route mapping, and sample/batch types.
- Encode the exact layer paths, vocab caps, time periods, training defaults, and artifact paths in configuration.
- Expose CLI help and command argument contracts. Commands not yet implemented must fail explicitly, never report placeholder success.

**Done when:** package imports, CLI help works, default configuration validates, invalid layer geometry/time settings produce actionable errors. Record the installation command and dependency versions.

## 2. Parsing, hourly labels, and calendar features

**Specification:** sections 2, 4 parsing rules, 5.

- Implement strict raw/label readers, route normalization, timestamp parsing, deterministic tie keys, and label-key validation.
- Build complete zero-filled label grids plus label-presence metadata.
- Implement exact historical/request calendar encodings and global count scaling fitted only on the allowed interval.
- Create small synthetic CSV fixtures with missing hours, unsorted events, timestamp ties, file-boundary tails, malformed rows, and held-out data.

**Done when:** valid fixtures align correctly; malformed inputs fail; absent labels become zero; held-out labels cannot alter training scale. Verify calendar values and lead boundaries with hand-calculated examples.

## 3. Disk-backed preprocessing and vocabulary artifacts

**Specification:** section 4.

- Stage selected event fields in bounded DuckDB batches and externally sort by the required keys.
- Fit capped training-only vocabularies with deterministic frequency/tie rules and separate PAD/MISSING/UNK IDs.
- Stream events into memory-mapped arrays; build route-hour offsets, metadata, mappings, scaling and separate evaluation-label artifacts.
- Implement atomic artifact publication, compatibility checks, `prepare`, and metadata-only `inspect`.
- Keep full-data execution for step 11; use fixtures here.

**Done when:** a fixture artifact can be reopened and reconstructs the expected event order/hour slices; empty ranges work; held-out categories cannot change training vocabularies; incompatible artifacts are rejected. Gate A preprocessing checks pass.

## 4. Daily sample dataset and epoch sampler

**Specification:** sections 2–3, 5, 11 Gate A.

- Enumerate compact `(route,cutoff,requested_day)` identities with complete 504-hour histories and complete 24-hour targets.
- Load histories by offsets without materializing overlapping raw windows on disk.
- Return ragged event IDs, historical counts/calendar, request conditioning, and separate training targets.
- Implement reproducible shuffling without replacement, including the final partial batch.
- Support a boarding-only data path that never reads raw events.

**Done when:** lead 1/61 and split boundaries are correct; full-grid structural sample counts match the specification; each epoch visits identities once; both branches use identical hour keys. All Gate A checks pass.

## 5. Universal multiscale block and event encoder

**Specification:** sections 6–7.

- Implement independent parallel Conv1d → GELU paths and ordered channel concatenation.
- Implement event embeddings and exact E1/E2 blocks.
- Mask inputs/intermediate outputs; perform masked max per hour; bypass empty hours.
- Initially use small padded fixture batches without checkpointing or tiling so this version provides a correctness reference.

**Done when:** path configurations/shapes match the plan; short event sequences work; padding beside longer hours does not change outputs or gradients; empty hours are finite zeros; embedding weights receive gradients.

## 6. Complete forecasting network

**Specification:** sections 5–7.

- Scatter event vectors into chronological hours and append calendars.
- Implement raw and boarding hourly branches, aligned fusion, S1/S2/S3 multiscale blocks and pooling.
- Implement history flattening, route/request conditioning, dense head, Softplus output and original-unit scaling.
- Expose `encode_history`, `predict_day`, and `forward` with the specified contracts.
- Add independent component/total parameter reporting.

**Done when:** output is `[B,24]`, finite and nonnegative; shapes match every stage; gradients reach both branches; nonembedding count is 128,704 and actual total fits 100k–400k; cached-history inference matches full forward.

## 7. Ragged collation, checkpointing, and exact tiling

**Specification:** section 8, 11 Gate B.

- Bucket nonempty hours by length, pack chunks within both configured limits, and restore their original positions.
- Checkpoint embedding-through-pooling per chunk with `use_reentrant=False`.
- Implement long-hour core/halo tiling, derive halo from actual configured paths, and preserve earliest-event max ties.
- Keep all events; do not let context cross hour boundaries.

**Done when:** chunked/checkpointed/tiled outputs and parameter gradients match the simple reference, including boundary maxima and ties; all Gate B checks pass. Record synthetic memory behavior without claiming real-data feasibility yet.

## 8. Fixed-cutoff evaluation and baselines

**Specification:** section 9.

- Implement global/per-route/per-lead/grouped-lead WAPE and MAE, including explicit zero-denominator handling.
- Implement fixed-cutoff inference for all 61 requested days with history encoding reused in eval mode.
- Implement the three-week weekday/hour mean baseline.
- Implement the boarding-only network variant and its parameter report.
- Add the evaluation command; use an initialized or fixture-trained model until real checkpoints exist.

**Done when:** metrics match hand calculations; observed September/October data cannot update validation history; baseline keys match neural evaluation keys; boarding-only execution does not access raw-event storage.

## 9. Training loop, checkpointing, and resume

**Specification:** sections 9–10.

- Implement original-unit MAE, AdamW, correctly weighted gradient accumulation, clipping, validation, best/latest checkpoints and early stopping.
- Persist config/artifact identities, optimizer, RNG, sampler epoch and training state.
- Implement epoch-boundary resume with compatibility validation.
- Wire `train` for boarding-only/full models and reusable training components for final refit.

**Done when:** a tiny fixture shows materially decreasing loss; predictions survive save/reload; resume restores state; final partial accumulation groups are weighted correctly; no full-data training has been launched.

## 10. Final refit and submission pipeline

**Specification:** sections 9–10, 12.

- Implement fresh January–October refit using the selected hyperparameters and best validation epoch count.
- Implement final fixed-history inference and the separate route-5 zero fallback.
- Validate template keys and write atomic, correctly ordered semicolon CSV output with provenance.
- Wire `refit` and `predict`, testing on fixture artifacts/checkpoints.

**Done when:** fixture integration produces exactly 14,640 valid unique submission keys, preserves template order, and never calls the neural model for route 5. Refit uses new vocabularies/scaling/model initialization and no hidden-period validation.

## 11. Real-data preparation and resource smoke checks

**Specification:** sections 4, 8, 11 Gate C, 12.

- Check real paths, available RAM/disk and configured staging limits.
- Run required validation-regime preprocessing with progress reporting; reuse verified compatible artifacts when available.
- Implement/run `smoke`: typical and busiest indexed history, complete forward/backward step, finite outputs/gradients, exact parameter count, timings and peak memory.
- Reduce batch/chunk sizes if necessary without truncating events or changing topology.
- Complete README usage and record every acceptance check as passed, failed, or not run with its reason.

**Done when:** all applicable Gate C checks pass, real memory/timing measurements are recorded, and the package can run the documented experiment commands. If resource feasibility fails, record the concrete bottleneck and fix it before full training.

## 12. Validation experiments — explicit execution phase

This is a potentially lengthy experiment phase, separate from implementing the package. Run it when the user requests training/experiments; completing steps 1–11 does not automatically launch it.

- Evaluate weekly-profile baseline on the fixed September–October run.
- Train boarding-only and full models using identical identity sets, split, loss and comparable schedules.
- Compare global WAPE, route/lead errors, training time and memory; report the full model's incremental value.
- Preserve all run configs/checkpoints. Do not claim the raw branch helps unless the results support it.

**Done when:** a reproducible comparison report exists and the selected model/hyperparameters/best epoch are identified. If the full model loses, report that result rather than silently redesigning it.

## 13. Final fit and forecast — explicit execution phase

Run after model selection and the user's request to produce the final forecast.

- Prepare/reuse January–October artifacts with newly fitted vocabularies/scaling.
- Refit the selected configuration from scratch for the selected epoch count.
- Predict November–December from the fixed October 11–31 history.
- Validate and deliver `outputs/submission.csv` plus provenance and limitations.

**Done when:** all 14,640 submission rows pass validation and final fitting/inference boundaries are documented. No hidden-label accuracy claim is made.

## Handoff instruction for an implementation agent

> Read `architecture_implementation_plan.md` and `implementation_steps.md`. Implement step N only, using the exact defaults and interfaces in the plan. Inspect existing work and preserve it. Complete the step's focused checks, update its status with evidence, and report changed files, results and blockers. Do not substitute single convolutions for multiscale blocks, add model features, silently drop events, or start later experiment phases. If earlier steps are incomplete, report the precise missing prerequisite before implementing dependent behavior.
