# Tasks: Parameterized Targets and Adaptive Curriculum

**Branch**: `012-parameterized-adaptive-curriculum` | **Date**: 2026-11-04
**Status**: Original phases A–D (T001–T066) and all six operator-failure repairs F001–F006 complete. Latest local validation: 654 tests and 2 subtests passed. The original operator attempt failed admission; current amended-scope native/container/HIP acceptance and research remain pending in T067–T074. No HIP or CI wait was performed during these repairs.
**Authority**: [spec](spec.md), [plan](plan.md), [research](research.md),
[data model](data-model.md), [experiment](contracts/experiment.md),
[curriculum](contracts/curriculum.md), [execution](contracts/execution.md),
[reliability review](reliability-review.md), [handoff](quickstart.md).

## Execution rules

These 74 tasks implement the accepted design, not a new research choice.
Use phases A–E from the plan rather than introducing one critical phase per story:
the operator explicitly requires **exactly two dedicated critical phases, B and C**.
Story subgroups and labels preserve independently testable slices. A contains
routine setup and foundations; D contains routine delivery and broad validation;
E is one consolidated operator session plus evidence reconciliation.

Read the applicable contract before each subgroup. Add independent counterexamples
before changing its implementation; the constitution explicitly requires tests for
arithmetic, admission, leakage, resume and containment. Existing correct behavior
may satisfy a task after inspection and a relevant passing check: do not duplicate
it. Paths below are repository-relative; paths marked **new** are intended additions.
Keep the listed small module boundaries; do not build a second learner or scheduler
framework. Commit **and push** after each completed subheader task group, recording
validation and any remaining evidence gap. Do not wait for CI unless newly instructed.

`[P]` means disjoint-file work may run concurrently **after its stated phase entry
requirements**. It does not authorize bypassing dependencies or spawning agents.
Otherwise execute in ID order. All B counterexample tasks marked `[P]` can be
written after A, but each implementation must wait for its preceding tests. Finish
all B before C and all C before D. A defect in an invariant found during D reopens
the owning B/C task; it is not a third critical phase.

No task here retroactively completes 011 T174 or T177–T181. Record new 012 evidence
and original references. Run no HIP commands before E. Unavailable native/container
checks remain unavailable, never PASS. The mandatory operator acceptance and the
optional, separately frozen research campaign have distinct completion records.

## Phase A — Routine setup and foundational contracts

**Goal**: Closed interfaces, independent fixtures and legacy protection ready for B.
No scheduler, quota, deadline or transaction algorithm belongs in scaffolding.

### A1 — Establish authority and record boundaries

- [X] T001 Inspect merged 011 follow-ups and current handlers; record reuse versus new work, original input requirements, the T177–T181 carry-forward and immutable T174 disposition in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md` (new), referencing `specs/011-controlled-generalization/validation/merge-closeout.md` without editing historical task completion.
- [X] T002 Define closed record/schema shells for `PAC-screen/v1`, `pac-exp3s-leitner/v1`, inventory, observations, review/adaptive state and receipts in `src/oeis_learn/generalization/contracts.py` and `src/oeis_learn/generalization/schemas/pac.schema.json` (new); transcribe every field/visibility constraint in `data-model.md`, reject unknown fields/versions at new boundaries, and preserve historical readers. Leave policy validation for B and canonical dispatch for C.
- [X] T003 [P] Add independently specified kernel fractions, queue timelines, structural counts and fault-boundary input fixtures to `tests/fixtures/generalization/pac-policy.json` (new) and document their hand-derived provenance in `tests/fixtures/generalization/README.md`; do not derive expected results with production code or label prior planning calculations production tests.

### A2 — Legacy guardrails and coverage ownership

- [X] T004 [P] Add old-study/P3/schema rejection fixtures in `tests/unit/test_generalization_legacy.py` and `tests/unit/test_generalization_contracts.py`; preserve original recipe meanings and reject an old P3 relabeled PAC, unknown fields and `visit-scheduler/v1` as adaptive state.
- [X] T005 Record every original failure class and added policy risk against the task IDs in this document in `specs/012-parameterized-adaptive-curriculum/reliability-review.md`; retain prevention, counterexample, evidence and recovery columns and distinguish proven failures from suspected mechanisms.
- [X] T006 Check shells/fixtures and existing portable host help without HIP; record commands, results and unresolved dependency availability in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`, and confirm `scripts/foundation_run.py` and `scripts/generalization_inputs.py` still work under `python3 -S` before entering B.

## Phase B — Critical algorithms and state invariants

**Entry**: A complete. **Exit**: Pure policy counterexamples pass and all new
mathematical, admission, transaction and resource rules are fixed for C to consume.

### B1 — US1: Structural admission and comparison identity (P1)

- [X] T007 [P] [US1] Add independent draw-identity, all-16-exclusion, native-disagreement and post-exclusion quota-deficit counterexamples in `specs/012-parameterized-adaptive-curriculum/validation/test_structural_policy.py` (new); preserve historical 256-task guards and prove a completed shortage cannot become a timeout/reseed retry.
- [X] T008 [US1] Implement a versioned structural-scope adapter in `specs/012-parameterized-adaptive-curriculum/validation/structural_policy.py` (new), reusing `specs/011-controlled-generalization/validation/prospective_evaluation_policy.py`: retain all 2,288 draw IDs, seed/offset/counter keys and identity reservoir capacity 8,192, all 16 exclusions, exactly 128 tasks/27 families (68 certified/18 families, 26 ambiguous, 34 unresolved), 20/100 terms, eight attempts, 60-second portfolios, seven controls and bank seeds; parameter endpoint DEFERRED, no assessment execution.
- [X] T009 [US1] Add comparison-freeze/control counterexamples to `tests/unit/test_generalization_execution.py`, then implement `PAC-screen/v1` admission and compatibility rules in `src/oeis_learn/generalization/execution.py` and `src/oeis_learn/generalization/experiments.py`: require exact initializer/order/model/optimizer/exposure/diagnostic/backend proof for matched A, otherwise explicit reference-only or predeclared fresh A; default B/C and D off, no trained initializer, outcome-driven arm switch or fabricated historical measurements. Preserve "Frozen before outcomes; no old P3 relabeling; public safe projection" and "Missing proof is not match; private paths/weights excluded from shared output".
- [X] T010 [US1] Add finite pool/probe/anchor admission tests in `tests/unit/test_generalization_cohorts.py`, then extend `src/oeis_learn/generalization/cohorts.py` and `src/oeis_learn/generalization/exclusions.py` for 32-family/32,768-row identity, eight independent probes/family from at most 256 frozen draws/family, exact source/full-output duplicate rejection, custodian final-evaluation collision checks, disclosed shared training windows, four distinct anchors/family by frozen hash/strata and enough distinct batch rows; enforce "No empty family; exact counts; learner receives training role only", with no silent pool removal, reseeding or quota relaxation.

### B2 — US2: Paired target safety (P1)

- [X] T011 [P] [US2] Extend independent grounding/sharing/control-dependency/over-eight-parameter counterexamples in `tests/unit/test_generalization_refresh_targets.py`; verify every original row survives as an exact paired target or counted whole-row literal fallback, including repeated constants and unsafe guards.
- [X] T012 [US2] Reuse and only where necessary fix paired-target policy in `src/oeis_learn/generalization/targets.py`; enforce "Equal underlying rows; fallback retains original literal; private source handling follows existing dataset boundary", identical final-source validation/fitter/at-most-eight-parameter limits, and no partial-control parameterizer or eligibility drift between B/C.

### B3 — US3: Published kernel, reward and deterministic streams (P1)

- [X] T013 [P] [US3] Add hand-calculated Graves Eq. 1 tests in `tests/unit/test_generalization_curriculum.py` (new): first alpha=1/2, probabilities (35/96,61/192,61/192) for positive reward and (37/120,83/240,83/240) for negative reward under the fixture constants, zero-reward q=(.5,.3125,.1875), missing unchanged, selection-time denominator, extreme logweights/nonfinite rejection and duplicate/forbidden feedback; distinguish Graves sharing from Auer and the legacy heuristic.
- [X] T014 [US3] Implement the pure stable log-domain Graves EXP3.S kernel in `src/oeis_learn/generalization/curriculum.py` (new) with K>=2, epsilon=.25, eta=.001, beta=0, alpha=1/(valid_updates+2), replacement block draws and stored selection probability; derive K=32 bounds 1/128 and 97/128, apply sharing for observed zero only, and reject custom caps/priority mixing and legacy state.
- [X] T015 [US3] Add independent scaler/stream cases to `tests/unit/test_generalization_curriculum.py`, then implement signed loss gain per actual charged block token, past-only linearly interpolated 20/80 quantiles, fewer-than-five/tied warmup-zero markers, the 4,096-value uniform reservoir/seen counter/separate RNG, deterministic two-of-eight probe rotation and independent seeded per-family Fisher–Yates streams in `src/oeis_learn/generalization/curriculum.py`; cover outliers, wrap/skip cursors, checkpoint continuation, review filler without duplicate IDs, and preserve raw as well as scaled gain.

### B4 — US3: Per-anchor review and feedback coverage (P1)

- [X] T016 [P] [US3] Add independently expected acquisition/promotion/demotion/incidental/missing-grade timelines and all 32 cycle-offset service simulations in `tests/unit/test_generalization_review.py` (new), including 128 synchronized due anchors, repeated failure, impossible capacity, terminal truncation and no-due fallback; assert 512-update due-delay and 8,704-update exposure bounds only while training continues.
- [X] T017 [US3] Implement the pure 24-bandit-plus-eight-review-update policy in `src/oeis_learn/generalization/review.py` (new): one obligation/anchor, `(due_update, anchor_id)` ordering, acquisition without grade, queues 0..4 and 512*2^q intervals, anchor-specific pre-update recall transition only after committed practice, incidental clock reset without promotion and uniform-family fallback; enforce "Grade belongs to that anchor; initial acquisition and incidental exposure separate; no promotion without committed update". Report 25% review opportunities in updates and 18.75% full-cycle implicit uniform mass, not token guarantees.
- [X] T018 [US3] Add coverage/budget stop counterexamples in `tests/unit/test_generalization_diagnostics.py` (new), then encode feedback disposition in `src/oeis_learn/generalization/diagnostics.py`: missing pre/post retains debits but no kernel/scaler update; two consecutive censored rewards or required-reward budget exhaustion pause INCOMPLETE; two eligible review blocks without valid acquired-anchor grades pause; two semantic rotations without a family result mark retention incomplete. Missing is never zero or mastery.
- [X] T019 [US3] Implement frozen monitoring and observer schedules in `src/oeis_learn/generalization/diagnostics.py` with tests in `tests/unit/test_generalization_diagnostics.py`: common eight-probe/family initial/final/every-256-update loss panel, four-family rotating one-attempt/five-second semantic panels, B's three nominal uniform shadow reward opportunities and up to eight shadow anchor observations, C's actual grades, optional D's four reward blocks; keep policy-required counts/costs separate and forbid monitoring-based kernel credit or anchor promotion.

### B5 — US4: Transactions, restore and discarded work (P1)

- [X] T020 [P] [US4] Add an independent state-transition/fault matrix in `tests/unit/test_generalization_curriculum_state.py` (new), covering reservation, successful/failed optimizer update, pre/block/post observation, pending reward, duplicate IDs, review grade, snapshot publication, stale journal and terminal partial block; require "At most one pending learner update within one pending block; never repeat completed gradients or apply feedback twice".
- [X] T021 [US4] Implement model-bound pending-block and review transaction transitions in `src/oeis_learn/generalization/curriculum_state.py` (new) and bind them to `src/oeis_learn/generalization/state.py`: one chosen family/eight contiguous updates, retained pre-panel/probability, no intervening review/other gradient, exactly-once scaler/kernel completion before next decision, atomic post-optimizer anchor transition, separate action/valid/update/service counters and persistent pause reasons. Enforce "Same panel around one contiguous family block; common trends are monitoring, not reward credit; no false zero for unavailable".
- [X] T022 [US4] Add complete-generation/discarded-segment tests in `tests/unit/test_generalization_persistence.py`, then extend `src/oeis_learn/generalization/checkpoints.py` and `src/oeis_learn/generalization/state.py` to enforce "Atomic generation manifest hashes every component; weights-only is not resumable"; enforce "Named kernel and epsilon-derived probability bounds; reject old visit-scheduler state; complete identity-bound restore"; inventory/policy/source/environment mismatch rejects restore, newer journal only debits lost work, and recomputed work receives new segment IDs without refunding tokens/time or replaying newer policy on old tensors.
- [X] T023 [US4] Validate the combined pure state against uninterrupted and interrupted traces in `tests/unit/test_generalization_curriculum_state.py`, including reservoir/row/probe RNGs and cursors, censored streaks, pending pauses, failed review updates and partial blocks; record exact clean-resume versus charged crash-loss expectations in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md` before runtime binding.

### B6 — US4: Complete deadline and peak-storage arithmetic (P1)

- [X] T024 [P] [US4] Add independent exact-boundary/over-boundary nested time and peak-live byte fixtures in `tests/unit/test_generalization_operational_budgets.py` and `tests/unit/test_generalization_budgets.py`; cover existing longer limits, absent measurements, slow cold startup/load/save/hash/copy/exit, optional fresh A/D, resumed proof, finalization, simultaneous old/new/pinned snapshots and recovery packages, and reject larger outer limits masking a short inner/schema cap.
- [X] T025 [US4] Implement PAC resource arithmetic in `src/oeis_learn/generalization/budgets.py` and `src/oeis_learn/generalization/resources.py`: max(existing justified, calculated), preparation >=24h and learner >=12h raised by 3*forecast_work+2h plus startup/publication, generation >=6.5h or longer prior allowance, >=3*largest representative ancillary cost plus overhead, summed serial/explicit parallel critical-path parents and >=1h finalization plus cleanup floors; include complete diagnostic forecast/reservation (initial 20%, enlarged before freeze), disk-backed scratch/all live copies, no silent cap/zero measurement/refund/automatic midrun extension, and unchanged scientific 60s/5s budgets.

### B7 — US5: Denominators and permissible conclusions (P2)

- [X] T026 [P] [US5] Add independent equal-total/different-family, paired-discordance, missing-arm and censored-completion fixtures to `tests/unit/test_generalization_metrics.py`; distinguish 64/68 from 65/68 complete, ambiguity from unresolved execution, task resampling from seed uncertainty, and token allowance from useful exposure or matched wall time.
- [X] T027 [US5] Implement/reuse structural aggregation and comparison admission in `src/oeis_learn/generalization/metrics.py` and `src/oeis_learn/generalization/reconciliation.py`: fixed certified family-macro denominator, paired task counts, >=65/68 per compared arm, separately reported ambiguous/unresolved/censored strata, B–A only with matched A, C–B package, optional D–B allocation and C–D review substitution, explicit missing history and no automatic winner or positive-gain acceptance gate.

## Phase C — Critical installed execution and provenance integration

**Entry**: All B invariants tested. **Exit**: Canonical installed CPU path binds
actual updates, policy state, evidence and resources; actual device proof awaits E.
No broad reporting/wrapper work belongs here.

### C1 — US1: Canonical PAC authority and admission (P1)

- [X] T028 [US1] Add canonical PAC registration/role authority tests in `tests/integration/test_generalization_registration.py` and `tests/contract/test_generalization_authority.py`, including all-original-input gating before research generation/learning and rejection of a wrapper-only permissive policy bypass.
- [X] T029 [US1] Register PAC through `src/oeis_learn/pipeline_validation/generalization_execution.py`, `src/oeis_learn/pipeline_validation/generalization_setup.py`, `src/oeis_learn/pipeline_validation/generalization_commands.py` and `src/oeis_learn/generalization/role_contracts.py`; bind frozen comparison/control/cohort policy and source/image identities to canonical CLI requests, keeping old recipes unchanged and protected selection/assessment truth outside learner roles.

### C2 — US2: Installed paired preparation and batch binding (P1)

- [X] T030 [US2] Add real prepared-input/target-sidecar/role isolation integration cases in `tests/integration/test_generalization_preparation.py` and `tests/integration/test_generalization_training.py`; include corrupted family hashes, missing probe quota, protected feedback injection, control fallback, identical concrete rows and fixed B order with actual encoded batches.
- [X] T031 [US2] Bind original corpus, hash-bound family sidecar, finite probes/anchors and paired target proofs through `src/oeis_learn/pipeline_validation/generalization_preparation.py` and `src/oeis_learn/generalization/preparation.py`; execute native/reference admission before research, keep exact data/probe role boundaries and emit private proof plus safe coverage/deficit summaries.
- [X] T032 [US2] Extend `src/oeis_learn/pipeline_validation/generalization_training.py` to retain family/row IDs through real prepared batches, reuse existing learning target/fitter support and identical fresh B/C initialization, and charge exact canonical target tokens without clipping batches to fake 25M equality; fixed B remains fixed and acceptance tensors cannot become research initialization.

### C3 — US3: Actual contiguous feedback and review updates (P1)

- [X] T033 [US3] Add installed real-tensor CPU tests in `tests/integration/test_generalization_curriculum.py` (new) for pre/eight-updates/post binding, non-padding sums/counts, RNG/mode restoration, no intervening gradient, nondegenerate scaler reward, changed finite parameters, anchor acquisition then grade/update, filler incidental exposure and failed optimizer non-promotion; mocks may inject failures but cannot substitute for the learner.
- [X] T034 [US3] Wire pure curriculum/row/probe policy into the existing learner using `src/oeis_learn/pipeline_validation/generalization_curriculum.py` (new minimal adapter) and `src/oeis_learn/pipeline_validation/generalization_training.py`; execute actual contiguous family blocks, matching no-gradient pre/post probes with mode/RNG preservation and synchronization as required, charged review batches and exact anchor recall excluding BOS/pad but including EOS, committing policy state only at the B-defined boundaries.
- [X] T035 [US3] Wire common panels, shadow B observers, optional D observations and persistent incomplete/stop rules through `src/oeis_learn/pipeline_validation/generalization_diagnostics.py` and `src/oeis_learn/pipeline_validation/generalization_curriculum.py`; test scheduling and accounting in `tests/integration/test_generalization_curriculum.py`, separating target recall, semantic execution, censored probes and policy-conditioned reward from matched final evaluation.
- [X] T036 [US3] Exercise production eight-update/32-update clocks and all charged channels through installed CPU fixtures in `tests/integration/test_generalization_curriculum.py`; verify empirical counts, floor-derived selection behavior, fixed B order, D-off default and resumed row/probe choices against the pure policy oracle, recording evidence in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`.

### C4 — US4: Full-state snapshots and execution envelope (P1)

- [X] T037 [US4] Add installed pause/crash/publication tests to `tests/integration/test_generalization_runtime.py` before persistence edits: pause inside block and after gradients/before reward, fail before/after grade/update/debit, lose unsaved segments, ENOSPC mid-publication, reject weights-only/identity mismatch, and compare actual optimizer/tensors/RNG/scaler/queue/next batch on clean resume.
- [X] T038 [US4] Extend `src/oeis_learn/pipeline_validation/generalization_persistence.py` and `src/oeis_learn/pipeline_validation/generalization_runtime.py` to snapshot/restore complete model-bound adaptive state atomically, including pending observation and all counters/RNGs/ledgers; resume it before new gradients, preserve old verified generation and discarded-work accounting, retain hourly full saves plus initial/final/4/8/16/25M pins with deduplication and the constraint "Small bounded journal; not authority to restore newer scheduler state".
- [X] T039 [US4] Add end-to-end requested/effective limit and environment propagation tests in `tests/integration/test_generalization_recipes.py` and `tests/contract/test_generalization_roles.py`, including every acceptance/calibration/resume/proof/export branch, a stale PASS file with failed process and a changed no-Lt/engine identity.
- [X] T040 [US4] Bind resource graph, actual engine/source/image identities and existing no-Lt process-start environment through `src/oeis_learn/pipeline_validation/generalization_execution.py`, `src/oeis_learn/pipeline_validation/generalization_runtime.py`, `src/oeis_learn/container_runtime.py` and relevant registered schema bounds in `src/oeis_learn/generalization/schemas/pac.schema.json`; propagate unclipped inner/outer limits, cleanup/publication reserves and private disk scratch, retaining canonical containment and no silent CPU fallback.
- [X] T041 [US4] Extend canonical acceptance coverage in `src/oeis_learn/generalization/feature_acceptance.py` and `src/oeis_learn/pipeline_validation/generalization_acceptance.py`, with real-handler tests in `tests/integration/test_generalization_acceptance.py`: derive encoded token/update minima, declare reduced acceptance cadence distinct from production, reach parameter/fallback, non-warmup reward, acquired due graded review, both pending-block resume boundaries, initial/final load and fault export; implement HIP-marked proof cases in `tests/integration/test_generalization_gpu.py` without executing HIP here.
- [X] T042 [US4] Run installed CPU lifecycle/fault fixtures and inspect source closure in `tests/integration/test_generalization_curriculum.py`, `tests/integration/test_generalization_runtime.py` and `tests/integration/test_generalization_acceptance.py`; record exact clean-resume equality and separately charged crash work in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`, identify unavailable native/device checks explicitly and close C only when all local critical correctness gates pass.

## Phase D — Routine delivery, diagnostics and complete local validation

**Entry**: C complete. **Exit**: A tested, stdlib-only serial handoff composes canonical
operations, every reliability row has local evidence, and only actual workstation
inputs/native capacity/device evidence or optional research remain outstanding.

### D1 — US1: Structural and historical-control handoff (P1)

- [X] T043 [P] [US1] Add complete/missing/incompatible original-input fixtures in `specs/012-parameterized-adaptive-curriculum/validation/test_inputs.py` (new), covering bounded-search exhaustion with an explicit-root remedy, all 16 exclusions, weights-only provenance, matched versus reference-only controls and immutable failed historical records.
- [X] T044 [US1] Implement read-only explicit-root/inventory and control-audit assembly in `specs/012-parameterized-adaptive-curriculum/validation/inputs.py` (new), invoking canonical validators; inventory the named original chain in `quickstart.md`, emit exact missing requirements and MATCHED/REFERENCE_ONLY reasons, and bind structural scope/deficits using B policies rather than recoding them.
- [X] T045 [US1] Connect scoped preparation, retained checkpoint inference and publication to existing canonical operations in `specs/012-parameterized-adaptive-curriculum/validation/run.py` (new); keep helper versus producer identity separate, apply admission before any research inference, and validate fixture receipts/rejection paths in `specs/012-parameterized-adaptive-curriculum/validation/test_run.py` (new).

### D2 — US2: Representation evidence and inference parity (P1)

- [X] T046 [P] [US2] Add report projection and evaluation parity fixtures in `tests/integration/test_generalization_reporting.py` (new) and `tests/integration/test_generalization_scoring.py`, covering supported/fallback fractions by family, row accounting, fitted assignments, identical attempt/fitter limits and forbidden private-source exports.
- [X] T047 [US2] Wire paired-target coverage, fitter parity and tokens/examples/updates/cost projection into `src/oeis_learn/pipeline_validation/generalization_reporting.py` and `src/oeis_learn/pipeline_validation/generalization_scoring.py`, preserving exact raw/fitted distinctions and explicit unavailable evidence; do not label equal token allowances equal-data or equal-time comparisons.

### D3 — US3: Allocation, retention and complete cost evidence (P1)

- [X] T048 [P] [US3] Add reconstructibility/long-run/omitted-observation fixtures in `tests/integration/test_generalization_diagnostics.py`, including more than 128 progress windows, fewer realized grades than opportunities, optional D overhead, missing historic curves and positive scaled reward with negative raw gain.
- [X] T049 [US3] Extend bounded policy/retention diagnostics and progress projections in `src/oeis_learn/pipeline_validation/generalization_diagnostics.py` and `src/oeis_learn/pipeline_validation/generalization_reporting.py` to expose all clocks, selected probabilities/reasons, acquisition/recall/incidental exposure, pending anchors, raw/scaled gain, censoring, useful/reserved/discarded/unspent tokens and complete preparation/probe/review/checkpoint/end-to-end costs; compact without losing whole-run coverage or reconstructible totals.

### D4 — US4: One serial wrapper and measured readiness (P1)

- [X] T050 [US4] Add dependency-graph, cancellation, no-GPU-before-admission, no-research-by-default and no-second-learner wrapper tests in `specs/012-parameterized-adaptive-curriculum/validation/test_run.py`; simulate all terminal dispositions and prove `recover` never launches work and `resume` repeats only the smallest incompatible/unfinished closure.
- [X] T051 [US4] Complete stdlib `prepare/run/resume/recover` orchestration in `specs/012-parameterized-adaptive-curriculum/validation/run.py`: freeze exact scope before outcomes, compose canonical handlers serially, reuse only sealed compatible stages, preserve failures, accept research only from an already authorized frozen manifest, and export automatically on success/failure without an intermediate manual command. No policy arithmetic, learner or fitter implementation in this helper.
- [X] T052 [US4] Add calibration/budget-binding fixtures in `specs/012-parameterized-adaptive-curriculum/validation/test_calibration.py` (new), then implement orchestration/report assembly in `specs/012-parameterized-adaptive-curriculum/validation/calibration.py` (new): short/long families, cold/warm native and early/later shards, production-sized actual update/save/load, complete reward/grade/common-panel cycle, synchronized costs/sample counts/worst observations/uncertainty; consume B's budget math and block research if projected work plus margins cannot fit.
- [X] T053 [US4] Validate existing deterministic bounded worker/cache reuse and containment in `tests/integration/test_generalization_preparation.py`, `tests/unit/test_pipeline_disk.py`, `tests/unit/test_pipeline_supervision.py` and `tests/unit/test_container_runtime.py`: fresh worker execution state, parallel/single equivalence, cancellation, nested symlink/escaping path, insufficient space, cold startup/long exit, scanner/watchdog handler lifecycle and fresh output ownership; make only necessary routine fixes in `src/oeis_learn/pipeline_validation/generalization_preparation.py`, `src/oeis_learn/pipeline_validation/disk.py` and `src/oeis_learn/pipeline_validation/supervision.py`, reopening B/C for invariant changes.

### D5 — US4: Terminal evidence and minimal recovery (P1)

- [X] T054 [P] [US4] Add real packer/verifier tests in `tests/integration/test_generalization_archive.py` and `tests/unit/test_generalization_acceptance_archive.py`: success, negative scope, interruption, missing nested result/COMPLETE/quota ledger, stale producer, process or cleanup failure after application PASS, export failure after primary failure, legitimate helper paths and traversal/private-payload rejection.
- [X] T055 [US4] Extend canonical public collection/finalization in `src/oeis_learn/pipeline_validation/generalization_archive.py`, `src/oeis_learn/pipeline_validation/generalization_finalization.py` and `src/oeis_learn/generalization/acceptance_archive.py` for PAC records; enforce "Reuse only complete compatible closure; primary/secondary errors both preserved", bind original report bytes/verifier/archive hash/terminal receipt, include bounded status/deadline/quota/progress/omission records and exclude weights/state/truth/private host paths.
- [X] T056 [US4] Wire independently verified minimal fallback and read-only recovery into `specs/012-parameterized-adaptive-curriculum/validation/run.py`, with actual packer fixtures in `specs/012-parameterized-adaptive-curriculum/validation/test_run.py`; leave original failure immutable, report last-resort location if export itself fails, and demonstrate stage reuse without pretending failed/unverified outputs passed.
- [X] T057 [US4] Exercise a complete fixture campaign, interrupted compatible resume and incompatible-source minimal retry through `specs/012-parameterized-adaptive-curriculum/validation/test_run.py`; assert command/config/source/image/helper identities, <=60s progress or explicit unavailable/stalled signal, application/process/cleanup outcomes, actual archive verification and no needless repeated successful stage.

### D6 — US5: Structural comparison and before/after case analysis (P2)

- [X] T058 [P] [US5] Add scoped reconciliation/case-selection fixtures in `specs/012-parameterized-adaptive-curriculum/validation/test_analysis.py` (new): equal totals with different family wins, 64/65 completion boundary, unmatched A, D absent/present, missing categories, stable ordering and raw/parameterized/fitted-source permissions; historical 256-task analysis must keep rejecting the new scope without the adapter.
- [X] T059 [US5] Implement a versioned 128-task adapter in `specs/012-parameterized-adaptive-curriculum/validation/analysis.py` (new) around `specs/011-controlled-generalization/validation/post_pilot_analysis.py` and existing canonical metrics; produce family-macro/task-paired/censoring and cost/retention comparisons with declared B–A/C–B/optional-D attribution and no winner or seed-generalization claim.
- [X] T060 [US5] Add deterministic selection and human-readable examination of at least 12 available case/arm records in `specs/012-parameterized-adaptive-curriculum/validation/analysis.py`, spanning shared/exclusive success, continuation failure, no-fit, timeout/decode and simple-control wins; retain permitted before/after sources/assignments/reasons, distinguish structural failure from solver/decoder/execution limitations and explicitly list unavailable categories without inventing examples.
- [X] T061 [US5] Reconstruct every reported aggregate and token/example/update/time total from fixture exports using `specs/012-parameterized-adaptive-curriculum/validation/test_analysis.py` and `tests/integration/test_generalization_evidence.py`; record scientific/software status separation and exploratory positive/negative/inconclusive examples in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`.

### D7 — Shared portability, regression and operator delivery

- [X] T062 Check closed PAC schemas, canonical CLI registration/help and every new Docker/Podman role command using `tests/contract/test_generalization_roles.py`, `tests/integration/test_generalization_registration.py` and `tests/unit/test_generalization_host.py`; verify stdlib-only host entry, explicit engine/mounts/no socket, image/source/environment equivalence and all effective deadline branches, recording unexecuted engine operations honestly in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`.
- [X] T063 Run the relevant combined non-HIP unit/contract/integration and 012 helper suites in the pinned environment after changes stabilize; record exact commands/results and skipped dependency reasons in `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`, fix failures in their owning groups and rerun only affected checks plus necessary closure. Do not run HIP or claim hardware proof from CPU results.
- [X] T064 Reconcile every task/FR/NFR/SC and all 18 reliability rows against concrete passing local checks and explicit E-only evidence in `specs/012-parameterized-adaptive-curriculum/reliability-review.md` and `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md`; stop handoff for unexplained gaps in contracts, quotas, transactions, packaging or nested limits.
- [X] T065 Publish and verify against actual help/fixture invocation one copy-pastable operator block in `specs/012-parameterized-adaptive-curriculum/quickstart.md`, with explicit existing branch/engine/original roots/output and plan identity, scientific purpose/data/arms/25M limits, measured ETA versus generous ceiling, resume/recover variants and conditional research behavior; share the block directly with the user at handoff, not only a document link.
- [X] T066 Update `specs/012-parameterized-adaptive-curriculum/plan.md`, `specs/012-parameterized-adaptive-curriculum/tasks.md` and `specs/012-parameterized-adaptive-curriculum/validation/implementation-ledger.md` to mark only evidenced local work complete, list actual workstation/HIP prerequisites and optional research separately, and commit/push the locally validated handoff without waiting for CI.

## Phase E — One consolidated operator session and evidence reconciliation

**Entry**: D complete. The operator runs **T068 once** on the normal compatible
path. T069–T073 inspect its packaged results; they are not additional HIP commands.
Failures use the same wrapper's minimal `resume` or read-only `recover`, recording
the reason. Keep hardware tasks unchecked until actual evidence is supplied.

### E1 — Freeze the exact authorized scope and run once

- [ ] T067 Record explicit operator scope (acceptance only by default; research B/C plus qualified/labeled A only if authorized, fresh A/D only if predeclared), original root inventory, source/image/plan identities and finite initial resource graph with explicit calibration gaps in `specs/012-parameterized-adaptive-curriculum/validation/operator-handoff.md` (new); the wrapper may raise and seal operational allowances from calibration before research but cannot alter scientific choices after outcomes or silently extend a running allowance.
- [ ] T068 [US4] Execute the single verified command from `specs/012-parameterized-adaptive-curriculum/quickstart.md` via `specs/012-parameterized-adaptive-curriculum/validation/run.py`: full CPU/native admission, needed qualification, representative calibration, real HIP acceptance and verified export, continuing to retained baselines and new serial arms only for the already frozen research scope; record actual invocation/outcomes in `specs/012-parameterized-adaptive-curriculum/validation/hip-acceptance.md` (new), retaining first-cause failures and smallest retry boundary.

### E2 — Verify evidence from that session

- [ ] T069 [US1] Inspect actual original-input/control/native/quota receipts from T068 and reconcile exactly 128/27 and 68/18+26+34 with all 16 exclusions and original draw identities in `specs/012-parameterized-adaptive-curriculum/validation/structural-admission.md` (new); retain parameter DEFERRED and explicit control disposition, block failed scientific admission and reference inherited 011 follow-ups without checking old tasks.
- [ ] T070 [US2] Verify real HIP changed finite parameters, actual checkpoint evaluation, matched initialization, target-pair grounding/fallback counts and inference parity from T068 in `specs/012-parameterized-adaptive-curriculum/validation/hip-acceptance.md`; no mocked updates or acceptance tensors as research initializers.
- [ ] T071 [US3] Verify actual pre/block/post nondegenerate reward, selected probability, kernel/scaler update, anchor acquisition then due graded optimizer update, observation costs and policy clocks from T068 in `specs/012-parameterized-adaptive-curriculum/validation/hip-acceptance.md`; distinguish reduced acceptance cadence from production clock proof and target recall from semantic success.
- [ ] T072 [US4] Verify actual fresh/resumed device/model/optimizer/RNG/data/adaptive/scaler/review equality at both pause boundaries, discarded-crash debit, no-Lt environment/source closure and independently checked success/failure archives from T068 in `specs/012-parameterized-adaptive-curriculum/validation/hip-acceptance.md`; keep any unavailable device/failure proof pending rather than inferring it from old 011 evidence.
- [ ] T073 [US5] If authorized research completed in T068, run the scoped analysis over its final reports, examine passing/failing before/after cases and reconcile costs/denominators/attribution into `specs/012-parameterized-adaptive-curriculum/validation/research-results.md` (new); otherwise record research DEFERRED with exact missing evidence. Only check this task complete for actual research analysis; explicit deferral does not block software acceptance and does not authorize another campaign.

### E3 — Close out with truthful completion

- [ ] T074 Reconcile actual evidence, software readiness, scientific interpretability, deferred optional research and all failure/retry boundaries in `specs/012-parameterized-adaptive-curriculum/validation/closeout.md` (new), `specs/012-parameterized-adaptive-curriculum/tasks.md` and `specs/012-parameterized-adaptive-curriculum/reliability-review.md`; commit/push public evidence references and bounded reports only, preserve private artifacts and historical failures, and do not auto-launch seeds/ablations or select a winner.

## Dependencies and incremental delivery

Phase order is **A -> B -> C -> D -> E**. Within a subgroup, tests precede
implementation. Cross-subgroup dependencies beyond the phase barriers are:

| Consumer | Required work |
| --- | --- |
| T009–T010 | T007–T008 structural identities and A closed records |
| T015 | T013–T014 kernel; T010 inventory contract |
| T017–T019 | T016 queue counterexamples; T014–T015 clocks/scaling; T010 finite inventory |
| T021–T023 | T020 fault matrix, T014–T019 policy, T009 frozen identity |
| T025 | T024 arithmetic cases and T009/T019 selected arms/observation graph |
| T027 | T026 independent cases and T008–T009 scope/control policy |
| T031–T032 | T028–T030 authority and preparation cases |
| T034–T036 | T031–T033 actual batches and installed counterexamples |
| T038–T042 | T034–T037 actual updates and restore tests; T039 before T040; T040 before T041 |
| T045 | T043–T044 validated original input assembly |
| T047/T049 | Their T046/T048 cases; C observations and role contracts |
| T051–T053 | T045 canonical orchestration and T050 wrapper tests |
| T055–T057 | T054 packer cases and T051 wrapper; T055 before T056 |
| T059–T061 | T058 analysis cases, T047/T049 reports and T055 exports |
| T062–T066 | All D story subgroups; handoff requires T063–T064 closure |
| T068 | T067 frozen operator scope and every D local gate |
| T069–T072 | Actual T068 evidence, not elapsed time or bare PASS |
| T073 | Authorized completed research in T068, T069 denominators and T070–T072 acceptance |
| T074 | T069–T072 verified or honestly blocked; T073 completed or explicitly deferred |

US1 establishes shared comparison identity before US2/US3 learning. US2's paired
batches precede US3's adaptive integration. US4 spans foundational recovery through
delivery and gates every operator story. US5 consumes their frozen data/evidence;
it cannot establish missing matched controls or device proof itself.

### Story goals, independent acceptance and parallel opportunities

| Story | Tasks | Independent acceptance | Disjoint-file parallel example |
| --- | --- | --- | --- |
| US1 (P1) | 10 | Small structural fixture rejects independent exclusions/deficit; full scope reconciles or blocks; original retained checkpoint/control audit requires no new learner | After A, T007 with US2 T011; after C, T043 with T048 |
| US2 (P1) | 8 | Every pair independently grounds or retains counted fallback; real matched small updates and equal fitter policy | After A, T011 with US3 T013; after C, T046 with T054 |
| US3 (P1) | 14 | Published numeric fractions, all queue offsets, signed/censored reward, actual charged review and deterministic decisions/restore | After A, T013 and T016 use separate test files; after C, T048 with T058 |
| US4 (P1) | 22 | Fault-injected actual learner/packer; exact clean restore, honest crash debit, smallest compatible retry and first/secondary errors preserved | After A, T020 and T024 use separate test files; after C, T054 with T058 |
| US5 (P2) | 7 | Equal-total/different-family fixtures reconstruct all aggregates, enforce 65/68 and matched attribution, select deterministic available cases | After A, T026 with T020; after C, T058 with T043 |
| Shared setup/polish/closure | 13 | Closed schemas, legacy compatibility, portability, full traceability and honest handoff | A T003/T004 are disjoint; later shared closure is serial |

**MVP**: US1's structural admission/control/reference analysis using existing
checkpoints, with no fresh learner. Demonstrate its software slice with fixtures
during B/C/D; perform its actual inference inside the same E invocation rather than
adding a separate HIP session. Incremental local checkpoints then add US2 paired
targets, US3 adaptive/review behavior, US4 full recovery and US5 analysis. These are
reviewable milestones within A–D, not permission to bypass the two critical phases
or launch research before complete admission and acceptance.

## Requirement traceability

All references below identify planned work, not passing evidence. Each task can
only be checked after its implementation and stated validation are available.

| Requirements | Task coverage |
| --- | --- |
| FR-001, FR-002, FR-003 | T002, T004, T009, T028–T029, T043–T045, T067, T069 |
| FR-004 | T010, T012, T030–T032, T070 |
| FR-005, FR-006, FR-007 | T001, T007–T010, T028–T031, T044–T045, T068–T069 |
| FR-008, FR-009 | T011–T012, T030–T032, T046–T047, T070 |
| FR-010, FR-011 | T010, T015, T018–T019, T028–T035, T048–T049 |
| FR-012, FR-013 | T013–T017, T020–T021, T033–T036, T071 |
| FR-014, FR-015 | T016–T017, T020–T023, T033–T038, T049, T071–T072 |
| FR-016, FR-017, FR-018 | T013–T015, T018–T025, T033–T036, T048–T052, T071 |
| FR-019, FR-020 | T002, T020–T023, T037–T042, T072 |
| FR-021 | T022–T025, T037–T040, T048–T049, T052, T072 |
| FR-022, FR-023 | T009–T010, T024–T025, T028–T031, T039–T045, T050–T057, T062, T067–T069 |
| FR-024, FR-025, FR-026 | T039–T042, T048–T057, T062–T066, T068, T072, T074 |
| FR-027, FR-028, FR-029 | T026–T027, T046–T049, T058–T061, T069, T073 |
| FR-030 | T033, T036–T042, T054–T057, T068, T070–T072 |
| FR-031 | T009, T019, T025, T029, T050–T052, T059, T065, T067–T074 |
| NFR-001, SC-010 | A–E boundaries, T001, T023, T042, T064, T066, T074 |
| NFR-002, SC-005 | T041, T050–T057, T065, T068, T072 |
| NFR-003, SC-009 | T005, T024, T037–T042, T053–T057, T062–T064, T074 |
| NFR-004 | T024–T025, T039–T040, T052–T053, T062, T065, T067–T068 |
| NFR-005, NFR-006 | T002, T004, T010–T012, T028–T032, T039–T040, T053–T056, T062 |
| NFR-007, SC-008 | T017, T022, T025, T038, T048–T049, T052–T053, T057 |
| NFR-008, SC-006 | T015, T019, T025–T027, T048–T049, T052, T061, T073 |
| SC-001 | T007–T010, T028–T031, T044–T045, T069 |
| SC-002 | T011–T012, T030–T032, T046–T047, T070 |
| SC-003 | T013–T019, T033–T036, T071 |
| SC-004 | T020–T023, T037–T042, T072 |
| SC-007 | T009, T026–T027, T058–T061, T069, T073–T074 |

## Reliability ownership before hardware

The full prevention/evidence/recovery definitions remain in [reliability-review.md](reliability-review.md).
Each row below names both implementation and required regression tasks; D's T064
must link actual check results, not merely cite this table.

| Original failure class / additional design risk | Implementation owner | Counterexample / proof | Required evidence and recovery |
| --- | --- | --- | --- |
| Engine/qualification environment mismatch | T029, T040 | T039, T062, T072 | Requested/observed engine/source/image/env; requalify incompatible closure |
| Fresh directory/host dependency | T044, T051, T053 | T043, T050, T053, T062 | Input/output owner; explicit root or safe sibling/resume |
| Cold startup/inspection/overlay delay | T025, T040, T052 | T024, T039, T053 | Startup/work/exit costs; amend frozen operational plan before retry |
| Nested learner/evaluation/persistence timeout | T025, T040 | T024, T039, T052, T062 | All requested/effective limits and exhausted boundary; last complete snapshot |
| Repeated preparation compile/hash cost | T031, T052–T053 | T030, T052–T053 | Cold/warm/later-shard throughput; sealed-cell reuse only |
| Finite quota deficit | T008, T010, T031 | T007, T010, T030, T069 | Original draws/exclusions, deficits; no reseed/blind timeout increase |
| Smoke never reaches actual boundary | T034, T041 | T033, T036–T037, T070–T072 | Actual update/reward/review/load counts and checkpoint hashes |
| Disk/tmpfs/live copy exhaustion | T025, T038, T040 | T024, T037, T053 | Peak required/free categories; preserve complete prior state/no auto-prune |
| Excess snapshots/truncated progress | T038, T049 | T037, T048, T057 | Save count/time and whole-run totals; hourly/latest verified generation |
| Process-to-process backend divergence | T038, T040 | T037, T039, T072 | Actual full-state equality and no-Lt closure; changed backend requalifies |
| Diagnostic interference/watchdog lifetime | T035, T053 | T033, T053, T057 | Mode/RNG/handler/cleanup status and ordered errors |
| Cross-image/newer-journal restore | T021–T022, T038 | T020, T022–T023, T037, T072 | Identity/pending/discarded segments; no policy-only replay |
| Missing nested outcome/COMPLETE/primary cause | T055–T056 | T039, T054, T057, T072 | App/process/cleanup/verifier, quota, omissions and ordered errors |
| Packer helper paths/stale producer | T055–T056 | T054, T056–T057, T072 | Verified archive and original producer/helper closure; minimal rerun |
| Wrong EXP3 variant/probability | T014–T015 | T013, T015, T036, T071 | Named kernel, selected p, raw/scaled reward, valid clock |
| Cross-family/double reward credit | T021, T034, T038 | T020, T023, T033, T037, T071–T072 | Panel/model/block IDs, exactly-once feedback, censored/discarded events |
| Wrong anchor grade/service bound | T017, T021, T034 | T016, T020, T033, T036, T071 | Anchor grade/exposure/queue, cycle offsets, pending terminal items |
| Observation cost/short timeout recurrence | T019, T025, T035, T052 | T018–T019, T024, T039, T048, T052, T071 | Complete probe/grade/panel counts/costs; required-feedback pause |

## Completion accounting

74 tasks: A 6, B 21, C 15, D 24, E 8. Story totals: US1 10, US2 8,
US3 14, US4 22, US5 7; shared 13. Implementation completed A/B/C/D T001–T066 (66 tasks);
all eight E tasks T067–T074 stay unchecked. Every implementation task has an incremental
implementation or counterexample commit; local evidence is recorded in the ledger.
T073 is conditional research analysis and may remain explicitly deferred at
software closeout. Failed/missing admission or device proof is a real blocker to
its dependent claims; no favorable scientific result is required for software PASS.


## Post-D operator failure repairs (2026-11-05)

The operator supplied a verified negative archive and requested both the scientific
quota blocker and nested-cause reporting be fixed. These follow-ups do not mark E
complete or rewrite the failed 27-family run. The revised evaluation scope must be
frozen before any model outcomes; training, exclusions and draw law stay unchanged.

- [X] F001 Add the original public failure counters and independent worst-duplicate source-capacity/old-scope rejection regressions before implementing the amendment.
- [X] F002 Freeze and implement a new 128-task/23-family structural scope (68 certified/15 families), preserving all 2,288 draws, all 16 exclusions and historical v1 readers; publish exact changed quotas and removed families.
- [X] F003 Bind the new scope through admission, scoring, metrics, analysis and offline archives; require exact new population identity and preserve historical denominator guards.
- [X] F004 Add nested-role/first-cause/cleanup/export failure counterexamples; preserve bounded safe cause types and quota details without private paths, truth or tensors.
- [X] F005 Implement bounded nested failure capture and verified public export/recovery for new and existing failed runs, keeping original records immutable.
- [X] F006 Run affected and combined non-HIP regression closure, update fingerprint and operator instructions, commit/push repairs and stop without HIP or CI wait.
