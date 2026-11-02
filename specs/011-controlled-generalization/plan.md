# Implementation Plan: Controlled Generalization Experiments

**Branch**: `011-controlled-generalization` | **Date**: 2026-10-30 | **Spec**: [spec.md](spec.md)

**Input**: `specs/011-controlled-generalization/spec.md`, including FR-035–036's explicit pilot sequence.

## Summary

Implement the complete evaluation, corpus-refresh, target-comparison and verified-replay capability, then use it for a staged research campaign. The operator-approved next research run is a fresh paired static-versus-refreshed pilot; historical E0 evaluation and P0 seed replication are deferred. Execution order does not remove broader changes from implementation scope.

Use the existing portable host wrapper, container roles, exact native fitter, independent reference evaluator, checkpoint store, pipeline runner and evidence packaging. Add a prospective `generalization` policy layer with strict role-specific data contracts. Separate qualification identity from study identity so canonical endpoints can share one original qualified producer without weakening study provenance.

The Spec Kit planning workflow's Phase 0/1 below creates design documents only. Actual implementation has four phases A–D, with **B as the sole complex-algorithm phase**. Do not confuse research stages E0/P0/P1/etc. with implementation phases.

## Technical Context

**Language/Version**: Host Python standard library, project Python >=3.11; pinned containers use Python 3.12. Native Rust 1.90.0 is selected in Dockerfile.

**Primary Dependencies**: Existing locked PyTorch stack (GPU base ROCm 7.2.1 / PyTorch 2.9.1), native synthesis core, existing exact affine/SMT path, NumPy and existing JSON validation. No new host or runtime package is required by this design; container locks remain authoritative.

**Storage**: Content-addressed JSON/JSONL manifests and immutable shards in the existing workspace; private checkpoint/state files; bounded shareable evidence archives. No external database/service.

**Testing**: Existing pytest unit/contract/integration suites, native conformance tests and separately marked HIP tests. Add independent counterexamples for admission, isolation, arithmetic grounding, selection, resume and qualification matching.

**Target Platform**: AMD Ryzen AI Max+ 395, 128 GB shared memory, 2 TB SSD, NixOS host with Docker/Podman connection. One learner; bounded native workers. No driver changes.

**Project Type**: Existing Python/Rust research CLI and container pipeline.

**Performance Goals**: One normal serial hardware acceptance invocation; one execution per required qualification scope. Preserve eight-attempt/60-second target portfolios and report censoring. No unmeasured throughput claim. T113 raises the acceptance-only ceiling to 395 minutes (6h35), including startup and recovery: retained evaluation, refresh and target-format each reserve25 minutes and replay/resume100 minutes. Generated evaluation envelopes are720 seconds for the two retained checkpoints and1200 seconds for each post-training policy. Replay learners remain600 seconds; small-model learners and all proofs get180 seconds, and container cleanup up to30 seconds inside each role envelope. T111 reserves generator process startup/exit separately inside work and charges setup/model loading before generation. Token/model/seed limits, per-portfolio semantics and research recipes are unchanged; this is a ceiling, not a predicted duration. Acceptance-013 completed both paired learners but timed out at approximately72 seconds during generator process completion.

**Constraints**: Exact source integers/indices, strict mechanical provenance, visible-only proposal/fitting, isolated selection supervision, sealed assessment, conservative dependency closure, finite generation/search/storage. Full research runs are separate from qualification.

**Fresh-start delivery (T132–T137)**: The temporary runner under `validation/` binds
a new P1 screen protocol before learning, runs one current-source acceptance and then
canonical preparation/training plus the existing isolated generator/scorer roles.
The operator-authorized command has a60-hour outer ceiling after builds, including
finalization reserves; each learner remains at6 hours/25M tokens. Qualification and
research use64GiB/four CPUs. Both latest checkpoints are evaluated with frozen matched
controls; partial exposure remains descriptive, no adaptive checkpoint ranking or
assessment scoring occurs. Complex quota/exclusion and persistence repairs are in
the single reopened Phase B; temporary orchestration/export/handoff are routine C/D.
See [the handoff](validation/fresh-pilot-readiness.md). Actual new-source HIP and
scientific results remain pending, regardless of local test completion.

**Diagnostic containment (T100)**: Automatic role/image-inspection diagnostics use
timestamp-only heartbeats and explicit phase markers, never asynchronous traversal
of another thread's Python frames. External process/container deadlines remain
authoritative even when the GIL prevents a heartbeat. Diagnostic liveness is not
proof of progress or successful execution.

**Scale/Scope**: Historical baseline plus four new selection/assessment cells of 128 rows each; up to 25M target tokens per fresh research arm; three paired seeds for confirmation; 4,096 replay entries, two/task, at most 25% of batch. Explicit operational ceilings are in [contracts/experiments.md](contracts/experiments.md).

**Replay runtime routing (T102)**: Isolated replay audits run on CPU in the same
immutable image as the learner (`runtime.images[1]`), without GPU access. Admission
and restore therefore use the same native binary, guest, engine policy and
reference implementation. The qualification image remains responsible for its CPU
suite and other roles. Exact replay evidence comparison remains mandatory; no
cross-image equivalence or identity substitution is introduced.

## Constitution Check

Gate assessed before research and again after design: **PASS at design level**, no amendment or justified violation. Runtime gates remain unexecuted.

| Principle | Design evidence / remaining execution gate |
| --- | --- |
| I: Exact values and numerical semantics | Preserve bigint profile, signed canonical integer strings and indices. Lossy encoder features never replace authoritative values. Parameter grounding checked independently. |
| II: Supported language and acceptance | Reuse whole-program protection, bounded affine/SMT and final-source verification. No new primitive, solver or learned correctness oracle. |
| III: Bounded/recoverable operation | Explicit aggregate/time/storage ceilings, crash ledger, external deadlines, atomic state commits and partial bundles. Test containment in B/D. |
| IV: Workstation and reproducibility | Container-pinned dependencies, complete update-boundary resume, real HIP parameter update/load evidence in D. Documentation does not qualify hardware. |
| V: Provenance/isolation | Separate train/selection/assessment schemas and mounts; historical exposed diagnostics remain exploratory. Selection supervision is not hidden evaluation. No OEIS replay. |
| VI: Honest claims | Finite agreement, completion bounds, seed variation and scientific/software statuses remain distinct. No proof/novelty promotion. |
| Workflow | All complex work in B; local contracts first, consolidated HIP session after integration; long runs await actual gates. |

## Project Structure

### Documentation (this feature)

| Path | Purpose |
| --- | --- |
| `spec.md`, `checklists/requirements.md` | Requirements and specification quality validation |
| `research-review.md`, `source-register.*`, `citation-inventory.json` | Existing primary-paper assessment and pilot evidence |
| `research.md` | Implementation choices and resolved technical questions |
| `data-model.md` | Entities, invariants, role permissions and transitions |
| `contracts/cli.md` | Proposed canonical CLI behavior and compatibility |
| `contracts/artifacts.md` | Study, qualification, checkpoint and bundle contracts |
| `contracts/experiments.md` | Frozen recipe definitions, staged pilots and budgets |
| `quickstart.md` | Post-implementation validation and run guide |
| `tasks.md` | Future speckit-tasks output; intentionally not generated here |

### Source Code (repository root)

| Path | Planned responsibility |
| --- | --- |
| `src/oeis_learn/generalization/` (new) | Pure policies: contracts, cohorts, refresh, targets, metrics, selection, replay_policy, budgets. All nontrivial logic and reference tests in B. |
| `src/oeis_learn/pipeline_validation/` | Adapt existing runner/adapters/receipts/reuse/bundle; add prospective generalization orchestration and qualification contract. |
| `src/oeis_learn/data/bigint_programs.py` | Structural-descriptor/variant API for fixed-support refresh; preserve historical counter behavior. |
| `src/oeis_learn/fresh/{bigint_worker,bigint_state,replay}.py` | New profile routing, full resume, checkpoint pins, versioned task-first replay. |
| `src/oeis_learn/experiments/source_snapshot.py` | Include runtime-authoritative 011 contracts and new modules/helpers in conservative closure. |
| `crates/oeis_synthesis_core/src/` | Reuse parameters/dependence/fit/affine; change only if a demonstrated safety defect requires it, inside B. |
| `scripts/foundation_run.py`, `scripts/foundation_local.py` | Preserve stdlib entrypoint; route new canonical command family through existing dispatcher. |
| `scripts/generalization_hip_serial.py` (new temporary helper) | Serial subprocess orchestration only; no independent acceptance logic. Freeze before qualification. |
| `configs/foundation/pipeline_studies/` | New content-addressed prospective recipes, independent of compiled-source identity. |
| `tests/unit/`, `tests/contract/`, `tests/integration/`, `tests/fixtures/` | Pure reference cases, contract/mount tests, fixture workflows and real HIP evidence. |

**Structure Decision**: Extend the current single project. Separate policy from I/O so one focused algorithm phase can establish stable contracts for routine integration. Historical recipe/profile routes remain untouched.

## Phase 0 — Research completed for planning

The project's plan skill requested research delegation. Read-only research examined runtime/qualification and learning/data integration independently; conclusions were reconciled with current code in [research.md](research.md). No additional primary-paper claims were needed beyond the earlier review. All technical questions have design decisions; hardware feasibility is a measured execution gate, not an unresolved design placeholder.

## Phase 1 — Design completed for planning

Entities and access boundaries are in [data-model.md](data-model.md), public contracts in [contracts/cli.md](contracts/cli.md) and [contracts/artifacts.md](contracts/artifacts.md), research progression in [contracts/experiments.md](contracts/experiments.md), and acceptance scenarios in [quickstart.md](quickstart.md). Post-design constitution review passes as above. Proposed commands are not implemented or runnable yet.

## Implementation sequence (binding for speckit-tasks)

| Phase | Work | Exit condition |
| --- | --- | --- |
| A — Contracts and scaffolding | Register versions and schemas, role/mount contracts, interfaces, fixtures and dispatch skeletons. Enumerate counterexamples and test expectations. No algorithm implementation. | Contracts and independent expected results reviewable; every algorithm-bearing task assigned to B. |
| **B — All complex algorithms and reference tests** | Deterministic quota/isolation selection; structural-descriptor refresh and cursor; safe paired targets; accounting/metrics/oracle bounds; checkpoint selection and exact resume-state transitions; task-first replay/eviction; qualification compatibility and receipt integrity; milestone pinning and proposal-bank rules. Include independent/differential tests here. | All reference, safety, leakage, replay, resume and matching tests pass through stable interfaces. No known algorithm work remains for C/D. |
| C — Routine integration | Wire learner, host/container CLI, manifests, mounts, existing runner, telemetry, artifacts and serial helper to B interfaces; add end-to-end wiring tests and docs. | Full CPU/fake-adapter fixture workflow and failure packaging pass; source/image/helper closure frozen. |
| D — Consolidated validation and delivery | One serial real-HIP acceptance invocation, qualification reuse checks, representative new-path smoke/update/resume, archive integrity and concise report. | Complete qualifying evidence or explicit failure with minimal justified rerun; no automatic research campaign. |

B is the handoff boundary for a more powerful model. Its handoff includes this design, invariant matrix and golden counterexamples; its output includes reviewed algorithms, reference tests and stable interfaces. A discovered algorithm defect reopens B and affected gates. Do not implement a workaround in C/D and call it integration. Routine acceptance tests may run in D; designing arithmetic/admission/selection algorithms belongs to B.

## Test and evidence strategy

Operator-authorized diagnostic iteration (T114): a disposable script and its
tests live under this feature's `validation/` directory, outside the existing
production qualification inventory. Their hashes accompany every diagnostic
bundle. Reuse the original producer only through unchanged canonical compatibility
checks. Run selected independent acceptance cases without repeating the broad
qualification suites; a failed E0 still blocks its replay initializer dependency.
Small probes call the installed native admission/restore and actual HIP learner/
full-state resume paths. A traced retained probe preserves the original weights,
prompts and time budgets. Instrumentation logs bounded owner/worker events, never
cross-thread stack traversal. Neither selective execution nor traced probes confer
feature qualification. Full current-source acceptance remains the final gate.

- Preserve historical v3/v4 fixtures and CLI semantics; verify new profiles cannot silently alter them.
- Hand-computed metric examples include unequal family sizes, unresolved outcomes, failed completion eligibility and ties. Oracle unions never count as deployable selection.
- Independent safety cases include inactive control flow, impossible quotas, prefix/shift collisions in later shards, loop-carried parameter dependence, repeated nonlinear coefficients and full-horizon replay failures.
- Replay tests cover empty yields, duplicate variants, task-weight invariance, per-task/global caps, deterministic eviction and snapshot restoration.
- Resume compares uninterrupted versus interrupted execution on next batch IDs, RNG/data/replay state, token budget, checkpoint selection and device-appropriate weight agreement. A weights-only artifact must fail resume.
- Qualification tests vary source, unmapped files, image, actual GPU/runtime, precision, suites and resource contract; corrupt original PASS payloads and REUSED-only chains must fail.
- Integration failure injection proves fresh restart roots, verified stage reuse, bounded log tails and valid partial bundles. D proves real HIP loading/update and no CPU fallback.

## Complexity Tracking

No constitutional exceptions. The added policy layer and qualification contract are required to isolate experimental choices and permit safe cross-command reuse. A helper-level skip flag, editing frozen recipes, or simply toggling resume on the old screen would violate those requirements.

## Readiness

The [task breakdown](tasks.md) is generated and ready for implementation review. This planning change contains no runtime implementation and no new training/HIP measurements. Both the first pilot sequence and the complete implementation scope are explicit.


## Prospective HIP execution repair after isolated Linear reproduction

T128 freezes generalization-hip-execution/v1 in one stdlib descriptor. Prospective
GPU roles and the bigint-gpu qualification suite receive the four no-Lt/rocBLAS
startup flags before interpreter/BLAS initialization. Installed CUDA learner and
generator paths verify them and the accepted hipBLAS preference. A policy hash
binds CUDA full-state compatibility, current qualification runtime observations,
real-device resume evidence and stage receipt identity. Historical model commands
and the generic gpu qualification defaults retain their numerical settings.
This is one measured runtime-integration repair, not new sampling/admission or
selection algorithm work. Production promotion requires fresh source/image
qualification and the unchanged consolidated final acceptance gates.


## Post-acceptance readiness update — 2026-11-01 Pacific

Acceptance-016 completes D for the frozen ef886a7 source/images: canonical offline
verification passes all five cases and feature qualification;905 source entries
match exactly. Earlier planning statements about unimplemented/runtime-pending
work are historical. Science remains UNMEASURED. See
[acceptance review](validation/acceptance-016-review.md).

The [E0 readiness review](validation/e0-readiness.md) confirms existing core
implementation while identifying the missing concrete research authority and
public export/orchestration handoff. Prepare must resolve the frozen input study;
shared-bank evaluation needs its explicit digest argument. Research time envelopes
must account for the actual roster and per-checkpoint role allocation, setup,
scoring and packaging. No acceptance fixture is promoted to a research benchmark.
Keep the measured ROCm/no-Lt baseline; identify original research input metadata
before constructing and testing the single bounded E0 launch. No new algorithm,
production runtime, image or dependency change is part of this review.


## Fresh-start delivery — 2026-11-01 Pacific

The operator authorized the fresh P1 alternative in FR-035/036. Historical inputs
are now available; their target format is not a prerequisite for this experiment.
Retain the passing ROCm/no-Lt numerical policy. Acceptance-016 proves its frozen
software fixtures, not research-scale preparation or learner throughput.

Reopen one Phase-B group for demonstrated research-scale preparation and
persistence defects, with independent counterexamples before repairs. Follow with
routine role/CLI integration and a temporary validation-directory orchestration
helper. It must freeze the research roster, enforce preparation/coverage/isolation
and accounting gates, preserve actual partial outcomes and package public results.
Changing production code or authoritative contracts invalidates reuse of the old
source closure; perform any necessary fresh acceptance in the same serial operator
invocation before research. Never use acceptance-sized data as the new benchmark.


## Feasible pilot revision (T139–T142)

Keep the20-term exact cross-role window rule. Replace v1's parameter-count-only
roster selection with deterministic source-behavior screening and an explicitly
versioned wider coefficient law, shared by both arms. Validate every protected
cell and all12 fixed training shards with the same admission/quota code before
hardware work. Actual admitted target tokens, not a nominal floor, establish
one-pass capacity. Reference feasibility is not native qualification.

All sampling, admission, worker ordering/containment and feasibility policies are
in one reopened Phase B. C wires these into the temporary runner; D records actual
CPU measurements and publishes the handoff. Reference workers are bounded and
ordered; GPU learners/inference remain serial. Current-source native/HIP acceptance
and a reduced paired end-to-end smoke must pass before the full learners. Smoke
outcomes gate software health only, never model quality or protocol selection.
