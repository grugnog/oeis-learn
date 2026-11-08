# Implementation Plan: Progressive Mathematical Discovery

**Branch**: `013-progressive-mathematical-discovery` | **Date**: 2026-11-08 | **Spec**: [spec.md](spec.md)

**Input**: `specs/013-progressive-mathematical-discovery/spec.md`

**Status**: Phase 0 research and Phase 1 design complete. Runtime implementation, feature acceptance, source qualification and research execution remain pending. This document does not launch a campaign.

## Summary

Implement the complete five-stage pathway as a versioned extension of the installed generalization workflow: capability and language audit; genuinely refreshed program structures; bounded setting/size search; recoverable longer training and OEIS-derived self-learning; learned components and scoped discovery review. Add a bounded combinatorial-source pilot and explicit suitability records for other databases.

Keep the existing D allocation recipe as the starting policy and the repaired drain schedule for evaluation. Preserve all historical PAC contracts and evidence. Reuse the qualified model, native evaluator/fitter, independent reference, role projection, supervision, ledger and checkpoint machinery. New contracts describe what the old PAC experiment deliberately prohibited; do not remove its guards to make the new experiment fit.

The public interface will be `foundation_run.py ... discovery ...`, a thin canonical host dispatcher. Installed preparation, learner, generator, scorer and evidence roles own the work; there is no second learner, fitter, engine service or generic orchestration framework. The host remains standard-library-only. Optuna runs inside a bounded CPU role.

## Technical Context

**Language/Version**: Existing Python >=3.11 package and stdlib host wrapper; existing Rust native evaluator/fitter and pinned compiler/runtime images. No new host compiler or service.

**Primary Dependencies**: Existing pinned PyTorch/ROCm, native bigint service, independent Python reference, JSON-schema validation, container workflow. Add an exact locked Optuna release in the container dependency closure, with ask/tell compatibility tests. Keep initial precision fp32. Sage-assisted source production is a separate optional pinned acquisition/preparation image, never a learner dependency or downloaded-code execution path.

**Storage**: Immutable content-addressed JSON manifests, bounded append-only event journals, mechanical-data shards, target/component archives, full tensor checkpoints and bounded ZIP evidence under workspace `run-state`. No external database service; Optuna receives a reconstructed in-memory study from sealed history.

**Testing**: pytest unit/contract/integration; independent arithmetic and censored-metric fixtures; real CPU updates and crash recovery; installed-wheel/source-closure tests; mandatory native/container and actual-HIP gates; bounded real-device rehearsal before research. Long-task evidence is distinct from shortened clock fixtures.

**Target Platform**: NixOS/Linux, Ryzen AI Max+ 395, 128 GB shared RAM, existing Docker/Podman workflow; one serial GPU owner and bounded CPU services. Shared-memory accounting must not add overlapping allocator and cgroup counters.

**Project Type**: Research CLI with isolated installed roles and offline verifiable evidence.

**Performance Goals**: Progress or explicit stalled/unavailable state within 60 seconds; every training task <=86,400 active seconds including startup/online search/checkpoint/finalization. Positive scientific gates use prospective +2 pp gain, >=95% completion and <=2 pp retention loss. No throughput/accuracy speedup is promised. Preserve fixed 1,024-token, four-second fitting-service and 60-second portfolio budgets in the base evaluation profile.

**Constraints**: Exact source values and offsets; immutable historical contracts; no evaluation truth in learning; charge rejected draws, failed searches and discarded updates; finite nested resource graph; prospective metadata-driven freeze; current-source qualification; no automatic imported teacher or new host service.

**Scale/Scope**: All five stages plus source pilot. Initial research recipe uses 16 screening trials, at most three finalists and two confirmation seeds; 25M/100M/400M cumulative-token milestones; finite preallocated continuation slots, not an unlimited 24-hour loop. Cohort and aggregate budget defaults are specified in the contracts and may be revised only before scores through a new freeze. Tiny acceptance fixtures use separate DIAGNOSTIC_ONLY profiles.

## Constitution Check

Pre-research review and post-design review: PASS at design level. Adopted constitution 2.0.0 remains unchanged. No runtime acceptance, device readiness or mathematical discovery is asserted.

| Principle | Design and required execution evidence |
| --- | --- |
| I — Exact semantics | Exact decimal integer strings and offsets; existing bigint-body/v1 initially; versioned primitive semantics and independent edge-case oracle before extension |
| II — Independent acceptance | Final expanded source checked by native and independent reference; component substitution proof is not an all-domain sequence proof; witnesses isolated from learner |
| III — Bounded recovery | Reserve-before-work journals, per-task active clock, aggregate ledger, atomic checkpoint/data/library cut and bounded cleanup; no allowance reset on crash |
| IV — Workstation feasibility | Same portable container boundary, fixed precision, shape-specific actual HIP update/restore/calibration; no extrapolated inference-memory claim |
| V — Provenance | Synthetic strict track retains random-origin mechanical support. User-requested OEIS and combinatorial learning are distinct assisted profiles with explicit adoption in the frozen manifest; truth remains role-isolated. Existing artifacts are not relabeled |
| VI — Honest discovery | Finite agreement, conjecture, bounded verification, proof and novelty are separate fields. Proof promotion uses registered sound checkers or independently reviewed scoped proof; no generic LLM certificate authority |
| Workflow | Independent tests accompany acceptance/state changes; full five-stage acceptance exists even when research is negative; a frozen eligible transition needs no repeated permission |

The constitution's historical Docker wording is implemented through the already adopted Docker/Podman portable workflow and AGENTS.md; no additional host dependency is introduced. Optuna and optional mathematical source tools are container dependencies. An externally assisted run requires the concrete manifest to record its selected track, sources and initialization; the user has authorized designing that pathway, not an unbounded search budget.

## Project Structure

### Documentation (this feature)

- `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`.
- `contracts/cli.md`: proposed command interface and exit/status rules.
- `contracts/experiment.md`: cohorts, metrics, tuning, curriculum and scientific gates.
- `contracts/execution.md`: role channels, budgets, recovery and evidence.
- `contracts/learning-discovery.md`: targets, OEIS admission, components, sources and claims.
- `checklists/requirements.md`: existing specification quality review.
- `tasks.md`: produced by the subsequent tasks workflow, not by this planning pass.

### Source Code (repository root)

Existing integration points:

- `scripts/foundation_run.py`, `scripts/foundation_local.py`: thin registration/dispatch only.
- `src/oeis_learn/pipeline_validation/generalization_{cli,commands,runtime,roles,preparation,training,persistence,qualification_io}.py`: explicit version dispatch and existing worker adapters.
- `src/oeis_learn/generalization/{contracts,budgets,checkpoints,metrics,refresh}.py`: reuse stable primitives, preserve old version validators.
- `src/oeis_learn/generalization/pac_*` and `src/oeis_learn/pipeline_validation/generalization_pac_*`: reusable allocation/update logic behind an explicit new controller adapter; historical research hard guards stay intact.
- `src/oeis_learn/fresh/{bigint_worker,bigint_diagnostics,pipeline_scheduler}.py`, `src/oeis_learn/decoder/`, `src/oeis_learn/synthesis/`: actual model, sampler, fitter and independent final acceptance.
- `src/oeis_learn/experiments/source_snapshot.py`, `src/oeis_learn/pipeline_validation/{closures,diagnostics_collect}.py` and existing bundle/finalization modules: extend inventories, installed assets and bounded evidence.

Proposed additions, with final small-file splits allowed during implementation:

- `src/oeis_learn/generalization/progressive/`: closed contracts, exposure groups, deterministic structural streams, metrics/gates, trial proposals, budget state, target archive, components and claim/revocation logic. No engine or model ownership here.
- `src/oeis_learn/pipeline_validation/progressive_{cli,commands,roles,training,evaluation,preparation,evidence}.py`: host/installed adapters through existing roles and supervised invocation.
- `src/oeis_learn/generalization/schemas/progressive-*.schema.json` and `policies/progressive-*.json`: installed closed versioned assets; explicit package data/source qualification coverage.
- `configs/foundation/progressive/`: acceptance and research blueprints, never frozen manifests with guessed workstation paths or checkpoint hashes.
- `tests/unit/test_progressive_*.py`, `tests/contract/test_progressive_*.py`, `tests/integration/test_progressive_*.py`: independent algorithm, boundary, real installed-role and actual-HIP tests.
- Root `Dockerfile`, `pyproject.toml`, `uv.lock` and existing CI suite inventories: dependency pinning and installed acceptance coverage.

**Structure Decision**: Add one policy namespace and a small set of canonical adapters. Existing run_torch/forward/optimizer and native fitting remain the implementation of learning/execution. Refactor common internals only with legacy regression coverage; do not route new studies through PAC-screen/v1 by weakening its validation.

## Implementation phases and dependencies

### A — Closed contracts, exposure and execution authority

Define the records/state machines in [data-model.md](data-model.md), stage DAG, frozen policy registry, conservative clock/debit rules, feature/source inventory and offline verification. Implement role projections and negative cross-role reads before wiring learning. Establish command surfaces and deterministic acceptance fixtures with explicit diagnostic purpose.

Exit: malformed plans, missing caps, stale qualification, exposed assessment members, duplicate task leases, incompatible continuation and forged authority are rejected. Historical 011/012 records retain their behavior.

### B — Capability and mathematical representation

Implement four-stratum census/publication, final-source scoring, conditional family-aware uncertainty, baseline adapters and coverage witness audit. Start with bigint-body/v1. Add only witnessed necessary arithmetic extensions through a new versioned profile; qualify parser/codec/native/reference/fitter interactions as a unit. Fixed-budget drain remains the control.

Exit: actual checkpoints and baselines can be evaluated and independently reconciled; missing support and censoring cannot manufacture positive gates. No broad learning claim is made from a small fixture.

### C — Refreshed structures and bounded tuning

Create deterministic verified structural shards and fixed family/operator bins, retaining EXP3.S allocation within the new controller. Add replayable stream state and matched static/refreshed pilots. Add optimizer schedule/regularization/accumulation support, seeded architecture alternatives and shape checks. Then integrate serial Optuna ask/tell proposals, pruning, trial accounting and independent finalist confirmation.

Exit: a real small learning path exercises both support modes, interruption and tuning; static/refresh differences and all costs are visible; trial proposal replay survives crashes without changing the chosen configuration.

### D — 24-hour tasks, progressive tokens and OEIS adaptation

Extend installed learner state through a new version and integrate campaign/task clocks, milestone transitions and budget-preserving continuation. Implement yield audit, isolated training-continuation admission, concrete target archive and mixed supervised replay. Freeze a continued-synthetic control from the same initializer. Add report/export for every terminal outcome.

Exit: real CPU/HIP acceptance demonstrates complete-state continuation, an OEIS search/admit/update cycle and its negative gate. A separate sustained hardware run records actual long-task behavior; synthetic clocks alone do not satisfy this evidence.

### E — Reusable components, external data and discoveries

Implement typed training-source extraction and bounded macro expansion with independent semantics, library-guided search with expansion into the existing codec, checkpoint/library binding and matched no-library evaluation. Add FindStat-derived source admission and bounded object-conditioned pilot; other database groups get suitability records. Build discovery dossiers, proof/novelty separation and transitive refutation quarantine.

Exit: at least one actual learned component is used in generation in acceptance, not merely counted in a report. Source and component negative cases and a refuted claim traverse the same installed path as research.

### F — Consolidated qualification and staged research

Run CPU/native, installed-wheel, isolation, both-engine portability where required by the existing foundation, actual HIP and shortened end-to-end acceptance; collect one complete bounded package. Only then execute the frozen research DAG, serially, with milestone stops and reserved continuation slots. Reuse compatible evidence. Research stage failure prevents dependent research, not completion of remaining software acceptance work.

Tasks must include concrete work in A–F and acceptance for all five stages. No phase is replaced with a vague future-work placeholder. Commit coherent groups and push without waiting for CI jobs; fix failures identified during ordinary review.

## Validation and traceability

| Spec requirements | Design owner | Mandatory counterexamples / acceptance |
| --- | --- | --- |
| FR-001–006 | A, all contracts | pilot-only completion rejected; exposure/cross-track/assessment leakage; mutable plan |
| FR-007–011 | B, experiment | hidden witness access; uncertified or unsupported case; low-completion winner; incapable language vs failed search |
| FR-012–016 | C, learning-discovery | coefficient-only refresh; duplicate/rejection exhaustion; changed resumed draw; eval-derived target |
| FR-017–021 | C, experiment | pruning grace violation; lost failed trial; changed proposal after resume; two-seed CI mislabeled |
| FR-022–026 | A/D, execution | nested ceiling; reboot without final receipt; discarded work; partial-token task; plateau buying extra time |
| FR-027–030 | D, learning-discovery | prefix overfit; training feedback labeled holdout; insufficient yield; unmatched control budget |
| FR-031–035 | E, learning-discovery | capture/cyclic component; hidden expansion cost; invalid proof; transitive contamination |
| FR-036–037 | E, learning-discovery | rounded/non-integral values; arbitrary coefficient order; cross-source duplicates; short sequence |
| FR-038–041 | A/F, execution/CLI | CPU fallback; missing producer error; incomplete evidence marked verified; historical result mutation |

All acceptance changes require independently expected tests; mocks alone cannot qualify arithmetic, learning or hardware. A documentation check verifies references and contract coverage now; implementation tests are future tasks, not claimed by this plan.

## Complexity Tracking

No constitutional violation is accepted. The additional versioned policy is necessary because existing PAC fixes corpus, initialization, budgets and replay exclusions. Reusing its version would invalidate historical comparisons. A container-local tuning adapter and bounded component language are justified research additions; a new training service, general scheduler or imported-teacher system is not.
