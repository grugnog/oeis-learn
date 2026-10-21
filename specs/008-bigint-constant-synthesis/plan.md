# Implementation Plan: Exact Bigint and Parameterized Sequence Synthesis

**Branch**: `008-bigint-constant-synthesis` | **Date**: 2026-10-18 | **Spec**: [spec.md](spec.md)

**Input**: `specs/008-bigint-constant-synthesis/spec.md`; user instruction to keep all hard algorithm work in one phase for a stronger-model handoff.

## Summary

Add one versioned bigint synthesis path: a Rust bigint library compiled to a private WASM runtime, a typed logical program language lowered to approved arithmetic calls, and a native Rust exact constant fitter. Keep Python for neural learning and an independently implemented arithmetic reference, never as a fitting fallback. Preserve signed-256 as an explicitly supported historical/control profile. Widen the hybrid byte input without changing its weights for each supported integer width.

Integrate those capabilities with matched literal/parameterized training, progressive generic data, visible-only search and verified training replay. Compare a generic-only foundation/control with a separately labelled assisted branch; report finite coverage and structural transfer separately from novelty/proof claims. Remove the old Python solvers and their unsupported callers after migration.

**All difficult algorithm implementation is in implementation Phase 2.** It is a substantial phase with internal milestones and reviewable commits, not several phases disguised as integration work. See [algorithm-phase.md](algorithm-phase.md) for its complete handoff. Later semantic changes return to Phase 2. This document and its companion contracts are design artifacts, not implemented APIs or evidence of successful GPU qualification.

## Technical Context

**Language/Version**: Current Python 3.12 container / standard-library-only host wrapper; current Rust toolchain 1.90.0. Keep these unless the isolated dependency qualification records a justified pinned toolchain change.

**Primary Dependencies**: Existing locked Wasmtime 20.0.2 and PyO3 0.20; proposed `dashu-int` 0.6.2 with default features disabled for WASM, and Rust `z3` 0.21.1 with vendored native source. These exact candidate versions are researched, not build-qualified. Phase 1 must resolve and commit the complete binding/z3-sys/native-source lock tuple or explicitly revise the proposal before Phase 2. Do not combine an unrelated runtime upgrade with this feature.

**Storage**: Workspace-local immutable canonical JSON manifests, exact integer strings, sealed role inputs, append-only budget/evidence records, complete checkpoints and bounded analysis bundles. Compiled code caches include profile, runtime bytes, toolchain, engine and source hashes. Explicit aggregate per-role service defaults cap pending work at8 batches/32 MiB, one active batch, result cache4096 entries/64 MiB, compiled cache32 entries/64 MiB serialized charge and disk cache256 entries/512 MiB; FIFO waits consume parent deadlines and deterministic LRU governs eviction.

**Testing**: Rust unit/property/contract tests; independent Python exact-arithmetic differential tests; portable container integration, retirement/import audit, resume/leakage tests; explicitly marked actual ROCm forward/backward and cache-equivalence tests; bounded three-seed experiment qualification. The planning turn runs document validation only.

**Early native test access**: Phase 2 supplies a bounded qualification-only core test driver for execution, fitting and scoped replay before T025/T036. Python differential/admission fixtures use that driver plus the independent reference; they do not depend on Phase 3's production worker/client. The driver is not installed as a production command.

**Target Platform**: Linux Docker or Podman with a usable engine and supported ROCm GPU for neural work; CPU qualification has no GPU requirement. No account, broker, SSH, VM, sudo or particular distribution dependency is introduced.

**Project Type**: Containerized research CLI with native library/worker, model training and auditable candidate execution.

**Performance Goals**: Batch solving and production execution in native code; call Python reference only on shortlisted final assignments. Report phase-separated latency/throughput at64/256/512/1024/4096-bit legal supplied values and up to16,384-bit generated intermediates/results. Compare old/new arithmetic only on their shared supported small-value semantics; distinguish wider-value results, exclusions and any separate benchmark-only profile. Initial budgets are finite qualification defaults in the contracts, not throughput promises. Phase 5 binds measured cost to the exact experiment scope and may preregister a revised finite budget before execution; a diverse32 receipt cannot admit a complete transfer screen. More than 5x median small-value evaluation slowdown against the old backend triggers a review before long runs; this is an investigation threshold, not permission to alter semantics. Do not infer accelerator utilization from parameter allocation alone.

**Constraints**: Exact full multiplication; fresh runtime state per term; no hidden terms in fitting; no native arithmetic callback escaping WASM fuel; no solver status treated as verification; old profile identities unchanged; independent reference retained; source/trace/solver/memory/work caps enforced; no silent fallback; no arbitrary program completion.

**Scale/Scope**: 20 visible terms, normally 100 total evaluation indices; up to eight parameters; initial 4,096-bit visible/parameter magnitude capacity and 16,384-bit intermediate/result ceiling; 32 candidates per bounded native batch. Profiles may increase resource limits without changing the variable-width encoder's parameter shapes, but always acquire a new identity. This is bounded arbitrary-precision arithmetic, not unlimited computation.

**Research decisions**: Checkpoint selection/curriculum use visible fit; transfer promotion uses independently verified `continuation_accuracy_100` for the sealed selection from the exact-fitting equal-budget portfolio. Final OEIS evaluation stays sealed throughout representation, assistance and encoder studies. A distinct post-decision final plan fixes cohort, checkpoints, arm roster and budgets before explicit final scoring; exposure and allowance survive interruption.

## Constitution Check

The adopted constitution remains v2.0.0. User-approved assisted training is the explicit separate decision allowed by principle V; the strict branch remains generic-only. The user's portable Docker/Podman requirement governs over the historical workstation description. No constitutional amendment is proposed.

| Gate | Before research | After design / required implementation evidence |
| --- | --- | --- |
| I: exact values and declared semantics | Pass: separate numerical and input profiles | Pass in design: exact transport, full intermediate checks, no embedding-injectivity claim; arithmetic suite still required |
| II: independent acceptance | Pass: final-source validation retained | Pass in design: native assignments are provisional until independent reference verification; independent parser/oracle tests required |
| III: bounded execution/recovery | Pass: finite budgets mandated | Pass in design: runtime/store/process/solver/reference limits and ledger transitions specified; exhaustion/cleanup tests required |
| IV: reproducibility and workstation feasibility | Pass: portable containers and complete identity | Pass in design: pinned dependency feasibility gate, exact resume contract and actual GPU qualification; no hardware evidence claimed |
| V: provenance and isolation | Pass: assisted decision recorded | Pass in design: explicit track ancestry, isolated hidden values, parameter/structure/behavior splits and clean final cohort |
| VI: honest discovery/proof | Pass: finite matches separated | Pass in design: alternatives retained; scoped SMT replay is not theorem proof or novelty evidence |

**Gate interpretation**: Documentation passes design review; dependency build, runtime, GPU, and research promotion gates are pending implementation. If a dependency cannot meet the contract, revise the recorded design; never install a silent fallback or weaken a gate.

## Project Structure

### Documentation (this feature)

- `spec.md` — accepted scope and outcomes.
- `plan.md` — this implementation sequencing and constitution review.
- `research.md` — choices, alternatives, source evidence and qualification limits.
- `data-model.md` — identities, records, validation and state transitions.
- `contracts/runtime.md` — typed language, WASM boundary, numerical and resource contract.
- `contracts/solver.md` — native fitting, reference admission and evidence protocol.
- `contracts/workflow.md` — portable command/config/migration/export contract.
- `contracts/experiments.md` — curriculum, comparisons, component studies and promotion rules.
- `algorithm-phase.md` — self-contained stronger-model Phase 2 handoff and exit gates.
- `quickstart.md` — post-implementation validation/run guide; marks new commands/configs clearly.
- `checklists/requirements.md` — existing specification-quality checklist.
- `tasks.md` — dependency-ordered implementation tasks, generated by `/speckit.tasks`, preserving the six phases and single algorithm handoff.

`/speckit.plan` produced the design; the subsequent `/speckit.tasks` step generated [tasks.md](tasks.md). Task generation is not implementation or runtime qualification.

### Source Code (repository root)

| Location | Planned role |
| --- | --- |
| `crates/oeis_bigint_runtime/` (new) | Minimal separately locked wasm32 crate, private arena and approved arithmetic; never depends on native Z3/Python |
| `crates/oeis_synthesis_core/` (new) | Typed language, parameter abstraction, exact affine/SMT fitting, source substitution and profile-aware native execution orchestration |
| `crates/oeis_wasm_evaluator/` (existing) | Retained signed-256 backend plus explicit bigint adapter, thin binding/native worker entry point; no weakening old validator |
| `src/oeis_learn/synthesis/` (new) | Thin transport/admission facade, independent bigint source parser/reference, evidence validation; no Python fit algorithms |
| `src/oeis_learn/decoder/{program_codec,wat_grammar,sampler,wat_decoder,inference_cache}.py` | Profile dispatch, parameter-aware grammar and production cached/uncached generation using Phase 2 contracts |
| `src/oeis_learn/encoder/`, `experiments/bootstrap_models.py` | New length-extensible hybrid profile and explicit four-group ablation controls; old profiles retained by name |
| `src/oeis_learn/data/bootstrap_programs.py`, `fresh/{inputs,bootstrap_views,worker,search,replay,diagnostics}.py` | New generic profile, parameter family splits, recipe integration and evidence collection |
| `scripts/foundation_{run,local,host,tasks}.py`, `src/oeis_learn/container_validation.py` | Existing portable entry point/role routing; new profile/suite/config controls without new host requirements |
| `src/oeis_learn/fresh/{bundle,checkpoint_diagnostic}.py` | Compact mandatory summaries, phase metrics, independent artifacts and scoped diagnostics |
| `docker/dev/`, root `Dockerfile`, dependency locks, `.github/workflows/` | Cached source-pinned native/WASM builds, non-root image read permissions, parallel CPU/native test partitions |
| `configs/foundation/bigint_*.yaml` (new) | Versioned qualification, generic and assisted recipes referenced in quickstart |
| `tests/{contract,unit,integration}/` and crate tests | Existing counterexamples ported to supported interfaces plus semantic, isolation, migration and GPU gates |

**Dependency direction**: `oeis_wasm_evaluator` (binding/worker) depends on `oeis_synthesis_core`; the core never imports the binding crate. The core defines a native execution-backend trait and owns its bigint implementation; the binding/worker supplies the retained signed-256 backend adapter through that trait. This is a Rust-to-Rust call, not a Python callback. The guest runtime crate depends on neither host crate and is consumed as a hashed WASM artifact. Numerical profile/codec identities stay stable for retained controls; changed build/source/binary identities are still recorded honestly.

**Structure decision**: Extend the current CLI/container system. The WASM crate has a separate dependency closure; the native core is reusable without importing Python. A native worker isolates solver crashes/deadlines. The Python facade transports batches and independently verifies accepted candidates rather than reimplementing fitting.

## Implementation Phases

These are implementation phases **1–6**. Spec Kit's planning-workflow “Phase 0 research / Phase 1 design” are completed by the documents in this PR and are not additional implementation phases.

| Phase | Work and owner suitability | Depends on | Exit gate |
| --- | --- | --- | --- |
| **1 — Build and contract preparation** | Routine engineering: create crate/package skeletons, dependency source locks, vendored build layer, protocol/schema stubs that fail closed, fixture manifests, retirement inventory and test targets. No solver, compiler or encoder algorithm implementation. | This design | Pinned standalone WASM and native solver smoke builds work in both image targets; imports/features and arbitrary UID readability inspected; contract fixtures exist and incomplete features remain disabled |
| **2 — Algorithm core: stronger-model handoff** | **All hard algorithm work**, across Rust and Python: typed language/codec/grammar/cache semantics, bigint runtime and lowering, independent parser/reference, dependence/parameter analysis, affine and symbolic fitting, evidence semantics, variable-length hybrid math, generic structure/parameter generation, canonicalization/splits, curriculum/search/ranking/replay decision logic and differential/property tests. | 1 | Every algorithm acceptance gate in `algorithm-phase.md` passes; supported APIs and profile semantics frozen; no remaining semantic design deferred to later phases |
| **3 — Production integration and artifacts** | Wire qualified algorithms into preparation, learner, generation, scoring, checkpoints, role mounts, native batching, CLI/config, telemetry and prioritized exports. Integration logic may validate schemas and route calls but must not invent algorithms. | 2 | Bounded CPU end-to-end flow, budget-preserving resume, provisional-to-verified admission, hidden-mount tests and deterministic fixture reports pass |
| **4 — Migration and full retirement (complete)** | Migrate active CLI/evaluation/RL/qualification callers; remove retired Python solvers, duplicated grounding, obsolete scripts/configs/examples and unused dependencies; rewrite tests around supported entry points. Keep deliberate compatibility profiles and independent verification. | 3 | Completed by reviewed clean-image evidence at T088; zero old solver reachability/install surface, preserved historical scope and passing replacement regressions |
| **5 — Qualification and performance screen** | Execute CPU/native/container suites, actual HIP tests, resource/recovery cases, old-checkpoint raw/fitted diagnostic, measured batching/runtime benchmarks, scope-identified diverse32 and complete-screen cost preflights, and export checks. Calibrate and preregister measured resource/config values without redesigning algorithms. | 4 | Qualified immutable profile/image report; a scope-correct 32-example diverse gate and artifact completeness; separate later-screen admission; any semantic defect returns to Phase 2 |
| **6 — Controlled research and longer-run readiness** | Run pre-registered matched seeds; separate parameterization, assistance, replay and encoder studies; keep the frozen final cohort sealed until all choices/checkpoints are locked, then run an explicit immutable-plan final evaluation; prepare bounded longer-run configuration only after transfer gates pass. Negative results are a valid deliverable. | 5 | Reproducible comparison/ablation/final-study bundles and explicit promote/stop decision; full-scale run remains gated and is not launched automatically by this planning task |

Within Phase 2, use multiple commits and checkpoints without declaring new phases. `algorithm-phase.md` defines the handoff inputs, work packages and complete exit test matrix. If later integration reveals a new mathematical/compiler/learning decision, record it as a Phase 2 reopening, not “small wiring” for a weaker model. Build and documentation work can continue while that phase is handed off, but no dependent feature may pass its gate prematurely.

## Migration and Retirement Boundary

Delete old fitting implementation files after migration: `decoder/constant_solver.py`, `grounding.py`, `dixon_solver.py`, `qfnia_solver.py`; retire `mersenne61_filter.py` and recurrence-template fitting when the completed inventory confirms their solver-only role. Remove `fresh/search.py::grounded_variants` as a competing fitting path. No historical fit algorithm may survive under a compatibility name.

First separate `grounding_reference.py` from its imports of production `grounding.instructions`; replace required reference behavior independently in Phase 2. Move supported scoped evidence replay out of `grounding_evidence.py` into the native service; preserve read-only historical evidence status and independent arithmetic checks. Remove Python `z3-solver` after every active verifier/fitter caller is migrated. Keep SymPy only where independently supported discovery/certificate code still uses it.

Audit `cli/main.py`, `evaluation/{experiments,synthesis,repair_gate}.py`, `experiments/{config,qualified_models}.py`, `rl/{trainer,elite_buffer}.py`, current fresh paths and historical run008/009/010 scripts. Migrate supported functions; retire obsolete launch scripts/configs together. Preserve tests as counterexample fixtures against the new native API rather than preserving obsolete imports. Keep original historical specs/results and the old-profile evaluator/reader; git history provides obsolete executable reproductions.

## Validation and Traceability

| Spec coverage | Algorithm phase responsibility | Integration/qualification |
| --- | --- | --- |
| FR-001–006, FR-013–018; SC-001–003 | Numerical contract, independent oracle, parameter semantics, affine/SMT and scoped outcomes | Contract tests, pinned WASM inspection, process limits, old-profile equivalence and source-index tests |
| FR-007, FR-019–023, FR-026; SC-004/007–009 | Generator/splits, feature architecture, conditioning controls, ranking and curriculum/replay rules | Matched recipe plumbing, complete metrics, cached/uncached CPU/HIP tests and bounded comparisons |
| FR-008–009, FR-024–025 | Provenance transition, alternative-retention and selection rules | Track/leakage mounts and ancestry audit; frozen final cohort and discovery-status reports |
| FR-010–012, FR-027–031; SC-005–006 | Budget/admission state semantics and historical compatibility decisions | Portable wrappers, complete checkpoints, retirement audit, negative/resume/export tests |

Run mathematical/property tests in the native CPU job, container contract tests in their own job and ordinary Python regressions in the existing parallel jobs. Reuse the current shared CI image/cache workflow; do not rebuild the same ROCm image per test partition. GPU tests remain explicit user-run gates unless a qualified GPU CI runner is available. Preserve mandatory source/wheel content checks needed to ensure the runtime artifact is actually packaged; do not add unrelated distribution testing.

## Complexity Tracking

No unjustified constitutional violations. Separate WASM and native dependency closures are required to keep solver/host code out of metered candidate arithmetic. A distinct reference interpreter is deliberate verification independence. Three numerical/model identities (historical signed-256, bigint runtime, variable-width hybrid) are explicit experimental controls rather than silent replacements.
