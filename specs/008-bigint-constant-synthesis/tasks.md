# Tasks: Exact Bigint and Parameterized Sequence Synthesis

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [algorithm-phase.md](algorithm-phase.md), [contracts/](contracts/), [quickstart.md](quickstart.md).

**Prerequisites**: Read those documents, repository `AGENTS.md` and `.specify/memory/constitution.md` before implementation. Tests are required by the specification and constitution. This document creates work; unchecked tasks are not implementation or qualification evidence.

**Organization**: Preserve the plan's six implementation phases. At the user's explicit request, **all hard algorithm work is in Phase 2**, with user-story work packages inside phases rather than a separate implementation phase per story. This intentionally overrides the template's usual one-phase-per-story layout. Multiple commits/sessions within Phase 2 are expected. Later semantic defects return to Phase 2 and its stronger-model owner; they are not delegated as integration fixes.

## Format and execution rules

- Each task has an ID, optional `[P]`, a story label for story work, and repository-relative file paths. New files are proposed implementation locations, not claims that they exist today.
- `[P]` means independent of its stated parallel partner **after their common prerequisites pass**; it does not authorize executing every marked task together. The parallel table below defines these groups. Unmarked tasks execute in listed order within their phase unless a dependency explicitly permits otherwise.
- Write test tasks before their corresponding implementation, observe the intended failure, then make them pass. Reuse qualifying existing implementations/tests when possible; record evidence instead of rewriting them unnecessarily.
- Commit and push after each coherent task group to the draft PR. Do not wait on slow CI after pushing. Report unavailable GPU/engine gates with commands; never mark them passed from CPU-only evidence.
- Feature flags remain disabled until their gates pass. No Python solver fallback, hidden-data fitting, silent CPU fallback or unrecorded budget refund is permitted.
- Documents under `specs/008-bigint-constant-synthesis/validation/` below are compact evidence summaries with artifact hashes/locations. Bulky logs, datasets and bundles belong under workspace `tmp-validation/`, `tmp-analysis/`, `runs/` or configured equivalents, not in git.

## Phase 1: Build and contract preparation

**Goal**: Establish build feasibility, exact contract stubs, compatibility fixtures and the retirement inventory without implementing compiler, solver or encoder algorithms.

- [ ] T001 Inventory supported and retired solver consumers, commands, configurations, tests and examples in `specs/008-bigint-constant-synthesis/retirement.md`; record old path/interface, consumers, disposition, replacement, rationale and regression fixture, satisfying “Every old solver caller accounted for; no executable Python fitting fallback remains installed”.
- [ ] T002 Create separately locked guest and native skeletons in `crates/oeis_bigint_runtime/Cargo.toml`, `crates/oeis_synthesis_core/Cargo.toml` and `crates/oeis_wasm_evaluator/Cargo.toml`; enforce binding → core, independent guest dependency closure, and a core execution-backend trait with no dependency back on the binding.
- [ ] T003 Qualify the proposed dashu-int 0.6.2 standalone WASM build and Rust z3 0.21.1/vendored-native tuple against the pinned toolchain; commit exact guest/native `Cargo.lock` files and build flags, and record imports/features, exact z3-sys/native source and any justified pin revision in `specs/008-bigint-constant-synthesis/validation/dependencies.md`; no unrelated Wasmtime upgrade or runtime download.
- [ ] T004 Package the pinned smoke artifacts in both image targets through `Dockerfile`, `docker/dev/install.sh` and `docker/dev/build-native.sh`; extend deterministic native source-closure identity beyond the current hard-coded inventory and verify arbitrary non-root UID readability and absence of native solver/Python imports from the guest artifact.
- [ ] T005 Create fail-closed schema/API stubs in `src/oeis_learn/synthesis/contracts.py` and `crates/oeis_synthesis_core/src/protocol.rs` for the versioned records in `data-model.md`; unknown fields/profiles fail, unfinished operations return unsupported, mathematical integers are strings, and no stub grants admission.
- [ ] T006 Freeze independently derived arithmetic, exact-index, old signed-256 and historical false-fit fixtures in `tests/fixtures/bigint/manifest.json` with provenance and hashes; include factorial 0..99, large cancellation, two-large-operand multiplication and the 16-versus-4 rejection case without deriving expectations from production lowering.
- [ ] T007 Add dependency/artifact-content smoke tests in `tests/contract/test_bigint_build.py` and a build smoke executable in `crates/oeis_synthesis_core/examples/dependency_smoke.rs`; test an exact integer beyond i64 through the native binding and standalone WASM instantiation without treating this as arithmetic qualification.
- [ ] T008 Extend `.github/workflows/foundation.yml`, `.github/workflows/ci-image.yml` and `.github/workflows/container-images.yml` with native/contract test partitions using the existing shared cached image and source locks; avoid repeated ROCm builds, preserve required artifact packaging checks and keep actual GPU tests separately marked.
- [ ] T009 Capture retained signed-256 and historical model regression results in `specs/008-bigint-constant-synthesis/validation/compatibility-baseline.md`, identifying source, locks and fixture hashes; distinguish stable numerical/codec identities from changed binary/source identities.
- [ ] T010 Record the Phase 1 gate and stronger-model handoff in `specs/008-bigint-constant-synthesis/validation/phase-1.md`; require working pinned builds, inspected imports, readable packaged artifacts, fail-closed stubs, retirement inventory and compatibility fixtures before opening Phase 2.

**Checkpoint**: Build feasibility is demonstrated, not assumed from package documentation. No training or new-profile production use is enabled.

## Phase 2: Algorithm core — single stronger-model phase

**Entry**: T010 passes. Read [algorithm-phase.md](algorithm-phase.md) in full. The work packages below are subdivisions of **one phase**, not additional implementation phases. Tests exercise every contract limit, including failures and independent expectations.

### US1 — Exact language, arithmetic and large inputs (P1)

**Independent test**: A model-free frozen corpus returns exact values/classified failures; a separate encoder suite proves finite width-extensible behavior. Historical signed-256 fixtures remain unchanged.

- [ ] T011 [P] [US1] Write canonical-value, source-index, typed-language and codec contract tests in `tests/contract/test_bigint_language.py`; cover canonical decimal/sign-magnitude boundaries, malformed literals, negative/offset indices, distinct control/handle types, parameters, unsupported operations and old-vocabulary mismatch (parallel with T012–T013 after T010).
- [ ] T012 [P] [US1] Write differential arithmetic and containment tests in `tests/integration/test_bigint_runtime.py` using `tests/fixtures/bigint/manifest.json`; cover every runtime failure class, boundary products, independent disagreement injection, fresh state, forbidden imports/memory/privileged exports, all per-term/aggregate caps and healthy recovery (parallel with T011/T013 after T010).
- [ ] T013 [P] [US1] Write independent parser/reference tests in `tests/unit/test_bigint_reference.py`; require separately parsed final source, exact arithmetic and indices, intermediate-limit rejection despite cancellation, step/deadline accounting and no admission after reference exhaustion (parallel with T011–T012 after T010).
- [ ] T014 [US1] Implement canonical integer/profile types in `crates/oeis_synthesis_core/src/value.rs` and `src/oeis_learn/synthesis/contracts.py`; enforce `0|-?[1-9][0-9]*`, bounded pre-parse lengths and canonical sign/minimal-LE bytes; freeze ExecutionProfile “No unknown operations; old-profile records immutable; resource changes alter identity” and ResourceProfile “Positive finite bounds with sensible nesting; aggregate budgets independent of per-term limits”.
- [ ] T015 [US1] Implement typed parsing/validation in `crates/oeis_synthesis_core/src/language.rs` for actual signed `$index`, bigint/control/bool locals, supported operators and bounded structured control; enforce 64 KiB source, 4,096 AST nodes, nesting32, 16 bigint/8 control locals and LogicalProgram “Complete source/EOS, types/locals valid, no unresolved parameter reaches execution, no fabricated shared IDs”.
- [ ] T016 [US1] Implement the immutable capped bigint arena and approved operations in `crates/oeis_bigint_runtime/src/lib.rs`; enforce checked handles, full multiplication, sticky failure status, host-only literal/input loading and bounded serialization, with exact borderline-result checks, no handle reuse/reset/free and no native arithmetic callbacks escaping fuel.
- [ ] T017 [US1] Implement typed lowering and the private two-instance Wasmtime adapter in `crates/oeis_synthesis_core/src/lowering.rs` and `crates/oeis_synthesis_core/src/runtime.rs`; register only approved ABI functions individually, use one capped memory/fresh store per term, charge initialization/arithmetic/serialization to one fuel allowance and enforce all `contracts/runtime.md` limits without implicit index rebasing.
- [ ] T018 [US1] Implement independent source parsing and Python-int interpretation in `src/oeis_learn/synthesis/reference_parser.py` and `src/oeis_learn/synthesis/reference.py`; do not import the production parser/AST/lowering, preserve exact negative/offset indices and independently enforce step, bit, control and deadline limits.
- [ ] T019 [US1] Add the named checked signed-256 execution adapter in `crates/oeis_wasm_evaluator/src/synthesis_adapter.rs` using the core trait; preserve existing validator and exceptional semantics, explicitly reject unsupported wrapping/scalar/bitwise/streaming translations and verify T009 fixtures without silently promoting old programs to bigint.
- [ ] T020 [US1] Implement `bigint-codec/v1` and typed constrained generation in `src/oeis_learn/decoder/bigint_codec.py`, `src/oeis_learn/decoder/bigint_grammar.py` and profile dispatch in `program_codec.py`/`wat_grammar.py`; preserve BOS/PAD/EOS, complete literals, reusable p0..p7 and 1,024 generated-token limits, with no completion repair or silent old-embedding resize.
- [ ] T021 [US1] Write large-input feature/chunk and device/cache tests in `tests/unit/test_bigint_encoder.py` and `tests/integration/test_bigint_gpu.py`; cover signed neighbors at 256/512/1024/4096 bits, 4097-bit differences, chunk boundaries, padding invariance, width-independent parameter shapes, old-width feature tolerances and real HIP forward/backward updates.
- [ ] T022 [US1] Implement finite large-integer feature math in `src/oeis_learn/encoder/large_integer_features.py` and adapt `magnitude_stream.py`, `difference_stream.py` and `modulo_stream.py` by explicit profile; use bit lengths/leading bits for logs, exact residues/differences, frozen odd primes, valuation cap32 and defined zero handling without changing historical defaults.
- [ ] T023 [US1] Implement `hybrid-chunks/v1` in `src/oeis_learn/encoder/integer_chunks.py` and `hybrid.py`; follow the shared eight-byte/d64/four-head/ff256 chunk encoder contract, distinct pooled value/difference signs and lengths, masked mean pooling, growing device-correct sinusoidal buffers and d256 learned byte-gate fusion.
- [ ] T024 [US1] Qualify parameter-aware teacher forcing, masks, termination and cached/uncached decoding in `src/oeis_learn/decoder/sampler.py`, `wat_decoder.py` and `inference_cache.py`; add regression cases in `tests/integration/test_bigint_decoding.py` for new and retained profiles, prompt controls, device movement and EOS/token alignment without weakening determinism.
- [ ] T025 [US1] Execute the US1 CPU/native differential suites and record the runtime/encoder checkpoint in `specs/008-bigint-constant-synthesis/validation/algorithm-runtime.md`; explicitly list any pending actual-HIP checks, ensuring all semantic discrepancies are resolved before dependent integration is declared qualified.

### US2 — Semantic parameters and exact native fitting (P1)

**Independent test**: Fit fixed structures without training; independently reject false solutions and report ambiguity, unsupported semantics, scoped UNSAT and resource limits honestly.

- [ ] T026 [P] [US2] Write affine, rank-deficient integer-feasibility, nonlinear and control-taint tests in `crates/oeis_synthesis_core/tests/fitting.rs`; include large/shared/unused coefficients, nonintegral unique solutions, inconsistent systems, checked intermediate overflow and historical false fits (parallel with T027 after T025).
- [ ] T027 [P] [US2] Write public admission/evidence tests in `tests/contract/test_native_fitting.py`; inject provisional SAT, wrong 16-for-4 output, missing indices, reference timeout, fabricated constraints, smaller-bound UNSAT, unfinished uniqueness and unknown statuses; require zero false admissions (parallel with T026 after T025).
- [ ] T028 [US2] Implement dependency/control-taint analysis and deterministic eligible-literal abstraction in `crates/oeis_synthesis_core/src/dependence.rs` and `parameters.rs`; protect helpers/storage/control literals, cap eight first-occurrence parameters, preserve original/occurrence maps, never merge equal literals implicitly and reject parameter-tainted control semantically.
- [ ] T029 [US2] Implement bounded symbolic execution and exact affine forms in `crates/oeis_synthesis_core/src/symbolic.rs`; unroll only parameter-independent control, cap 1,024 iterations/loop and 100,000 request nodes, prove affine multiplication by dependence and retain all intermediate numeric guards.
- [ ] T030 [US2] Implement deterministic pivoted fraction-free elimination, exact rational back-substitution and original-equation/contradiction checks in `crates/oeis_synthesis_core/src/affine.rs`; route rank deficiency to integer feasibility instead of choosing rational free variables heuristically.
- [ ] T031 [US2] Implement bounded QF_LIA/QF_NIA translation in `crates/oeis_synthesis_core/src/smt.rs`; preserve exact arbitrary-size numerals and checked-profile intermediate constraints, keep contexts worker-local, reject unsupported semantics and never read UNKNOWN models as satisfying solutions.
- [ ] T032 [US2] Implement fitting orchestration, bound escalation and assignment enumeration in `crates/oeis_synthesis_core/src/fit.rs`; enforce eight parameters/assignments, 4096-bit coefficients, 16/64/256/1024/4096-bit search scopes under one budget, two-second native allowance and reserved reference time, deterministic discovered-set ranking and four explicit uniqueness states.
- [ ] T033 [US2] Implement final-source substitution and production visible checks in `crates/oeis_synthesis_core/src/substitute.rs`; check every actual visible index, enforce substituted-source/AST/literal limits, retain at most eight provisional assignments and never return public verification or UNSAT from a failed assignment.
- [ ] T034 [US2] Implement independent final admission in `src/oeis_learn/synthesis/admission.py`; enforce VisiblePrompt “Strict ordering and explicit missing-index rules; no hidden continuation field in fitter schema”, FitRequest/Result “Provisional native results cannot be admitted; bounds/coverage/phase required for failures” and VerificationRecord “Every visible term checked for admission; full horizon checked for continuation scoring; disagreement is a defect”.
- [ ] T035 [US2] Implement scoped evidence derivation/replay and identity-aware cache policy in `crates/oeis_synthesis_core/src/evidence.rs` and `src/oeis_learn/synthesis/evidence.py`; re-derive constraints from recorded source/profile/prompt, classify incomplete bounds/ambiguity, invalidate cross-profile/reference cache entries and keep timeout/unknown distinct from permanent UNSAT.
- [ ] T036 [US2] Run affine/SMT/admission counterexamples and differential properties, recording results in `specs/008-bigint-constant-synthesis/validation/algorithm-fitting.md`; require exact final-source verification, no Python trial callbacks and scoped replay with no proof-of-sequence claim.

### US3 — Contract, admission and budget state semantics (P1)

**Independent test**: State-transition fixtures reject stale/provisional responses, contract changes and budget refunds without requiring a container or trained model.

- [ ] T037 [US3] Write contract/ledger transition tests in `tests/unit/test_synthesis_state.py`; cover late worker generations, partially completed batches, cancelled/interrupted work, finalized runs, missing completion records, exact-contract resume and immutable child analysis identities.
- [ ] T038 [US3] Implement pure state/identity validation in `src/oeis_learn/synthesis/state.py`; enforce ExperimentContract “Freeze before execution; cross-profile/recipe changes create new experiments; unknown settings rejected” and Checkpoint “Resume is exact-contract only; weights-only artifact is analysis, not resumable state”, conservative durable debit/reconciliation and candidate provisional-to-admitted transitions.

### US4 — Encoder attribution, generic learning and search policies (P2)

**Independent test**: Deterministic datasets/policy fixtures prove meaningful splits, full denominators and hidden-free decisions; encoder ablations have explicit, tested fusion boundaries.

- [ ] T039 [US4] Write four-component attribution tests in `tests/unit/test_hybrid_ablation.py`; verify magnitude, modular, difference/valuation and exact-byte boundaries independently, no implicit renormalization, matched retraining masks, active/total parameter reporting and checkpoint mask identity.
- [ ] T040 [US4] Implement explicit ablation/retraining boundaries in `src/oeis_learn/encoder/tri_stream_encoder.py`, `film_fusion.py`, `hybrid.py` and `src/oeis_learn/experiments/bootstrap_models.py`; preserve frozen primes/FiLM/difference settings and distinguish sensitivity of trained weights from usefulness after retraining.
- [ ] T041 [US4] Write generic-family, paired-admission and overlap tests in `tests/unit/test_bigint_bootstrap_data.py`; include dead-code/trap counterexamples, shared parameter provenance, equal-output/different-source groups, detectable shifts and exact tiny/transfer composition/shortfall checks.
- [ ] T042 [US4] Implement versioned connected A–D generic sampling in `src/oeis_learn/data/bigint_programs.py` and profile dispatch in `bootstrap_programs.py`; generate meaningful carried-state loops/compositions/comparisons with at least16 variants/family, enforce ExampleFamily “Stage/key reproducible; equal values do not imply shared parameters; simplification cannot drop traps”, and use no simplifier unless independently qualified.
- [ ] T043 [US4] Implement canonical families, behavior/prefix/shift grouping and matched view selection in `src/oeis_learn/fresh/bigint_splits.py` and `bootstrap_views.py`; enforce SplitManifest “No detectable behavior group crosses incompatible splits; family overlap allowed only in the declared parameter-dev experiment”, paired token-limit exclusions, diverse32 and 512/128/128 counts, 10% transfer constant/offset caps and no silent underfill.
- [ ] T044 [P] [US4] Write prompt-control, template, search-budget and denominator tests in `tests/unit/test_bigint_search_policy.py`; cover fixed derangements, valid absent markers, original-versus-fitted candidate identity, partial coverage, no hidden reranking and total portfolio cost (parallel with T045 after T043).
- [ ] T045 [P] [US4] Write curriculum/replay/comparison tests in `tests/unit/test_bigint_learning_policy.py`; cover two-validation advancement, retained practice, dev/final replay rejection, fixed-panel selection, resumed replacement state, paired seeds and family bootstrap statistics (parallel with T044 after T043).
- [ ] T046 [US4] Implement frozen prompt derangements, valid unconditioned representation and visible-only non-neural template bank in `src/oeis_learn/fresh/conditioning_controls.py` and `template_baseline.py`; cap eight independently generated templates, disclose priors and prohibit metadata/hidden-output-driven templates.
- [ ] T047 [US4] Implement raw/same-candidate-rescue/equal-budget policy functions in `src/oeis_learn/fresh/synthesis_policy.py`; freeze one greedy plus seven stochastic attempts, 1024-token caps, 60-second total visible budget, reserved independent checks and visible-only ranking, with explicit beam budget reallocation and bounded incomplete-token evidence.
- [ ] T048 [US4] Implement full-denominator selection, stopping and advancement in `src/oeis_learn/fresh/learning_policy.py`; select synthetic visible competence before complete teacher-forced loss/update ties, require 80% stage fit for two validations with95% coverage, retain at least25% earlier examples and never use hidden continuation to select checkpoints.
- [ ] T049 [US4] Implement verified training-only replay admission/capacity/deduplication/replacement policies in `src/oeis_learn/fresh/replay.py`; retain exact sources/assignments/profile/track, cap later replay at25% of a batch, disable it in representation comparisons and expose complete resumable policy state.
- [ ] T050 [US4] Implement paired-seed/family-bootstrap metrics and gates in `src/oeis_learn/fresh/comparison.py`; use fixed denominators, 2000 seeded family resamples, separate seed spread, 31/32 diverse gate and exact 8/128-versus-each-control, mean6.25pp,95% coverage and ≤4/128 parameter-regression transfer rules from `contracts/experiments.md`.
- [ ] T051 [US4] Implement complete-screen cost accounting in `src/oeis_learn/fresh/cost_policy.py`; project all scheduled panels plus full end-of-screen controls using measured p95 phase costs and block recipes exceeding80% of total budget rather than dropping prompts or changing cadence silently.
- [ ] T052 [US4] Execute data, ablation and learning/search policy suites and record algorithm evidence in `specs/008-bigint-constant-synthesis/validation/algorithm-learning.md`; identify all model/data/decoder policies by hash and distinguish the joint gain/dropout recipe from an isolated causal finding.

### US5 — Assisted provenance and mathematical alternatives (P2)

**Independent test**: Fixed ancestry/archive fixtures prevent strict-track contamination and preserve alternative algorithms with appropriately scoped evidence.

- [ ] T053 [P] [US5] Write provenance/track-transition tests in `tests/unit/test_synthesis_provenance.py`; inject assisted checkpoints, named demonstrations, exposed final groups and legacy replay into strict lineage and require rejection (parallel with T054 after T052).
- [ ] T054 [P] [US5] Write alternative-retention/evidence-status tests in `tests/unit/test_synthesis_archive.py`; exercise same coefficients/different structures, cost/size trade-offs, deterministic cap/omissions and finite-match versus proof/novelty labels (parallel with T053 after T052).
- [ ] T055 [US5] Implement monotone assisted ancestry and overlap admission policy in `src/oeis_learn/fresh/provenance.py`; enforce AssistedManifest “Explicitly reviewed before use; no strict recipient; known exposed final data cannot become a fresh held-out test”, shared generic foundation lineage/cost and explicit historical-checkpoint migration approval.
- [ ] T056 [US5] Implement structural/cost alternative retention in `src/oeis_learn/fresh/candidate_archive.py`; cap eight verified alternatives per target, preserve non-dominated structure/cost representatives with deterministic ties and omissions, enforce CandidateArchiveRecord “Retain alternatives within frozen policy; omission decisions logged; finite match never implies novelty/proof”, and separate training representative choice from archive deletion.
- [ ] T057 [US5] Implement independent discovery-status validation in `src/oeis_learn/synthesis/discovery_status.py`; keep finite match, conjecture, bounded verification, proved statement and novelty review separate with evidence references, and preserve historical unreplayed status without executing retired solvers.
- [ ] T058 Record the complete Phase 2 exit review in `specs/008-bigint-constant-synthesis/validation/phase-2.md` against every row of `algorithm-phase.md`; freeze qualified APIs/profiles and all counterexample results, identify unresolved actual-GPU gates, and block later qualification if any semantic decision remains deferred.

**Checkpoint**: All mathematical/compiler/model/policy algorithms are implemented and CPU/native qualified. Actual HIP evidence may remain explicitly pending until Phase 5; a HIP-discovered algorithm defect reopens this phase. Later phases may persist and route these APIs, not invent new semantics.

## Phase 3: Production integration and artifacts

**Entry**: T058 passes the CPU/native semantic gate. Goals are API wiring, portable orchestration and useful artifacts. Tests use qualified Phase 2 policies rather than creating alternative implementations.

### US3 — Portable service, commands and durable state (P1)

- [ ] T059 [P] [US3] Write native transport/process tests in `tests/integration/test_synthesis_worker.py`; cover 4 MiB framing before allocation, byte/count batches≤32, malformed/duplicate IDs, partial responses, stderr bounds, kill/reap, stale generations and healthy restart (parallel with T060 after T058).
- [ ] T060 [P] [US3] Write CLI/config/resume/export contracts in `tests/contract/test_bigint_workflow.py`; cover conflicting flags, unknown settings, exact-image/contract resume, immutable analysis children, workspace persistence and mandatory core records under bundle caps (parallel with T059 after T058).
- [ ] T061 [US3] Add the framed worker entry point and thin binding in `crates/oeis_wasm_evaluator/src/bin/synthesis_worker.rs` and `src/lib.rs`, routing `fit`, `execute` and `replay_unsat` to the qualified core; keep stdout protocol-only, no hidden fields or trial callbacks and the old 1 MiB worker protocol unchanged.
- [ ] T062 [US3] Implement bounded transport/supervision in `src/oeis_learn/synthesis/client.py` and `worker_pool.py`; batch by bytes/count, preserve partial status, enforce worker RSS/deadline/cleanup, invalidate stale generations and call independent admission only on bounded provisional shortlists.
- [ ] T063 [US3] Wire new recipe parsing/resolved hashes in `src/oeis_learn/fresh/bigint_config.py`, `bootstrap_contract.py` and `scripts/foundation_tasks.py`; implement all `bigint-bootstrap/v1` sections with schema rejection, immutable explicit defaults and container-side YAML resolution while keeping host Python standard-library-only.
- [ ] T064 [US3] Expose planned suites, config/seed/fitting/ablation options and `compare` via `scripts/foundation_local.py` and the existing `scripts/foundation_run.py` entry point; preserve retained-profile commands and reject contradictory explicit settings and retired options with concise migration guidance.
- [ ] T065 [US3] Wire qualified ledger/contract transitions into `scripts/foundation_host.py` and checkpoint persistence in `src/oeis_learn/fresh/worker.py`; save weights, optimizer, RNG, data order, curriculum, replay and selection together, debit before work, prohibit refunds/image substitutions and preserve final/interrupted summaries on every exit.
- [ ] T066 [US3] Integrate role-specific inputs and native worker limits in `scripts/foundation_host.py`; isolate scoring continuations, keep network only for bulk-fetch roles, persist all durable data/cache under workspace and forbid runtime sockets, sudo, SSH, brokers or host-account assumptions.

### US1/US2 — End-to-end exact preparation and admission (P1)

- [ ] T067 [US1] Wire exact values/indices, paired generic archives and frozen split checking into `src/oeis_learn/fresh/inputs.py` and `scripts/foundation_tasks.py`; retain existing pinned bulk OEIS downloads, emit size/reason exclusions and enforce full requested split counts with private scoring payloads.
- [ ] T068 [US1] Wire `hybrid-chunks/v1` and new codec/profile identities into model/checkpoint construction in `src/oeis_learn/experiments/bootstrap_models.py` and `src/oeis_learn/fresh/worker.py`; explicitly reject vocabulary/input-profile mismatch and record device, total/active parameters and full feature configuration.
- [ ] T069 [US2] Replace fresh fitting routing in `src/oeis_learn/fresh/search.py` and `worker.py` with the single native client and independent admission; preserve raw/transformed/final candidates, assignments, scopes and phase timings, and seal visible-only choices before hidden scoring.
- [ ] T070 [US2] Integrate old-checkpoint raw/exact/paired analysis in `src/oeis_learn/fresh/checkpoint_diagnostic.py`; retain original numerical semantics, report unsupported candidates, create a separate child evaluation identity and never change parent weights, training contract or ledger.

### US4 — Training, controls and comparisons (P2)

- [ ] T071 [US4] Wire matched target views, curriculum/replay state and qualified selection/cost policies into `src/oeis_learn/fresh/worker.py` and `scripts/foundation_host.py`; validate all development teacher-forced examples, explicitly freeze32/32/32 periodic panels and require full128+128 end-of-screen coverage without hidden checkpoint selection.
- [ ] T072 [US4] Wire prompt controls, fixed-template baseline, cached/uncached samples and phase metrics into `src/oeis_learn/fresh/diagnostics.py`; preserve termination/legality/visible/continuation/diversity/rescue/ambiguity and fixed-denominator failures, stratified by size/length/complexity, with unavailable sensors represented as null plus reason.
- [ ] T073 [US4] Wire `compare` and four-input `diagnose` reports in `scripts/foundation_tasks.py` and `src/oeis_learn/fresh/comparison.py`; reject mismatched arms, list missing seeds, report paired counts/costs/intervals and distinguish post-training ablation from separately retrained variants without new statistics.
- [ ] T074 [US4] Commit `configs/foundation/bigint_smoke.yaml`, `bigint_literal_screen.yaml`, `bigint_parameter_screen.yaml` and `bigint_encoder_study.yaml` using the frozen experiment contract; use seeds20261016–18, d256/4+4, joint decoder recipe, LR1e-4, batch8,3000 updates/7200s,100/500-update cadence and explicit patience/preflight/promotion rules.

### US5 — Lineage and archive integration (P2)

- [ ] T075 [US5] Wire reviewed manifests, monotone lineage and alternative archive records into `src/oeis_learn/fresh/inputs.py`, `replay.py` and `worker.py`; add `configs/foundation/bigint_assisted_screen.yaml` that fails until a compatible generic foundation and reviewed demonstration manifest are supplied.
- [ ] T076 [US5] Wire discovery/ancestry summaries into `src/oeis_learn/fresh/diagnostics.py` and `bundle.py`; report shared foundation and incremental branch compute, alternative retention/omissions and separate finite-match/proof/novelty evidence without automatic promotion.

### US3 — Exports, validation suites and integration gate (P1)

- [ ] T077 [US3] Implement bounded core summaries/priority packing in `src/oeis_learn/fresh/bundle.py`; enforce Report/BundleManifest “Core records reserved first; no final/private payload leaks into training-visible bundle”, minimum16 MiB/core reserve8 MiB/default64 MiB, bounded detail chunks and hash/size/reason omissions, rejecting a cap that cannot contain required records.
- [ ] T078 [US3] Register `bigint-cpu`, `bigint-gpu` and `bigint-retirement` in `src/oeis_learn/container_validation.py`; emit concise per-stage cause/result/stdout/stderr paths and summaries, avoid read-only pytest-cache errors, and distinguish unsupported hardware from passing actual-GPU tests.
- [ ] T079 [US3] Add integrated prepare/check/train-fixture/generate/fit/score/resume/pack tests in `tests/integration/test_bigint_workflow.py`; inject hidden mounts, contract changes, partial results, disappearing temporary files and export pressure, asserting complete state and no hidden influence or budget refund.
- [ ] T080 [US3] Record the bounded CPU integration gate in `specs/008-bigint-constant-synthesis/validation/phase-3.md`; require native-only routing, independent admission, isolation/resume/exports and deterministic fixture reports; route discovered semantic defects back to T058 rather than patching algorithms here.

## Phase 4: Migration and full retirement

**Goal / US3 independent test**: Every supported caller uses the same native fitter; a clean installed image contains no executable Python solver, while independent checks and historical readers remain usable. Depends on T080.

- [ ] T081 [US3] Add installed-distribution/import/caller audits in `tests/contract/test_solver_retirement.py` from `specs/008-bigint-constant-synthesis/retirement.md`; reject old solver modules/dependencies, renamed copies, dynamic fallbacks and stale commands, while explicitly permitting independent reference and retained signed-256 controls.
- [ ] T082 [US3] Migrate supported callers in `src/oeis_learn/cli/main.py`, `evaluation/experiments.py`, `evaluation/synthesis.py`, `evaluation/repair_gate.py`, `experiments/config.py` and `experiments/qualified_models.py` to native transport/admission; document and remove unsupported methods rather than introducing compatibility fitting algorithms.
- [ ] T083 [US3] Migrate supported RL/replay consumers in `src/oeis_learn/rl/trainer.py` and `elite_buffer.py`; separate read-only historical evidence handling into `src/oeis_learn/synthesis/historical.py`, preserve scoped statuses and required certificate/reference checks and route supported UNSAT replay to the native core.
- [ ] T084 [US3] Port remaining false-fit/grounding regressions in `tests/contract/test_constant_solver_contract.py`, `test_symbolic_grounding_contract.py` and affected unit/integration tests to the new service and independent fixtures; preserve rejection power and historical artifact meaning rather than deleting failing tests.
- [ ] T085 [US3] Delete `src/oeis_learn/decoder/constant_solver.py`, `grounding.py`, `dixon_solver.py`, `qfnia_solver.py`, obsolete `grounding_reference.py`/`grounding_evidence.py`, and inventoried solver-only `mersenne61_filter.py`/`recurrence.py` code after T082–T084; remove `fresh/search.py::grounded_variants` and verify no renamed executable solver remains.
- [ ] T086 [US3] Remove inventoried obsolete run008/009/010 launch scripts and exclusive configurations/examples listed with exact paths in `specs/008-bigint-constant-synthesis/retirement.md`; retain original specs/results/reproduction commit IDs, update `README.md` migration guidance and account for every disposition without deleting supported subsystems by age.
- [ ] T087 [US3] Remove Python `z3-solver` and other proven solver-only dependencies from `pyproject.toml`, `uv.lock` and `docker/dev/install.sh`; retain SymPy only for documented active independent certificate/discovery consumers, keep native solver pins and rebuild clean installed images for the audit.
- [ ] T088 [US3] Close every entry in `specs/008-bigint-constant-synthesis/retirement.md` and record clean-install migration/compatibility evidence in `validation/phase-4.md`; require zero reachable old fitting implementations, all replacement counterexamples passing and no changes to historical numerical/evidence scope.

## Phase 5: Qualification and performance screen

**Entry**: T088 passes. These tasks execute integrated algorithms and calibrate explicit resource/config values; any new numerical, solver, encoder or decision policy reopens Phase 2. Actual engine/GPU access is an external gate, never fabricated.

- [ ] T089 [US1] Run frozen arithmetic, independent differential, codec and signed-256 compatibility suites through the qualification image; record exact source/image/runtime/compiler/lock hashes and zero-disagreement results in `specs/008-bigint-constant-synthesis/validation/arithmetic.md`.
- [ ] T090 [US2] Run native affine/SMT/scoped replay and all admission counterexamples, including large coefficients and partial/unknown outcomes; publish `specs/008-bigint-constant-synthesis/validation/fitting.md` with supported subset, bounds and provisional-versus-verified evidence.
- [ ] T091 [US3] Run resource/interruption/restart, role-isolation and ledger fault-injection suites under clean Docker and Podman; record deadline/cleanup bounds, healthy recovery, exact resume and socket/network/mount audits in `specs/008-bigint-constant-synthesis/validation/containment.md`; list any unavailable engine explicitly as pending.
- [ ] T092 [US1] Run actual ROCm tests from `tests/integration/test_bigint_gpu.py` via `validate --suite bigint-gpu`; record device and changed weights, finite4096-bit inputs/4097-bit differences, deterministic cached/full decoding and no CPU fallback in `specs/008-bigint-constant-synthesis/validation/gpu.md`, or give the user the exact quickstart command and leave this task unchecked.
- [ ] T093 [US2] Evaluate identified existing matched-run checkpoints through the retained-profile raw/exact paired diagnostic before parameterized training; record hashes, same-candidate rescue, unsupported fraction, continuation and fitting/reference cost in `specs/008-bigint-constant-synthesis/validation/checkpoint-rescue.md`; missing weights require an explicit evidence blocker, not substitution with canonical programs.
- [ ] T094 [US1] Measure old-backend versus bigint runtime at64/256/512/1024/4096/16384-bit scales with `scripts/benchmark_bigint.py`; separate compile/execute/solve/reference/generation/training time, cold/warm batch behavior and resource coverage, recording results in `specs/008-bigint-constant-synthesis/validation/performance.md`; investigate >5x median small-value slowdown without weakening semantics.
- [ ] T095 [US4] Prepare/check the frozen paired dataset and run complete-screen cost preflight using non-scoring fixtures via `scripts/foundation_run.py`; record complete512/128/128 counts, diverse32 composition, exclusions and projected p95 cost≤80% of budget in `specs/008-bigint-constant-synthesis/validation/screen-preflight.md`; preregister any resource/recipe revision before training.
- [ ] T096 [US4] Execute the diverse32 memorization gate and raw/shuffled/absent/template controls from `configs/foundation/bigint_smoke.yaml`; record complete coverage, termination, 31/32 verified greedy-plus-fitting continuation gate and selected checkpoint in `specs/008-bigint-constant-synthesis/validation/learnability.md`; failure blocks promotion, not an excuse to lower the threshold.
- [ ] T097 [US3] Exercise every supported quickstart command, clean retirement suite and normal/interrupted/bundle-cap exports; update `specs/008-bigint-constant-synthesis/quickstart.md` with verified commands and record mandatory artifact completeness, workspace persistence and historical reader results in `validation/operations.md`.
- [ ] T098 Record immutable qualification identities and all Phase 5 pass/fail/pending gates in `specs/008-bigint-constant-synthesis/validation/phase-5.md`; permit controlled screens only after required runtime, isolation, real-GPU, cost and learnability gates pass, and return algorithm defects to the Phase 2 owner.

## Phase 6: Controlled research and longer-run readiness

**Goal**: Collect interpretable evidence, including negative results. T098 is mandatory. A reviewed assisted manifest, access to checkpoints and GPU capacity are explicit execution prerequisites. No long run is launched simply to complete a task list.

- [ ] T099 [US4] Freeze complete matched contracts and an unexposed final cohort before the six screens in `specs/008-bigint-constant-synthesis/validation/preregistration.md`; identify common data/initialization, target vocab differences, seeds, selection/coverage/cost rules and already-exposed diagnostic cohorts, with no results-driven changes.
- [ ] T100 [US4] Run the three literal/parameterized seed pairs from `configs/foundation/bigint_literal_screen.yaml` and `bigint_parameter_screen.yaml`; retain workspace `runs/bigint-*` contracts/checkpoints/bundles and record all completed/interrupted/negative arms in `specs/008-bigint-constant-synthesis/validation/representation-study.md` without replacing inconvenient seeds.
- [ ] T101 [US4] Produce complete raw/rescue/equal-budget/correct/shuffled/absent/template comparisons using `scripts/foundation_run.py compare`; record full128+128 denominators, size/length/complexity strata, seed/family uncertainty and exact promotion/stop decision in `specs/008-bigint-constant-synthesis/validation/transfer-study.md`.
- [ ] T102 [US4] Freeze and run separate curriculum-then-replay comparisons only after selecting a supported recipe, using the qualified policies in `src/oeis_learn/fresh/learning_policy.py` and `replay.py`; record explicit configs, earlier-stage retention, training-only admissions, realized costs and outcome in `specs/008-bigint-constant-synthesis/validation/curriculum-replay-study.md`.
- [ ] T103 [US5] After a reviewed manifest and compatible generic foundation are supplied, run matched generic-only/assisted continuations from `configs/foundation/bigint_assisted_screen.yaml`; audit overlap and all descendants, report shared versus branch compute and retained alternative programs in `specs/008-bigint-constant-synthesis/validation/assistance-study.md`, leaving this task blocked rather than inventing demonstrations.
- [ ] T104 [US4] Run four post-training component ablations, then preregister and run selected matched retrainings from `configs/foundation/bigint_encoder_study.yaml` before examining final-cohort results; report active/total parameters, throughput, size/complexity strata, interactions and sensitivity-versus-retrained-utility in `specs/008-bigint-constant-synthesis/validation/encoder-study.md`.
- [ ] T105 [US5] Review discovery archives and final evidence labels using `src/oeis_learn/fresh/candidate_archive.py` and `synthesis/discovery_status.py`; publish finite matches, structural alternatives, omissions and separate conjecture/proof/novelty review in `specs/008-bigint-constant-synthesis/validation/discovery-review.md`, without automatically certifying novelty or a unique rule.
- [ ] T106 Produce the evidence-backed longer-run go/no-go report and proposed resource/throughput settings in `specs/008-bigint-constant-synthesis/validation/readiness.md`; link all bundles, pending blockers and gates, update `quickstart.md` next actions, and require a subsequent explicit run decision instead of automatically starting12+ hours of training.

## Dependencies and independent delivery

### Phase dependency graph

| Phase | Tasks | Entry | Exit / owner boundary |
| --- | --- | --- | --- |
| 1: Build/contracts | T001–T010 | Design documents | Pinned builds/fixtures/stubs ready |
| 2: Algorithm core | T011–T058 | T010 | T058: stable qualified semantic APIs; all hard work stays here |
| 3: Integration | T059–T080 | T058 | Bounded CPU workflow and persisted evidence |
| 4: Retirement | T081–T088 | T080 | Clean installed migration audit |
| 5: Qualification | T089–T098 | T088 | Real runtime/engine/GPU and bounded learnability evidence |
| 6: Research | T099–T106 | T098 | Reported transfer/attribution and explicit longer-run decision |

Phase 2 order is runtime/reference/codec/encoder → native fitting/admission → state → data/search/learning policies → provenance/archive → combined exit. T021 tests precede T022–T024 implementation; T039 precedes T040; T041 precedes T042–T043; each other test group similarly precedes its associated implementation. A later failing algorithm test returns to its Phase 2 task and invalidates affected qualification identities.

### User story completion order and acceptance

| Story | Depends on | Independently testable delivery | Full acceptance |
| --- | --- | --- | --- |
| US1 P1 | Phase 1 | T025 CPU arithmetic/input library, no trained model required | T089/T092/T094; exact values, limits, historical compatibility and actual-device updates |
| US2 P1 | US1 execution/reference | T036 native fitting against fixed candidates, no training required | T090/T093; independently verified fits, scope/ambiguity and checkpoint rescue evidence |
| US3 P1 | Qualified US1/US2 plus T038 state rules | T080 portable CPU workflow | T088/T091/T097; one installed fitter, recovery/resume and useful exports |
| US4 P2 | US1/US2, US3 production path | T052 deterministic data/policies and ablation interfaces | T095/T096/T100–T104; complete matched evidence and explicit negative/promotion outcome |
| US5 P2 | US2 admission, US4 grouping, US3 artifacts | T055–T057 fixture-based lineage/archive policies | T103/T105; reviewed assisted lineage and separate discovery evidence; reviewed data availability may block research |

**MVP**: US1's model-free bigint execution/reference corpus and large-input checks are the first useful increment (T025), kept behind an experimental profile. Complete the rest of the single Phase 2 handoff before beginning production integration. The first operational release includes US1–US3 and full retirement; a standalone arithmetic MVP is not a claim that the feature, GPU qualification or research gates are complete.

## Parallel execution examples

These are scheduling options, not a request to spawn agents. Use separate files and wait for prerequisites. Shared edits to `worker.py`, `foundation_host.py`, configuration dispatch and `hybrid.py` remain sequential.

| Story | Common prerequisite | Tasks safe to perform together | Join before |
| --- | --- | --- | --- |
| US1 | T010 | T011 language tests; T012 runtime tests; T013 reference tests | T014 implementation |
| US2 | T025 | T026 native fitting tests; T027 admission tests | T028 dependence implementation |
| US3 | T058 | T059 worker/process tests; T060 workflow tests | T061 transport implementation |
| US4 | T043 | T044 search/control tests; T045 learning/replay tests | T046 policy implementation |
| US5 | T052 | T053 provenance tests; T054 archive tests | T055 provenance implementation |

Native mathematical suites, Python regressions and portable container tests may run in distinct CI jobs against the same qualified cached image. GPU measurements sharing one device are serialized for interpretable timing. No new algorithm work moves outside Phase 2 merely to increase parallelism.

## Requirement and success-criterion traceability

| Requirement | Implementation tasks | Acceptance evidence |
| --- | --- | --- |
| FR-001–003, FR-013–015 | T011–T025, T067–T068 | T089/T091/T092/T094; SC-001–003/009 |
| FR-004–006, FR-016–017 | T026–T036, T061–T062, T069–T070 | T090/T093; SC-001/002 |
| FR-018 | T015/T018/T034/T043/T047/T066/T067/T071 | T079/T089/T091/T101; exact20/100 scope and hidden isolation |
| FR-007, FR-019–023 | T041–T052, T067/T071–T074 | T095/T096/T099–T102; SC-004/007–009 |
| FR-008–009, FR-024–025 | T053–T057, T075–T076 | T079/T099/T103/T105; lineage, clean final and discovery audit |
| FR-026 | T021–T024/T039–T040/T072–T074 | T092/T104; SC-003/004/009 |
| FR-010, FR-027–029 | T004/T037–T038/T059–T066/T077–T080 | T091/T097/T098; SC-002/006 |
| FR-011–012, FR-030–031 | T001/T081–T088 | T088/T097; SC-005 |

All nine success criteria are covered. Numerical/resource limits and required field relationships remain normative in the linked contracts and `data-model.md`; tasks cannot weaken them by omission. Unsupported hardware, missing checkpoints/demonstrations or an unsuccessful research gate are explicit blockers/negative outcomes, not authorization to skip evidence.

## Implementation strategy and completion rules

1. Deliver Phase 1 build feasibility and stop for the user's stronger-model handoff before T011. Dependency failures get a documented pin revision, not an alternate hidden backend.
2. Keep Phase 2 in one ownership window with reviewable runtime, fitter, encoder and policy commits. Use T025/T036/T052/T058 checkpoints; qualification depends on the combined exit.
3. Wire and test US1–US3, then finish full retirement before calling the native service the supported production path. Do not ship deprecated executable solver shims.
4. Qualify CPU/native/container behavior, then request actual-HIP results when unavailable locally. Update evidence as tests run; do not check off execution tasks from test-code presence.
5. Run the bounded staged research sequence only when prerequisites pass. Negative/inconclusive studies are complete research outcomes, but block longer-run promotion where the frozen gate requires improvement. Assistance and retraining studies need their own frozen manifests; no uncontrolled Cartesian sweep.
6. Every implementation commit identifies the affected task IDs and evidence. Later algorithm changes reopen Phase 2, create new affected contract identities and rerun relevant gates. Slow CI can be left running after push; failures remain visible and require repair before feature completion.

**Task-generation validation**: All106 tasks are unchecked, sequentially numbered and path-bearing; story labels and the11 conditional parallel tasks are audited. This statement validates this task document only, not the planned implementation.
