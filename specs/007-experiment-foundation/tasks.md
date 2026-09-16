# Tasks: Trustworthy Experiment Foundation

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts](contracts/cli.md).

**Status**: Implementation backlog; no task below is marked complete by this planning change. Tests are explicitly required by the feature specification. Write the relevant failing regressions before their fixes. Commands and newly named files are implementation targets, not claims that they exist now.

**Format**: `- [ ] Tnnn [P?] [USn?] Action with exact paths`. `[P]` means independent files within the same ready phase; it does not waive the dependencies below. Paths are relative to the repository root. Every task must leave its listed acceptance behavior reviewable; no training accuracy target is imposed.

## Phase 1: Setup and explicit adoption

- [ ] T001 Build a standalone Docker feasibility path in `docker/foundation/Dockerfile`, `docker/foundation/preflight.yaml`, `docker/foundation/requirements.lock`, `scripts/foundation/build_image.sh`, `scripts/foundation/run_container.sh` and `src/oeis_learn/cli/foundation_preflight.py`: resolve the plan's exact AMD image tag to a real digest, lock all installed Python dependencies/wheel hashes without replacing image PyTorch, inspect existing host GPU/device permissions and perform three finite FP32 forward/backward/AdamW parameter updates at 20 input terms and 1,024 body tokens (including zero, alternating signed-256 extremes and adjacent large-integer prefixes) within ten minutes after image availability. Persist actual identity/memory/timing evidence; fail `hardware_not_ready` without CPU fallback, driver installs or privileged containers. Provide its own preflight-only config and runnable module entry point, independent of the later foundation CLI/codec/pool. This standalone probe may run before adoption; it does not implement/start the comparison. (FR-017, SC-005)
- [ ] T002 Attach T001 feasibility evidence and record explicit maintainer adoption of `specs/007-experiment-foundation/constitution-rfc.md` in `specs/007-experiment-foundation/validation/adoption.md`; update `.specify/memory/constitution.md` status/date only if adopted, preserving the original ratification date and the historical compatibility map. If adoption is declined or evidence is absent, stop promotion beyond the standalone preflight and revise the proposal; do not claim approval. (FR-020)

**Checkpoint**: Feasibility evidence exists and the proposed governance is adopted. The unmodified legacy runtime may still fail numerical tests; preflight establishes only GPU feasibility.

## Phase 2: Foundational contracts and complete codec

- [ ] T003 [P] Add `tests/contract/test_foundation_artifacts.py` covering every required/unknown field, profile mismatch, hash/path rule, wrong horizon, `PROVEN`, floating integer, incomplete match and partial checkpoint in `specs/007-experiment-foundation/contracts/artifacts.schema.json` and `data-model.md`. Verify the schema examples are explicitly fixtures, never eligible run evidence. (FR-001, FR-005, FR-007, FR-008, FR-014, FR-019)
- [ ] T004 Implement `src/oeis_learn/experiments/{__init__,models,profiles}.py` and `configs/foundation/wat_profile.yaml` from the contracts. Preserve the exact constraints: "`schema_version` is exactly `foundation/v1`; unknown versions and unknown fields are rejected"; "A `Digest` is `sha256:` followed by exactly 64 lowercase hexadecimal characters"; "An `IntegerText` matches `^(0|-?[1-9][0-9]*)$`"; indices/counts/durations are nonnegative integers, not booleans. Implement exact range, path containment, enum/nullability and relationship checks in addition to JSON shape validation. Track/objective/initialization are exactly `strict_generic`/`sft`/`random`, horizon 20/100 and policy `prefix_rebased_zero`. Store the exhaustive opcode/wrapper/numeric/resource inventory and reject implicit backend/profile upgrades. (FR-001–002, FR-005, FR-007–008, FR-012, FR-018–019)
- [ ] T005 Implement strict configuration loading in `src/oeis_learn/experiments/config.py` and `configs/foundation/wat_smoke.yaml`, rejecting unknown keys, options for disabled solvers/optimizers/scaffolds/RL/proving, mismatched locks and incomplete identities. Persist every effective model constructor setting (dropout0.1, FiLM enabled, summary tokens disabled and explicit prime/modulus lists), FP32, limits and explicit unavailable/disabled values before work; the smoke is batch four, three updates, no scheduler/scaler. (FR-001, FR-012, FR-015, FR-017–018)
- [ ] T006 [P] Add `tests/unit/test_foundation_codec.py` for signed-256/i64/i32 bounds, negative and large constants, every declared local/branch scope, partially emitted literals, valid EOS, no UNK, no truncation and exact canonical token round trips. Include vocabulary sizes above 128 so a fixed-width mask cannot silently drop tokens. (FR-004, FR-013)
- [ ] T007 Implement `src/oeis_learn/decoder/program_codec.py` and adapt `decoder/{wat_grammar,grammar_masker,environment_tracker,sampler}.py` for `wat_body_decimal_v1`: fixed opcode/local tokens and signed digit immediates with explicit terminator, incremental type/scope masks and complete EOS. Enforce "`body_tokens` is nonempty, contains EOS exactly once at the end, contains no BOS/padding/unknown tokens, and is at most 1,024 tokens" and "Canonical source is at most 64 KiB". Preserve legacy codec identities for diagnostics; do not reuse their incompatible output weights. Depends on T004/T006. (FR-001, FR-004, FR-013)
- [ ] T008 Implement canonical artifact storage/identity helpers in `src/oeis_learn/experiments/artifacts.py`, initialize fixture support under `tests/fixtures/foundation/`, and register the strict command group in `src/oeis_learn/cli/{foundation,main}.py`. Use exact final-byte hashes, atomic writes, acyclic references, safe relative paths and an explicit diagnostic purpose. Reject unsupported commands until their slices are implemented. Depends on T003–T005. (FR-001, FR-011, FR-014–015, FR-020)

**Checkpoint**: Shared types, profiles, codec and CLI boundary are usable; legacy data and implicit defaults cannot enter the strict path.

## Phase 3: User Story 1 - Trust an accepted program (P1, first functional MVP)

**Goal**: Exact, bounded, independently checked WAT execution without requiring a model or OEIS corpus.

**Independent test**: Run G1 conformance with known counterexamples and injected failures; every accepted result agrees with the exact oracle and every error keeps its classified status.

- [ ] T009 [P] [US1] Extend `tests/unit/test_multi_limb_arithmetic.py` and `tests/contract/test_macro_instruction_lowering.py` with `1 * -1`, signed min/max, scalar `-2^63`, `2^64` constants, carries, native wrap, true logical overflow and lowered compilation. Expected arithmetic must not call production helpers/lowering. (FR-002–003, SC-001)
- [ ] T010 [P] [US1] Add `tests/unit/test_foundation_reference.py` for every allowed native opcode, signed division/remainder traps, structured branch stack, locals, all five macros, reference-step exhaustion and exact fresh-state semantics, including hand-calculated vectors independent of the shared parser. (FR-002–003, FR-006, SC-001)
- [ ] T011 [P] [US1] Add `tests/integration/test_foundation_execution.py` for malformed/unsupported source, bad arity, batch/single parity, numeric/runtime/limit distinctions, per-call versus aggregate fuel, blocked compilation/reference execution, killed workers and duplicate result delivery. Include a proposed `C*C` assignment C=4 for target4 and a local.tee-only declaration transformation. (FR-004–006, FR-018, SC-001)
- [ ] T012 [US1] Implement `src/oeis_learn/sandbox/wat_ast.py` with the frozen parser, type/return/scope validation and lossless canonical source emission. Forbid guest imports/memory/globals/calls and reserved helper namespace, enforce 32 i64/8 i32 fixed locals and depth limits, and accept no opcode outside the hashed inventory. (FR-001, FR-004, FR-006, FR-013)
- [ ] T013 [US1] Repair `src/oeis_learn/sandbox/{lowering,preamble}.py` and `sandbox/preamble.wat` for full signed-256 constants and exact checked add/sub/scalar multiplication; emit valid i64 limb literals and an unspoofable helper overflow marker, without treating native carries as overflow. Remove hardcoded helper fuel values from measured-cost fields. Depends on T009/T012. (FR-002–003, FR-016, SC-001)
- [ ] T014 [US1] Implement the bounded independent interpreter in `src/oeis_learn/sandbox/reference.py` using Python exact integers and separately written native-width rules; sharing AST parsing is permitted, calling production helpers/lowering/Wasmtime for expected values is not. Persist exact outputs, steps and failure reasons. Depends on T010/T012. (FR-003, FR-005–006, SC-001)
- [ ] T015 [US1] Implement prepare/execute/verify orchestration in `src/oeis_learn/sandbox/pipeline.py`, always validating final source and lowering before compilation. Enforce stage/outcome constraints, required independent evidence, full output counts and exact comparisons; reject test-injected false solver certificates and unsafe transformed source. Disable strict optional paths rather than repairing all legacy algorithms. Depends on T012–T014. (FR-003–005, FR-011, FR-013, SC-001–002)
- [ ] T016 [US1] Implement bounded workers/cache/watchdogs in `src/oeis_learn/sandbox/worker_pool.py`: fresh store/instance per term, reusable engine/module cache, exact cache keys, 32-request queue/1 MiB messages, per-call/aggregate work limits, two-second candidate deadline and reclamation within two more seconds. Kill/replace blocked workers; persist idempotent outcomes. Container isolation is integrated in T046. (FR-006, FR-016, FR-018, SC-001, SC-005)
- [ ] T017 [US1] Adapt `src/oeis_learn/sandbox/{runner,fallback_runner}.py` to the shared preparation/result contract for single and batch paths, including exactly four little-endian limbs and no `int(list)`/silent-zero coercion. Strict runtime selection is explicit Python/Wasmtime; mark the unqualified Rust adapter unavailable and prohibit auto-fallback. Depends on T015/T016. (FR-001, FR-004–006, SC-001)
- [ ] T018 [US1] Add `tests/integration/test_foundation_proof_boundary.py` and enforce result/archive checks in `src/oeis_learn/experiments/models.py` and `sandbox/pipeline.py`: "`proof_status` is exactly `not_claimed`". Reject the legacy shifted-identity false-proof label and imported `PROVEN`; finite matches remain finite evidence. (FR-019, SC-006)
- [ ] T019 [US1] Implement `foundation conformance` in `src/oeis_learn/cli/foundation.py`; run known regressions, 10,000 seed-7001 arithmetic vectors and 256 seed-7002 bounded structured programs using `tests/fixtures/foundation/conformance/`. Emit reports with zero unexplained disagreement as the pass criterion and explicit unavailable adapters; fix failures before exposing admission. (FR-003–006, FR-016, SC-001)

**Checkpoint**: US1 delivers a useful runtime conformance/admission core independently of trained weights.

## Phase 4: User Story 2 - Measure synthesis from twenty terms (P1)

**Goal**: Private exact truth, stable scoring units and genuine checkpoint-generated evidence.

**Independent test**: An instrumented parameter-dependent checkpoint fixture plus small source cohort exercises hidden/metadata perturbations, sealed selection, all denominator failures and finalization without requiring trained accuracy.

- [ ] T020 [P] [US2] Add `tests/unit/test_foundation_cohort.py` for signed bounds, source gaps/conflicting indices, offsets, complete100 versus prefix census, duplicate/prefix/shift connected components, large buckets, deterministic representatives/splits and requested-count shortfalls. Provide source-only fixture records under `tests/fixtures/foundation/source/` and `cohort.yaml`. (FR-007–010, SC-002)
- [ ] T021 [P] [US2] Add `tests/integration/test_foundation_evaluation.py` using an actual parameter-dependent model fixture: missing/wrong checkpoint, ignored weights, hidden/metadata perturbations, public-only seeds, no canonical substitution, capped/duplicate attempts, deterministic top-one, full denominator and final lock/seal crash/retry behavior. No trained success-rate threshold. (FR-008–011, FR-019, SC-002, SC-006)
- [ ] T022 [US2] Implement `src/oeis_learn/evaluation/foundation_cohort.py` with adapters in `data/benchmark.py`: exact first100 eligibility, hashed-window/exact-witness union-find grouping, frozen sorted-hash split and separate private truth/visible prompt/prefix-membership files. Enforce "observed_terms contains exactly 20 IntegerText values", one component/partition per eligible record, lexicographically smallest representative and no silent resizing. Preserve sufficient witness edges without quadratic all-pairs storage. Depends on T020. (FR-007–010, FR-013)
- [ ] T023 [US2] Extend `src/oeis_learn/evaluation/checkpoint.py` with a strict external-manifest loader validating final blob/config/profile/codec hashes, model architecture and checkpoint purpose before loading actual weights. Keep legacy inspection separate; do not convert a weights-only artifact to resumable state. Use complete fixture payloads from T021; the production writer follows in T041. (FR-001, FR-011, FR-014, SC-002)
- [ ] T024 [US2] Implement prefix-only generation in `src/oeis_learn/evaluation/foundation_synthesis.py`, reusing qualified sampler/model operations from `evaluation/synthesis.py`. Send only the narrow prompt view to the generation process, remove family/stage/ID scaffolds, enforce greedy-plus-seeded attempts and deadlines, and derive seeds solely from the public sampling projection plus visible values. Depends on T007/T015/T022/T023. (FR-008, FR-011–012, SC-002)
- [ ] T025 [US2] Implement immutable per-target candidate seals, deterministic `(body_token_count, source_sha256)` selection and full-horizon verification/scoring in `src/oeis_learn/evaluation/foundation_synthesis.py`. Require exactly100 outputs and independent evidence for success; preserve every frozen group in N; distinguish limited/failed candidates from incomplete infrastructure reports. "`selected` is frozen before full-stage verification." (FR-005, FR-008–011, FR-016, SC-002)
- [ ] T026 [US2] Implement `src/oeis_learn/evaluation/finalization.py`: decision lock before final access, visible-only proposal phase, durable per-target seals before hidden scoring, idempotent resume, no regeneration after feedback and no in-place checkpoint/protocol/stopping changes. Final report references the lock and all seals; missing seal after hidden exposure invalidates evaluation. (FR-009–011, FR-014, SC-002)
- [ ] T027 [US2] Adapt `src/oeis_learn/evaluation/readiness.py` and `cli/evaluate_canaries.py` so reference-kernel diagnostics never qualify model synthesis; remove invented memory values from the strict reporting path and require adopted profiles plus genuine checkpoint evidence. Legacy command output explicitly retains diagnostic provenance. (FR-011, FR-015–016, FR-019–020, SC-002, SC-006)
- [ ] T028 [US2] Implement `foundation freeze-cohort`, `evaluate` and `finalize` in `src/oeis_learn/cli/foundation.py` with the exact inputs/exit codes in `contracts/cli.md`, no truth mount passed to generation, and the fixture protocol at `tests/fixtures/foundation/evaluation.json`. Reject legacy20+100 manifests under the new version. (FR-007–011, FR-015)
- [ ] T029 [US2] Execute T020/T021 and add a representative CLI contract scenario in `tests/contract/test_foundation_cli.py` proving fixed denominators, actual-weight dependence, final sealing and zero qualifying scores from missing truth/checkpoints. Store the G2 software report under the run's `reports/` via `foundation_synthesis.py`. (FR-007–011, SC-002)

**Checkpoint**: US2 can report correct diagnostic synthesis outcomes from a fixture checkpoint; qualification still requires the run's actual hardware/provenance gates.

## Phase 5: User Story 3 - Preserve the strict learning track (P1)

**Goal**: An immutable, independently verified generic pool and a learner view that cannot import hidden or assisted material.

**Independent test**: Admission rejects each prohibited/corrupt record and a generated pool retains complete provenance and deterministic order.

- [ ] T030 [P] [US3] Add `tests/unit/test_foundation_admission.py` for imported OEIS/LODA programs, named teachers, old replay/weights, unknown/truncated tokens, missing conditioning, mismatched independent outputs, reserved-prefix collisions and cyclic/stale identities. Verify rejection reveals no matched target ID or continuation. (FR-012–013, SC-003)
- [ ] T031 [P] [US3] Add `tests/unit/test_foundation_pool.py` for deterministic sample counters, the declared generic sampler priors, source/output deduplication, shortest-then-hash representative selection, out-of-order worker completion, pool-build interruption/resume and bounded yield failure. (FR-001, FR-012–014, FR-016, SC-003–004)
- [ ] T032 [US3] Implement `src/oeis_learn/data/generic_programs.py` and `configs/foundation/generic_sampler.yaml` exactly from plan slice D: typed generic assignments/conditionals/counted loops, frozen probabilities/constants/depths/register counts and seed/counter provenance, with no named-family or target metadata branch. Emit the complete codec and report sampled priors/operator coverage. (FR-012–013, FR-016)
- [ ] T033 [US3] Implement `src/oeis_learn/data/program_admission.py` over T015 with exact100 production/reference agreement, token/source round trips, source/output limits and evaluator-owned prefix membership. "`origin` is exactly `generic_sample` for admitted 007 records." Hash the explicitly enumerated immutable program core; store evidence and AdmissionDecision separately referencing that ID, and never embed a decision reference in its own identity. (FR-001, FR-003–004, FR-012–013, SC-003)
- [ ] T034 [US3] Implement `src/oeis_learn/data/program_pool.py`: order decisions by sample counter, deduplicate sources and exact100-output fingerprints, archive rejected reasons/yields, checkpoint builder RNG/cursor/dedup/chunks, and publish an immutable manifest only after all admitted references verify. Frozen pool training uses no online generator/reservoir/replay. (FR-012–014, FR-016, SC-003–004)
- [ ] T035 [US3] Add the narrow trainer view and strict loader in `src/oeis_learn/data/program_pool.py`: program/codec IDs, exactly20 conditioning integers and valid body tokens only. Reject absent/unverified/mismatched pools; do not call `SftTrainer.load_or_generate_dataset` or the old named generator. Keep `data/synthetic_generator.py` labeled as a legacy/fixture path. (FR-008, FR-012–013, SC-003)
- [ ] T036 [US3] Implement `foundation build-pool` in `src/oeis_learn/cli/foundation.py`, isolating admission's membership access from the learner. Enforce the fixture's 64 unique outputs / 10,000 attempts / five-minute cap, and report constant fraction and rejection reasons on shortfall without adding teachers. (FR-012–013, FR-016, FR-018, SC-003)
- [ ] T037 [US3] Run the negative admission and pool-order tests plus the isolated generic pool fixture through `tests/integration/test_foundation_bootstrap.py`; verify every admitted record's provenance, evidence and round trip before releasing the pool. (FR-003, FR-012–013, SC-003)

**Checkpoint**: US3 supplies an auditable frozen pool with no requirement that the current model already solves an OEIS family.

## Phase 6: User Story 4 - Run and resume on the workstation (P1)

**Goal**: Strict SFT with complete continuation, true elapsed budgets, measurable device use and enforced resource limits.

**Independent test**: Deterministic CPU interrupted/uninterrupted equivalence plus a separate actual-GPU smoke and fault-injection suite.

- [ ] T038 [P] [US4] Add `tests/integration/test_foundation_resume.py` with injected clocks and interruption at a completed update, checking model/optimizer/active scheduler+scaler/all RNG/permutation/cursor/counters, checkpoint hash integrity, corrupt-newest rollback, prefetched order, duplicate events and nonrefundable crash-gap accounting. Compare full state and next examples, not weights alone. (FR-014, FR-016, SC-004)
- [ ] T039 [P] [US4] Add `tests/unit/test_foundation_config_metrics.py` for unknown/inactive options, measured-zero versus disabled/unavailable, stable effective configuration, monotonic same-boot/cross-boot recovery, backward clocks and bounded segmented logs. (FR-015–016, FR-018, SC-004)
- [ ] T040 [P] [US4] Add `tests/integration/test_foundation_resources.py` and `test_foundation_gpu.py` for explicit HIP tensor residency/changed parameters, unavailable GPU, slow/hung calls, external deadline, worker OOM/kill/replacement, queue/cache caps, fake disk quota exhaustion and mount isolation. Hardware tests are explicitly marked; use temporary bounded fixtures, not actual host exhaustion. (FR-006, FR-017–018, SC-005)
- [ ] T041 [US4] Implement `src/oeis_learn/tracking/training_checkpoint.py` using the external schema/loader: save all eleven payload state keys, with disabled scheduler/scaler explicitly null and no partially accumulated gradients; flush, hash, rename/fsync immutable blob before atomic manifest/latest publication. Verify newest-to-oldest on recovery and retain two valid generations plus one pinned final within40GiB. "Partial/corrupt files are quarantined." (FR-001, FR-014, FR-018, SC-004)
- [ ] T042 [US4] Implement `src/oeis_learn/tracking/budget_ledger.py` as the independent controller's durable start/heartbeat writer; account for active bootstrap/training/evaluation/checkpoint phases, exclude clean-pause/build time, and conservatively charge unobserved crash gaps through recovery. Block backward/inconsistent clocks, preserve ledger high-water mark on model rollback and enforce remaining-arm deadlines externally. Distinguish measured active from estimated charge; never reset the clock on resume. (FR-014, FR-016, FR-018, SC-004)
- [ ] T043 [US4] Implement `src/oeis_learn/tracking/foundation_metrics.py` and adapt `tracking/run_manager.py` for bounded append-only event segments, lifecycle/qualification separation and typed measured/disabled/unavailable metrics. "A measured zero is valid; disabled/unavailable values are null." Record generator/update/token/candidate/cost/device/RSS/allocator statistics without double-counting shared APU memory or substituting fixed values. (FR-015–016, FR-018, SC-005)
- [ ] T044 [US4] Implement `src/oeis_learn/rl/foundation_sft.py` reusing qualified training operations from `rl/sft_trainer.py`: random initialization, existing small FP32 backbone, complete new codec, only admitted pool input, mean-per-program masked token loss including EOS, AdamW3e-4/0.01, clip1.0, smoke batch4/three updates and explicit null scheduler/scaler. Named RNG streams and immutable data order must feed T041/T042; no implicit device or teacher fallback. (FR-012–016, SC-003–004)
- [ ] T045 [US4] Implement `foundation train`, `resume`, `inspect` and complete `preflight` registration in `src/oeis_learn/cli/foundation.py`, returning actual artifact paths and rejecting resume-time config overrides, weights-only state, missing profiles/pool or incompatible runtime. CPU use requires explicit diagnostic mode and remains unqualified for Ryzen readiness. (FR-011, FR-014–017, SC-004–005)
- [ ] T046 [US4] Complete `docker/foundation/compose.yaml` and `scripts/foundation/run_container.sh` with fixed role mounts, networking disabled, device exposure only to learner/preflight, no Docker socket in workloads, separate truth/generation/learner views, at most8 one-GiB workers and an88GiB learner/controller, swap/PID/cache/queue/disk limits and host/GPU memory monitoring. Controller launcher must kill/reclaim hung workloads within declared deadlines. (FR-006, FR-008, FR-017–018, SC-005)
- [ ] T047 [US4] Run the hardware-marked tests and end-to-end strict pool/model smoke via `tests/integration/test_foundation_gpu.py`, recording real runtime image/lock, finite three-update parameter changes, timings/resources and fault containment. Keep failed host compatibility or numerical probes visible; never turn CPU diagnostics into a hardware pass. (FR-017–018, SC-005)
- [ ] T048 [US4] Run interrupted/resumed CPU and corrupt-checkpoint recovery through `tests/integration/test_foundation_resume.py`, demonstrate equal next samples/update/budget under its deterministic clock, and report GPU nondeterminism limits separately in `docs/foundation.md`. Resolve concrete mismatches before claiming resume. (FR-014, FR-016, SC-004)
- [ ] T049 [US4] Integrate foundation readiness in `src/oeis_learn/evaluation/readiness.py` and `tracking/run_manager.py`: require adopted contract, exact/provenance/isolation/resume/device/containment/proof-boundary evidence for qualification. A fixture/override/legacy run remains diagnostic; failed/incomplete gate cannot be hidden by a final score. (FR-011, FR-015–020, SC-001–006)

**Checkpoint**: US4 proves bounded real-device operation and honest continuation; it does not establish useful learning from a three-update model.

## Phase 7: Cross-cutting validation and handoff

- [ ] T050 [P] Document implemented CLI, isolation, numerical scope, migration/rollback, evidence interpretation and recovery in `docs/foundation.md` and update `README.md`; reconcile `specs/007-experiment-foundation/quickstart.md` with actual command behavior, including measured versus pending gates and no automatic008 launch. (FR-002, FR-011, FR-014–020)
- [ ] T051 Run all new foundation tests and affected existing arithmetic/lowering/codec/synthesis/checkpoint suites, record exact commands/environment and any pre-existing failures in `specs/007-experiment-foundation/validation/implementation-results.md`, and resolve introduced failures. Re-run official Spec Kit prerequisites plus `specs/007-experiment-foundation/validate_artifacts.py` and perform read-only cross-artifact analysis after code-driven design changes. Documentation checks cannot substitute for runtime gates. (FR-001–020, SC-001–006)
- [ ] T052 Execute the implemented `specs/007-experiment-foundation/quickstart.md` scenarios, archive all G0–G6 evidence/immutable identities, and create `specs/007-experiment-foundation/validation/handoff.md` identifying passed/pending gates and the precise interface for008. Stop on an unresolved acceptance gate; hand off only the qualified WAT foundation and recorded limits, with native LODA/comparison training/proof expansion still deferred. (FR-001–020, SC-001–006)

## Dependencies & Execution Order

- **Setup**: T001 → T002. Adoption is a real gate, not an assumed task completion.
- **Foundation**: T003 and T006 can run in parallel after setup; T004 follows T003; T005 follows T004; T007 follows T004/T006; T008 follows T003–T005. Finish the phase before story implementation is integrated.
- **US1**: T009/T010/T011 are parallel regression work. T012–T017 build parser, arithmetic/reference, pipeline and workers in that dependency order; T013 and T014 can proceed separately once T012 exists and their regressions fail for the expected reason; both must pass before T015 integration. T018 proof boundary and T019 conformance close G1/G6.
- **US2**: T020/T021 are independent tests. T022 and T023 can develop independently, then T024 → T025 → T026. T027/T028 integrate reporting/CLI; T029 closes the slice. US1 is required for production qualification; model fixtures avoid depending on successful training.
- **US3**: T030/T031 are parallel tests. T032 and T033 are independent once US1/codec contracts exist; T034 → T035/T036 → T037 freezes and qualifies the pool. It can be developed alongside US2 with a private membership fixture, but real pool publication requires T022's frozen cohort.
- **US4**: T038/T039/T040 are parallel tests. T041/T042/T043 feed T044; T045/T046 integrate lifecycle and containers; T047/T048 collect device/resume evidence; T049 combines gates. Strict SFT requires US3's admitted pool. Evaluation from its real checkpoints requires US2.
- **Finish**: T050 can be prepared alongside validation once command contracts stabilize; T051 then T052. Do not start008's long experiment merely because document scripts return zero.

The smallest useful MVP is US1 after setup/foundation. US2 adds trustworthy diagnostic synthesis; US3 adds a strict pool; US4 makes the complete path resumable and workstation-qualified. Isolated component tests may use declared fixtures; those fixtures never qualify real experimental results.

## Parallel examples

| Ready phase | Independent work | Join condition |
| --- | --- | --- |
| Foundation | T003 artifact tests and T006 codec tests | Their implementation dependencies complete before T007/T008 integration. |
| US1 | T009 arithmetic, T010 reference and T011 worker/parity regressions | T015 requires corrected helpers and independent reference. |
| US2 | T020 cohort tests and T021 checkpoint/leakage tests | T024 requires actual loader plus private/visible views. |
| US3 | T030 admission tests and T031 sampler/pool tests | T034 requires valid generation and admission. |
| US4 | T038 resume, T039 metrics/config and T040 GPU/resource tests | T049 requires all runtime gates. |

Do not concurrently edit shared `cli/foundation.py`, `experiments/models.py` or `evaluation/readiness.py` from different slices without sequencing their integration. `[P]` is a file-independence hint, not permission for conflicting edits.

## Requirement and success-criterion coverage

Mappings identify substantive implementation/test work; the final aggregate validation tasks are additional coverage, not the sole implementation of a requirement.

| Requirement | Task IDs | Acceptance gate |
| --- | --- | --- |
| FR-001 | T003, T004, T005, T007, T008, T012, T017, T023, T031, T033, T041 | G1–G4 |
| FR-002 | T004, T009, T010, T013, T050 | G1 |
| FR-003 | T009, T010, T013, T014, T015, T019, T033, T037 | G1/G3 |
| FR-004 | T006, T007, T011, T012, T015, T017, T019, T033 | G1/G3 |
| FR-005 | T003, T004, T011, T014, T015, T017, T019, T025 | G1/G2 |
| FR-006 | T010, T011, T012, T014, T016, T017, T019, T040, T046 | G1/G5 |
| FR-007 | T003, T004, T020, T022, T028, T029 | G2 |
| FR-008 | T003, T004, T021, T022, T024, T025, T028, T029, T035, T046 | G2/G3 |
| FR-009 | T020, T021, T022, T025, T026, T028, T029 | G2 |
| FR-010 | T020, T021, T022, T025, T026, T028, T029 | G2 |
| FR-011 | T008, T015, T021, T023, T024, T025, T026, T027, T028, T029, T045, T049, T050 | G2 |
| FR-012 | T004, T005, T024, T030, T031, T032, T033, T034, T035, T036, T037, T044 | G3 |
| FR-013 | T006, T007, T012, T015, T022, T030, T031, T032, T033, T034, T035, T036, T037 | G3 |
| FR-014 | T003, T008, T023, T026, T031, T034, T038, T041, T042, T044, T045, T048, T050 | G4 |
| FR-015 | T005, T008, T027, T028, T039, T043, T044, T045, T049, T050 | G4/G5 |
| FR-016 | T013, T016, T019, T025, T027, T031, T032, T034, T036, T038, T039, T042, T043, T044, T048, T049, T050 | G4/G5 |
| FR-017 | T001, T005, T040, T045, T046, T047, T049, T050 | G0/G5 |
| FR-018 | T004, T005, T011, T016, T036, T039, T040, T041, T042, T043, T046, T047, T049, T050 | G1/G5 |
| FR-019 | T003, T004, T018, T021, T027, T049, T050 | G6 |
| FR-020 | T002, T008, T027, T049, T050, T051, T052 | G0/handoff |
| SC-001 | T009, T010, T011, T013, T014, T015, T016, T017, T019, T049 | G1 |
| SC-002 | T015, T020, T021, T023, T024, T025, T026, T027, T029, T049 | G2 |
| SC-003 | T030, T031, T033, T034, T035, T036, T037, T044, T049 | G3 |
| SC-004 | T031, T034, T038, T039, T041, T042, T044, T045, T048, T049 | G4 |
| SC-005 | T001, T016, T040, T043, T045, T046, T047, T049 | G0/G5 |
| SC-006 | T018, T021, T027, T049 | G6 |

## Implementation strategy

Deliver each story as a reviewed increment, preserve failed-gate evidence and fix the concrete cause before expanding runs. Keep all 52 tasks unchecked until their implementation/evidence exists. The 1–3-day language arms, native LODA runtime and any later architectural experiments are not tasks in007. A gate failure changes the implementation work needed; it does not silently authorize a teacher, wider numerical backend, new host dependency or additional experiment arm.
