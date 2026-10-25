# Tasks: Bounded Pipeline Throughput and Research Completion

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), all four [contracts](contracts/), [carryover.md](carryover.md), [quickstart.md](quickstart.md), constitution v2.0.0.

**Status**: Hardware-first work now includes the user-requested separate [model-size screen](validation/model-sizing-implementation.md):8.12M/31.90M/104.50M parameters,165 host tests and verified local export pass; current CPU/HIP sizing collection remains pending. Accepted co-scheduling O004 remains complete at5014881. Original counts77 complete/19 runtime pending/15 research held or blocked reflect reopening affected-sourceT110. No production default, learning-quality/curriculum or multi-day admission.

**Organization**: Shared setup/foundations precede story work. Delivery phases **1–6 retain the approved plan and CLI numbering** rather than renumbering them to match the generic template's setup phases. Phase 1 contains the US1 MVP and US2 measurement foundations; US2 continues in phase 2; US3 spans phases 3–4; US4 owns phase 5 and US5 phase 6. Each story has an independent acceptance boundary below.

**Format**: `- [ ] Tnnn [P?] [USn?] Description with exact repository-relative file paths`. New paths are implementation targets, not assertions that files already exist. `[P]` means parallel with the explicitly listed peers after common prerequisites; it never authorizes competing hardware measurements. Setup/foundational/polish tasks have no story label. Write the specified regression/contract tests first, demonstrate the relevant failure, then implement; tests are required by the specification and constitution for these changes.

**Completion rules**: Check code/local-test tasks only with their evidence; check hardware tasks only after reviewing the required real-device/engine bundle. A blocked research execution stays unchecked even when its orchestration code is complete. Record blockers, deferrals and negative outcomes separately. A measured no-adoption can complete an investigation but cannot satisfy SC-004's improvement claim. Prefix historical references with **008:** to distinguish them from this file's IDs. Do not alter historical checkboxes merely to transfer ownership.

## Shared setup

- [X] T001 Record the implementation baseline, supported command inventory and existing adapter ownership in `specs/009-pipeline-throughput/validation/implementation-baseline.md`; verify the PR #59 dependency and preserve existing behavior before refactoring.
- [X] T002 Create the stdlib-only host package boundary in `src/oeis_learn/pipeline_validation/__init__.py` and establish the static module layout from `specs/009-pipeline-throughput/plan.md`; do not add a scheduler service, host dependency or second trainer.
- [X] T003 [P] Add a machine-readable task-to-gate/carryover index in `specs/009-pipeline-throughput/validation/task-evidence.json`, initially pending, distinguishing code/local/HIP/engine/research dispositions and the explicitly deferred 008:T093.

## Shared foundations — required before story integration

- [X] T004 [P] Add schema/identity rejection tests in `tests/unit/test_pipeline_contracts.py` for unknown keys, nonfinite/negative timing, path-only identities, changed policies, duplicate logical IDs and immutable terminal records; use independent expected payloads and known old v1 fixtures.
- [X] T005 Implement shared canonical identity and confined-reference helpers in `src/oeis_learn/pipeline_validation/identity.py`, reusing existing hashes: "Unknown keys fail closed for versioned authority-bearing records", "Timing values are finite, nonnegative observations and never substitute for identities", and "a path string alone is never an input identity"; preserve old v1 identity/seal validators.

  > Every authority-bearing document has `version`, `identity` and an immutable canonical payload. Compute identity over the payload excluding its own identity and mutable presentation fields. Paths are workspace-confined references plus content hash; a path string alone is never an input identity. Large/private payloads are not embedded in public bundles.
  >
  > Record both full source snapshot and the explicitly versioned dependency closure used for a stage. Bind image digest, installed package inventory, native binary, guest, numerical/engine/fit/reference policies and workload hashes as relevant. Code paths that ignore an effective policy in contract identity must be corrected before reusing their artifacts. Old v1 identity and seal validators stay available; never rewrite an old record to look like v2.

- [X] T006 [P] Add resource-resolution tests in `tests/unit/test_pipeline_resources.py` for measured physical/ancestor limits, quota/cpuset, current availability, unknown sensors, shared CPU/GPU charging and reservation-versus-ceiling distinctions; prove the old Host 16 GiB default cannot override workstation90.
- [X] T007 Implement `ResourcePolicy` resolution/reservations in `src/oeis_learn/pipeline_validation/resources.py`: ceiling `min(floor(0.90 * physical_bytes), effective_engine_or_ancestor_limit)` rounded down to MiB, profiles historical16/capacity64/workstation90, expected peak plus `max(1 GiB, 10% peak)` headroom, one full-envelope compute role, and explicit unavailable accounting; "Resource policy changes acquire a new hash".
  <details>
  <summary>Required ResourcePolicy constraints (verbatim from data-model.md)</summary>

  > ResourcePolicy — `pipeline-resource/v1`
  >
  > Fields: effective CPU count/quota/cpuset; physical/currently available/engine/ancestor memory; user-authorized fraction0.90; selected historical16/capacity64/workstation90 profile, resolved enforcement ceiling, separate expected active peak reservation and safety headroom; device identity; native slot count and affinity; Torch/native thread limits; active/pending/result/lookahead counts and byte charges; per-slot and aggregate RSS watchdogs; role-wide cache caps; candidate ResourceProfile hash; stream/artifact/archive limits; external/cleanup deadlines. Missing sensors use explicit unavailable reasons. Distinguish observed RSS/shared-memory counters from enforced container limits.
  >
  > Resource policy changes acquire a new hash. A validation command may lower an invocation ceiling, but it cannot silently reinterpret a hardware comparison performed under different limits. At most one active compute role receives the full envelope.

  </details>

- [X] T008 Bind new manifests, adapters, tests and policy dependencies into `src/oeis_learn/experiments/source_snapshot.py` and `src/oeis_learn/pipeline_validation/closures.py`; implement `DependencyClosure` with explicit path/content hashes and conservative unmapped-change invalidation, while retaining mandatory current-image whole-package inspection.
  <details>
  <summary>Required DependencyClosure constraints (verbatim from data-model.md)</summary>

  > DependencyClosure
  >
  > Fields: versioned rule ID; explicit source/build/config/test/adapter paths and hashes; prerequisite artifact/receipt identities; exclusions with rationale; closure hash. An unmapped changed path invalidates relevant receipts conservatively. A whole-source/package check remains mandatory for the current image. Timing reuse additionally requires identical effective hardware and comparison conditions; reports disclose age and producer source even when compatible.

  </details>


**Foundation checkpoint**: T001–T008 pass locally; shared identities/resource contracts are usable without importing Torch or YAML on the host. Future story modules can be developed against these contracts before hardware evidence arrives.

## Phase 1 — US1: One-command validation (P1, MVP)

**Goal**: One invocation resolves ready scope, shares preparation, runs safe independent checks and exports one trustworthy bundle, including failures.

**Independent test**: A stdlib fake-engine invocation with a deliberately failed prerequisite builds each target once, blocks descendants, continues an independent stage, emits liveness and produces a verified non-PASS archive. Missing HIP never becomes PASS. No training or final access occurs.

### Tests first

- [X] T009 [P] [US1] Add full-host CLI/DAG tests in `tests/unit/test_pipeline_validation.py` for phase/all-ready selection, explicit unavailable phases, capability failures, build deduplication, continued independent stages, exit codes and engineering-authority rejection before engine access.
- [X] T010 [P] [US1] Add receipt/reuse tests in `tests/unit/test_pipeline_reuse.py` covering source/image/lock/adapter/workload/hardware/resource mutations, broken hashes, missing mandatory files, prior failures, unknown closure changes and compatible documentation-only changes with preserved producer identity.
- [ ] T011 [P] [US1] Add archive/failure tests in `tests/unit/test_pipeline_bundle.py` for arbitrary UID, unreadable mandatory files, symlinks/traversal, full disk, capped optional details, contradictory summaries, interrupted builds/startup and surviving minimal failure records.

### Implementation

- [X] T012 [US1] Implement `ValidationContract` parsing in `src/oeis_learn/pipeline_validation/contracts.py` and finite defaults in `configs/foundation/pipeline_validation.json`; engineering authority rejects learning/resume/final access, unknown stages/cycles/duplicate IDs fail closed, and "reservations must fit the parent"; encode all operator-contract caps without borrowing phase time.
  <details>
  <summary>Required ValidationContract constraints (verbatim from data-model.md)</summary>

  > ValidationContract — `pipeline-validation/v1`
  >
  > | Field | Meaning / validation |
  > | --- | --- |
  > | `scope`, `phases`, `stage_registry_version` | `engineering`; phase1–5 or `all-ready`; registered implementation versions only |
  > | `authority` | Explicitly denies learning/resume/final evaluation; permits only declared diagnostic disposable updates and bounded operations fixtures |
  > | `source`, `input_refs`, `weights_ref` | Producer/current source and closed dependencies, compatible completed run/prepared inputs, frozen selected weights hashes |
  > | `engines`, `resource_policy` | Primary engine, requested portability coverage, effective CPU/memory/device policy and finite limits |
  > | `workloads`, `comparisons` | Hashes of frozen manifests and prospective candidate/measurement policies |
  > | `budgets` | Global, per-engine build, phase, stage, role, cleanup/export, disk/log/archive and job limits; reservations must fit the parent |
  > | `reuse_refs` | Prior immutable receipt/bundle hashes, never permission to inherit a whole-run result |
  > | `excluded_authorities` | Training, T096 retry/resume, assisted-data invention and final access are rejected, not simply unused |

  </details>

- [X] T013 [US1] Implement `Invocation` and `StageReceipt` ledgers in `src/oeis_learn/pipeline_validation/receipts.py`: fresh output roots, append-only transitions, separate scope/engineering/learning/research/portability/long-run results and original producer links; "All terminal states are immutable" and "A missing terminal record is interrupted/partial, never inferred complete".
  <details>
  <summary>Required Invocation constraints (verbatim from data-model.md)</summary>

  > Invocation — `pipeline-invocation/v1`
  >
  > Fields: unique invocation ID; resolved contract hash; started/ended wall timestamps; monotonic origin/durations; selected scope; unavailable/unrequested registry stages; parent reuse links; phase/gate status; actual build/startup/compute/archive costs; consumed/reserved budgets including conservative charges for unacknowledged work; interrupted flag; output/ZIP paths and integrity; explicit historical T093/T096 dispositions.
  >
  > An invocation owns a fresh output root. Old invocation records are immutable. Repeating a failed test uses a new invocation, not an edit to a terminal status. A missing terminal record is interrupted/partial, never inferred complete.

  </details>

  <details>
  <summary>Required StageReceipt constraints (verbatim from data-model.md)</summary>

  > StageReceipt — `pipeline-stage/v1`
  >
  > Fields: stage ID/version; attempt ID; state/outcome/reason; exact executable/argument vector with no secrets; prerequisite receipt hashes; current source and dependency closure; image/artifact identities; workload/weights/policies; engine/hardware/resource identity where relevant; expected/started/completed work; elapsed/CPU/resource observations; mandatory file hashes; JUnit counts; omissions; reused-from identity; child role audits and cleanup result.
  >
  > Transitions:
  >
  > - `PENDING → READY → RUNNING → PASS | FAIL | PARTIAL | INCONCLUSIVE`.
  > - `PENDING | READY → BLOCKED` on failed/missing prerequisites.
  > - `PENDING → REUSED` only after validating a complete matching PASS receipt; preserve its original producer/measurement identity.
  > - `UNAVAILABLE` and `NOT_SELECTED` describe registry/scope availability; neither counts as execution.
  >
  > All terminal states are immutable. Stage-level PASS does not imply scientific success; a valid report of a failed performance hypothesis can complete a reporting stage while `adoption=NO_ADOPTION`. Top-level status retains separate `selected_scope`, `engineering`, `learning`, `research`, `portability` and `long_run_admission` results.

  </details>

- [X] T014 [US1] Implement the static stage registry in `src/oeis_learn/pipeline_validation/registry.py` with declared availability, prerequisites, capabilities, input closures, reservations and adapters; distinguish NOT_SELECTED/UNAVAILABLE/BLOCKED from executed success and keep phase 6 outside all-ready.
- [X] T015 [US1] Implement topological execution in `src/oeis_learn/pipeline_validation/runner.py`, serializing measurements/builds, continuing independent safe stages after failure, stopping globally on interruption/resource/role-audit breach, and reserving cleanup/archive time before dispatch; report every selected stage's terminal outcome.
- [X] T016 [US1] Factor existing qualification/profile operations into `src/oeis_learn/pipeline_validation/adapters.py`, `src/oeis_learn/container_validation.py`, `scripts/qualify_bigint_phase5.py` and `scripts/profile_bigint_pipeline.py`; retain legacy entry points while sharing Host/image checks and suppressing nested archives.
- [X] T017 [US1] Wire public options and early memory-policy resolution into `scripts/foundation_local.py` through `scripts/foundation_run.py`: phases 1–5/all-ready, manifest, read-only run/inputs, fresh output, repeated reuse, engines available/selected/both, lower-only max-seconds and scope-report; preserve existing suites and primary-image pinning semantics.
- [X] T018 [US1] Implement capability/image adapters in `src/oeis_learn/pipeline_validation/engines.py`: ensure each engine/target once, share each engine's 14,400-second build allowance, inspect current package/native/guest identities, require nonempty mandatory JUnit with no mandatory skips, and separate CPU-only readiness from real HIP updates.
- [X] T019 [US1] Implement dependency-bound PASS imports in `src/oeis_learn/pipeline_validation/reuse.py`; verify all mandatory artifacts and prerequisite hashes, bind timing to exact hardware/resource/measurement conditions, link old failures/costs, restart interrupted confirmation blocks and never refund research budgets.
- [X] T020 [US1] Implement build/startup/stage supervision in `src/oeis_learn/pipeline_validation/supervision.py`: 30-second heartbeat, prompt SIGINT forwarding, external deadlines, bounded process output and cleanup; preserve terminal evidence even when startup fails before a container ID exists.
- [X] T021 [US1] Implement `BundleManifest` and mandatory-first export in `src/oeis_learn/pipeline_validation/bundle.py`: 8 MiB/stream, 8 GiB transient, 128 MiB traces, 512 MiB ZIP, computed mandatory reserve at least 16 MiB, regular readable confined files and per-member size/hash checks; exclude weights/private truth/input payloads/caches/sockets and export one top-level ZIP.
  <details>
  <summary>Required BundleManifest constraints (verbatim from data-model.md)</summary>

  > BundleManifest
  >
  > Fields: invocation/contract identity; archive version; each member's relative path, byte count and SHA-256; mandatory/optional designation; omissions with reason; producer scope/source; excluded private categories. Reject path traversal, symlinks, unreadable mandatory records, hash/size mismatch or contradictory summary state. Core job/stage evidence takes priority over optional logs/traces. The archive is evidence, not a checkpoint and not a source of new execution authority.

  </details>

- [X] T022 [US1] Implement scope/verdict rendering in `src/oeis_learn/pipeline_validation/report.py`, including exact failed paths, source/image identities, omissions, consumed/reserved budgets and exits 0/1/2/3/130; a failed report must not erase an earlier failure or confuse selected-scope PASS with feature completion.
- [X] T023 [US1] Verify the complete host entry point in a stdlib-only subprocess and legacy-wrapper regression suites through `tests/unit/test_foundation_local.py` and `tests/unit/test_pipeline_validation.py`; record command/results in `specs/009-pipeline-throughput/validation/phase-1-local.md`.
- [X] T024 [US1] Register implemented phase-1 CPU/engine checks in `configs/foundation/pipeline_validation.json` and update `specs/009-pipeline-throughput/quickstart.md` with one usable all-ready command, input requirements, scope-report and fresh-output reuse; do not advertise unimplemented later stages as ready.
- [ ] T025 [US1] Run/review real-engine phase-1 orchestration, permission-safe export, deliberate failed-prerequisite and interruption fixtures via the combined entry point; retain the receipt index in `specs/009-pipeline-throughput/validation/phase-1-operator.json`, with missing second-engine/HIP evidence explicitly pending.
- [ ] T026 [US1] Close the US1 acceptance matrix in `specs/009-pipeline-throughput/validation/phase-1-operator-review.md`, confirming no separate manual build/test/profile/archive command, no dependent expensive work after failed prerequisites and no research authority inferred from validation.

## Phase 1 — US2 measurement foundations (P1)

**Goal**: Freeze representative work and finite decisions before candidate timing; supply a useful baseline without rerunning every historical investigation.

**Independent test**: Frozen complete/partial fixture pairs produce correct denominators, exclusions and no-adoption decisions; actual baseline observations establish only the scopes they measured.

- [X] T027 [P] [US2] Add comparison/adoption tests in `tests/unit/test_pipeline_comparisons.py` for complete versus partial work, full fixed-budget denominators, frozen exclusions, five-pair order, stratum regressions, reset/cold/warm mismatches and prohibition of favorable-pair reuse or adaptive extra samples.
- [X] T028 [US2] Implement `ComparisonContract` and calibration sealing in `src/oeis_learn/pipeline_validation/comparisons.py`: one cold baseline, one warmup, three baseline-only calibrations; derive `ceil(max(2 * max_elapsed, declared_minimum))` reservations within parent caps before candidates; "A candidate run cannot mutate its parent's expectations".
  <details>
  <summary>Required ComparisonContract constraints (verbatim from data-model.md)</summary>

  > ComparisonContract — `pipeline-comparison/v1`
  >
  > Fields: parent validation identity; workload membership; baseline/candidate source and policy hashes; primary metric; thresholds; seed/order roster; warmup/calibration/confirmation count; cold/warm cache policy; row/attempt/control denominators; token, service and group deadlines; numerical tolerances; telemetry/trace settings; frozen exclusion manifest; expected artifact/aggregate work; allowed outcome interpretations.
  >
  > If calibration derives finite reservations, store the derivation-rule hash, complete calibration receipt hashes and resulting child contract before the first candidate measurement. A candidate run cannot mutate its parent's expectations. Confirmation IDs differ from exploratory IDs; imported old samples never become fresh confirmation.

  </details>

- [X] T029 [US2] Freeze representative workloads/strata in `configs/foundation/pipeline_validation.json` and `src/oeis_learn/fresh/pipeline_profile.py`: cache positions 1/32/128/513/1024, generation32's 576 attempts, native legal 64–4096-bit inputs/16,384-bit intermediates, 500 disposable updates/256 validation rows, checkpoint/export/startup and separate 1,024-attempt complete-control cost; retain historical small-value/batch-one controls and declared gaps.
- [X] T030 [US2] Implement finite comparison execution/reset in `src/oeis_learn/pipeline_validation/measurements.py`: fixed-complete frozen subset with separate 180-second diagnostic ceilings, full32 fixed-budget 60-second coverage, isolated arm caches and identical prewarm, one warmup/two exploratory pairs and fresh five-pair AB/BA/AB/BA/AB confirmation; keep service/reference caps unchanged.
- [ ] T031 [US2] Implement observation modes and overhead calibration in `src/oeis_learn/tracking/pipeline_profile.py` and `src/oeis_learn/tracking/foundation_metrics.py`: ordinary 1 Hz sampling, unavailable sensors, host/device boundary attribution, interval unions, separate synchronized mode, optional one-update/eight-token traces capped at 40 MiB each, and ordinary-primary fallback if overhead exceeds 5%.
- [X] T032 [US2] Extend `src/oeis_learn/fresh/pipeline_report.py` with fixed-work versus fixed-budget reports, full denominator/stratum coverage, actual completed work equivalence, five-pair ≥10% gains and ≤1.05 stratum ratios, median/range and explicit NO_ADOPTION/INCONCLUSIVE; native request time is not labelled pure WASM and diagnostic fractions are not whole-training forecasts.
- [X] T033 [US2] Run baseline/comparison fixture tests and register the bounded baseline stages in `src/oeis_learn/pipeline_validation/registry.py`; record verified reservations, task accounting and test results in `specs/009-pipeline-throughput/validation/phase-1-measurement-local.md`.
- [ ] T034 [US2] Collect/review target HIP baseline and resource calibration through the same all-ready command; freeze child comparison identities and mandatory workload gaps in `specs/009-pipeline-throughput/validation/phase-1-baseline.json` before candidate timing, keeping the completed parent run read-only and diagnostic clones nonresumable as research.

## Phase 2 — US2: Inference efficiency (P1)

**Goal**: Reduce cached-decoder bookkeeping at unchanged batch eight, preserving complete candidates. End-to-end adoption waits for common v2 policy and phase 4 confirmation.

**Independent test**: CPU and actual-HIP cached/full replay plus greedy/seeded free generation preserve row/RNG mapping and completed tokens/sources; measured isolated gains or no-adoption are source-bound.

- [X] T035 [P] [US2] Extend `tests/integration/test_bigint_decoding.py` for batch1/8 positions through 1,024, mixed/reordered/repeated/finished/all-inactive rows, EOS at capacity and weight/prompt/codec/dtype/mode invalidation; require CPU logits atol1e-5/rtol1e-4 and exact completed tokens/sources.
- [X] T036 [P] [US2] Extend actual-HIP parity/update regressions in `tests/integration/test_bigint_gpu.py` with atol2e-5/rtol2e-4, finite forward/backward parameter changes, supported positional/backend modes and seeded prompt identity; CPU fallback cannot pass these cases.
- [X] T037 [US2] Hoist per-step position/row/active-index/length/mask bookkeeping out of layer loops in `src/oeis_learn/decoder/inference_cache.py`, using bounded cache-owned scratch with correct invalidation; preserve mathematics, FP32, backend, inactive state and baseline batch8.
- [X] T038 [US2] Remove only redundant greedy log-probability transfers in `src/oeis_learn/decoder/sampler.py` where the existing trajectory contract returns no value; preserve CPU grammar token needs, finite-logit checks, stochastic draws and prompt-specific RNG streams.
- [X] T039 [US2] Expose explicit baseline/candidate decoder choices and source-bound receipts in `src/oeis_learn/decoder/bigint_generation.py` and `src/oeis_learn/pipeline_validation/adapters.py`; retain sequential rollback and label timing-sensitive v1 whole-pipeline observations diagnostic only.
- [X] T040 [US2] Run focused CPU decoder/generation/cache tests and register phase-2 checks in `configs/foundation/pipeline_validation.json`; document results and remaining HIP gates in `specs/009-pipeline-throughput/validation/phase-2-local.md`.
- [ ] T041 [US2] Collect/review actual-HIP parity and frozen isolated inference comparisons with all ready checks in one invocation; store source-bound results in `specs/009-pipeline-throughput/validation/phase-2-inference.json`, rejecting unexplained completed-candidate differences regardless of tolerant logits.
- [X] T042 [US2] Record retain-experimental/rollback/no-adoption and resource conclusions in `specs/009-pipeline-throughput/validation/phase-2-decision.md`; do not adopt a production optimization from cache microtimings or infer whole-pipeline gains before phase 4.

## Phase 3 — US3: Deterministic policy and bounded native slots (P1)

**Goal**: Remove timing from selection explicitly, then qualify 1/2/4 slots under one capacity/cache owner. This phase can be developed and measured on CPU with fixed candidate fixtures.

**Independent test**: Permuting completed result arrival and cost yields identical v2 selection/archive/training representatives; native fixed-candidate runs preserve independent admission under faults, saturation and global caps. v1 artifacts retain their meaning.

### Tests first

- [X] T043 [P] [US3] Add policy-only regression fixtures in `tests/unit/test_bigint_search_policy.py` for changed arrival/cost, identical-source provenance ties, deadline equality, absent selections, unrelated unfinished controls and v1/v2 seal dispatch; expected v2 ranking must not use hidden scores.
- [X] T044 [P] [US3] Add retention/representative fixtures in `tests/unit/test_synthesis_archive.py` for deterministic structure frontiers, assignment-bit complexity, cap-eight omissions, duplicate/conflicting evidence, cross-control merge and preserved v1 readers.
- [X] T045 [P] [US3] Add fake-clock/out-of-order/fault supervisor tests in `tests/unit/test_pipeline_native_pool.py` for aggregate pending/active/result bounds, FIFO affinity, unique generations, stale/duplicate replies, deadline charging, crashes, cancellation and healthy-after-failure.
- [X] T046 [P] [US3] Add shared-cache tests in `tests/unit/test_pipeline_cache.py` covering exclusive disk ownership, aggregate reservations/eviction, arm isolation/cold/prewarm, stale policy keys, partial shortlist rejection and current-deadline independent admission on hits.

### Implementation

- [X] T047 [US3] Implement `LogicalJob`, `JobExecution` and arm-stage accounting in `src/oeis_learn/fresh/pipeline_jobs.py`: `job_id = hash(contract, snapshot, mode, arm, prompt_id, attempt_ordinal, method, seed)`, legacy timestamp IDs as cross-references, exactly-once terminal publication and distinct operation/generation/scoring completeness; "Absolute clocks, process IDs and arrival order are not inputs".
  <details>
  <summary>Required LogicalJob constraints (verbatim from data-model.md)</summary>

  > LogicalJob — `pipeline-job/v1`
  >
  > Fields: contract hash; model snapshot hash; group/mode/arm; stable prompt ID and ordinal; attempt ordinal/method/seed; candidate original/final source hash when available; numerical/fit/selection/archive policy hashes; logical work reservation; expected outcomes/outputs; parent group/prompt identity.
  >
  > `job_id = hash(contract, snapshot, mode, arm, prompt_id, attempt_ordinal, method, seed)`. Absolute clocks, process IDs and arrival order are not inputs. Retain the old timestamp-derived attempt ID only as a legacy cross-reference. Duplicate logical IDs within one manifest are invalid. A retry cannot create a fresh attempt allowance by changing its presentation name.

  </details>

  <details>
  <summary>Required JobExecution constraints (verbatim from data-model.md)</summary>

  > JobExecution — `pipeline-job-execution/v1`
  >
  > Fields: logical job ID; invocation/stage ID; queue/slot/global generation; enqueue/dispatch/finish/owner-observed times; original absolute parent and service deadlines; reserved/consumed work; generation/raw/fit/reference dispositions; worker/resource failures; source/assignment/verification refs; cache provenance; publication ordinal; late/duplicate/stale flags.
  >
  > Normal state path: `PLANNED → RESERVED → GENERATING → NATIVE_QUEUED → NATIVE_RUNNING → REFERENCE → READY_TO_PUBLISH → COMMITTED`. Skip inapplicable stages explicitly, for example templates do not use neural generation. Any nonterminal state can end as `UNSTARTED`, `INCOMPLETE`, `CANCELLED`, `RESOURCE_EXHAUSTED`, `INVALID` or `FAILED` with a stage/reason. A conclusive unsupported or visible mismatch is a completed operation but not an accepted solution. A terminal disposition is committed exactly once; late duplicates append a rejection observation without changing it.
  >
  > Record distinct notions of scheduled work completion, generation completeness and scientific scoring coverage. A fully consumed token budget can have a recorded terminal operation while generation remains incomplete. That does not make it verified or automatically suitable for fixed-work equivalence; comparisons must verify matching actual work and classify deadline/unknown cases separately.

  </details>

- [X] T048 [US3] Implement visible-only/v2 selection/seals in `src/oeis_learn/fresh/synthesis_policy.py` and `src/oeis_learn/fresh/bigint_diagnostics.py`: canonical final-source bytes/hash/attempt ordinal/job ID ordering, timing-free semantic identity, selected exact assignment/provenance/verification and explicit v1/v2 validation; preserve raw/rescue/control evidence isolation.
  <details>
  <summary>Required ArmEvidence and SelectionSeal constraints (verbatim from data-model.md)</summary>

  > ArmEvidence and SelectionSeal
  >
  > Arm evidence records scheduled/started/completed attempts, required stage dispositions, all failure reasons, selected candidate ref or absence reason, visible fit and independently completed100-index production/reference scoring. Compact records exist for every prompt/arm before verbose examples.
  >
  > `visible-only/v2` semantic seal binds prompt/snapshot/policies, selected source, stable attempt provenance, original/transformed hashes, exact assignment and independent verification hash. Measured duration is in a separate evidence record so it cannot change the semantic identity. v1 seals remain validated by their original schema. No selection plus unknown/resource-exhausted generation/fit is incomplete; conclusive no-fit can be a complete negative. Scoring a sealed selection requires both production/reference to complete before its scoring deadline.

  </details>

- [X] T049 [US3] Implement structure-pareto/v2 in `src/oeis_learn/fresh/candidate_archive.py`: source bytes versus sum(max(1, abs(value).bit_length())), empty assignment zero, one representative per structure then deterministic frontier fill capped at eight; use the same stable key for training representatives and report omissions/conflicts.
- [X] T050 [US3] Replace timing-based duplicate/representative paths in `src/oeis_learn/fresh/bigint_worker.py`, `src/oeis_learn/fresh/bigint_diagnostics.py` and `src/oeis_learn/fresh/replay.py`; bind generation/selection/archive/scheduling/measurement hashes in `src/oeis_learn/fresh/bigint_config.py`, preserving old signed-256 and v1 readers without changing Rust shortlist ranking.
- [X] T051 [US3] Register and execute serial v1/v2 policy-only fixtures in `src/oeis_learn/pipeline_validation/adapters.py`; record every selected-source/provenance/archive/training-representative difference in `specs/009-pipeline-throughput/validation/phase-3-policy.json` before treating v2 as the common concurrency baseline.
- [X] T052 [US3] Factor native process/pipes/selector/stderr/RSS lifecycle into bounded slots in `src/oeis_learn/synthesis/worker_pool.py`, keeping one request per process, existing 32-candidate/4 MiB framing and globally unique worker generations across starts/restarts; do not change the wire schema.
- [X] T053 [US3] Implement one owning-role capacity ledger in `src/oeis_learn/synthesis/service_policy.py`: workers1/2/4, global pending8/32 MiB, active16 MiB, result8/32 MiB, reserve output before dispatch, prompt_ordinal modulo N FIFO affinity and no work stealing; account inflight/temporary/cache bytes rather than multiplying per-slot limits.
- [X] T054 [US3] Implement single-owner cache publication and completeness-aware `CacheEntry`/`Reservation` handling in `src/oeis_learn/synthesis/client.py` and `src/oeis_learn/synthesis/service_policy.py`; start with result4096/64 MiB, metadata32/64 MiB, disk256/512 MiB; "Only complete qualified payloads may be reusable complete fit results" and "A hit still undergoes current-deadline independent admission".
  <details>
  <summary>Required CacheEntry and Reservation constraints (verbatim from data-model.md)</summary>

  > CacheEntry and Reservation
  >
  > Fields: semantic key, namespace/arm, producer contract and job, payload hash/byte charge, exact completion/discovery status, resource and solver/reference policies, admitted assignment provenance, timestamps as observations, reservation/publication state. Only complete qualified payloads may be reusable complete fit results; partial discoveries are diagnostics. A hit still undergoes current-deadline independent admission. Cache publication is atomic under one owner and cannot exceed aggregate reserved bytes.

  </details>

- [X] T055 [US3] Implement bounded native/reference job dispatch in `src/oeis_learn/fresh/pipeline_jobs.py` and `src/oeis_learn/synthesis/client.py` using at most N host task threads and immutable returns; owner alone publishes evidence, no fork after Torch, queue residence consumes original parent/service deadlines and strict owner-observed-before-deadline eligibility applies.
- [X] T056 [US3] Implement aggregate watchdog/cancellation/recovery in `src/oeis_learn/synthesis/worker_pool.py` and `src/oeis_learn/fresh/pipeline_jobs.py`: 1 GiB/slot and 4 GiB aggregate native RSS, simultaneous process-group cancellation with five-second total cleanup, conservative unreported-work debits and no implicit retries; escalate nonquiescent reference tasks to the external role watchdog without indefinite executor shutdown.
- [X] T057 [US3] Enforce effective CPU/thread settings through `src/oeis_learn/pipeline_validation/resources.py` and `src/oeis_learn/pipeline_validation/adapters.py`: sixteen CPU equivalents maximum, Torch two intraop/one interop, native one thread where configurable, actual process/thread reporting and unchanged candidate ResourceProfile limits.
- [X] T058 [US3] Integrate typed native jobs into `src/oeis_learn/fresh/bigint_diagnostics.py` while retaining a sequential driver and separate raw/fit/reference/scoring dispositions; compact evidence must include every prompt/arm, including unknown, unsupported, unstarted, cancelled, invalid, timeout and independent verification failure.
- [ ] T059 [US3] Run CPU exactness/false-fit/cache/policy/containment regressions and frozen 1/2/4 candidate comparisons under both-arm v2; register phase-3 stages in `configs/foundation/pipeline_validation.json` and record local results in `specs/009-pipeline-throughput/validation/phase-3-local.md`.
- [ ] T060 [US3] Collect/review target CPU/native scaling and fault/cleanup receipts using one phase-3 or all-ready invocation; save `specs/009-pipeline-throughput/validation/phase-3-native.json`, including queue/occupancy/transport/reference costs and current engine identity; do not require HIP for fixed-candidate work.
- [X] T061 [US3] Apply the preregistered fastest-eligible-median/fewer-workers-within-5% rule and record selected concurrency or sequential retention in `specs/009-pipeline-throughput/validation/phase-3-decision.md`; disclose incomplete discovery sets and prohibit an automatic eight/sixteen-worker expansion.
- [ ] T062 [US3] Review all policy/scheduler/admission/cache counterexamples against `specs/009-pipeline-throughput/contracts/scheduling.md` and close the source-bound phase-3 acceptance matrix in `specs/009-pipeline-throughput/validation/task-evidence.json`; keep arithmetic changes, if any, behind reopened semantic qualification.

## Phase 4 — US3: GPU/native overlap and joint confirmation (P1)

**Goal**: Overlap independent candidate work within one frozen snapshot and resource envelope, then choose at most one qualified combined configuration.

**Independent test**: Simulated overlap preserves stable rows/jobs/selections under completion permutations, drains before updates and recovers without unaccounted work. Actual-HIP paired runs determine adoption; high utilization alone does not.

- [X] T063 [P] [US3] Add overlap/backpressure tests in `tests/unit/test_pipeline_overlap.py` for one active attempt per prompt/arm, fixed round-robin readiness, earliest deadlines, reserved decoder lookahead, saturated result queues, blocked-producer cancellation, mode/group changes and late duplicate results.
- [ ] T064 [P] [US3] Add complete-state barrier/resume tests in `tests/integration/test_bigint_workflow.py` for changed-policy rejection, pending-work cancellation, optimizer/RNG/data/curriculum/replay/selection preservation, conservative budget recovery and explicitly nonresumable weights-only clones.
- [X] T065 [P] [US3] Add telemetry/decision tests in `tests/unit/test_pipeline_overlap_metrics.py` for interval unions, shared-memory non-double-counting, unknown charging, observation overhead fallback, confirmation reset and prohibition of partial-work speedup claims.
- [X] T066 [US3] Implement one GPU-owner coordinator in `src/oeis_learn/fresh/pipeline_scheduler.py` and integrate it with `src/oeis_learn/fresh/bigint_diagnostics.py`; overlap only within the existing group/mode/frozen snapshot, fixed round-robin eligible prompts, one active attempt each, stable publication and maximum two reserved decoder batches subject to tighter native/result caps.
- [X] T067 [US3] Implement versioned batch1/8/16 and conditional batch32 policies in `src/oeis_learn/decoder/bigint_generation.py`, preserving prompt-specific RNG/cache/length/EOS mapping and earliest absolute deadlines; keep training batch8 unchanged and report changed candidate sets as policy evidence rather than transparent equivalence.
- [X] T068 [US3] Integrate `CheckpointBoundary` drain/cancel receipts with `src/oeis_learn/fresh/bigint_worker.py` and existing `src/oeis_learn/fresh/bigint_state.py`: "A resumable boundary has zero unaccounted active/reserved candidate work"; persist effective policy hashes, stable frontier and cumulative debits with complete learning state before updates/save/selection/mode/group transitions.
  <details>
  <summary>Required CheckpointBoundary constraints (verbatim from data-model.md)</summary>

  > CheckpointBoundary
  >
  > Use the existing `bigint_state.CheckpointStore` and complete learning state. Add effective scheduling/selection/archive/measurement contract hashes, stable completed logical frontier, cumulative reservations/debits and a drained/cancelled synthesis-block receipt. A resumable boundary has zero unaccounted active/reserved candidate work. Preserve optimizer, RNG, data order, curriculum, replay and selection. Reject a changed-policy resume; weights-only diagnostic snapshots are explicitly nonresumable.

  </details>

- [ ] T069 [US3] Add bounded overlap/resource observations to `src/oeis_learn/tracking/pipeline_profile.py` and `src/oeis_learn/fresh/pipeline_report.py`, keeping ordinary timing free of per-stage GPU synchronization, bounded device events at drain boundaries, source-bound measurement mode and CPU scorer startup after GPU-role drain.
- [X] T070 [US3] Encode the finite phase-4 candidate sequence in `configs/foundation/pipeline_validation.json`: overlap lookahead1 then2, batch8 versus16 plus batch1 regression, same-envelope historical16/capacity64/workstation90 pressure trials and separate result32768/512 MiB + disk2048/4 GiB cache candidate; allow batch32 only after batch16 improvement and calibrated ≥10% unreserved envelope, with every branch reserved prospectively.
- [X] T071 [US3] Extend `src/oeis_learn/pipeline_validation/measurements.py` to select/freeze at most one combined candidate before fresh five-pair cold/warm confirmation against serial v2, accepting explicitly declared baseline component substitutes; keep fixed-budget full32 coverage separate and preserve all failed/inconclusive exploratory results.
- [X] T072 [US3] Run overlap/barrier/metrics CPU fixtures and register phase-4 stage dependencies in `src/oeis_learn/pipeline_validation/registry.py`; record local results and real-device pending gates in `specs/009-pipeline-throughput/validation/phase-4-local.md`.
- [ ] T073 [US3] Collect/review actual-HIP joint parity, aggregate stress/cancellation, representative update/validation and complete pipeline comparisons via one all-ready command; save immutable indexes in `specs/009-pipeline-throughput/validation/phase-4-joint.json`, including every failed/unstarted job and excluded stress case.
- [X] T074 [US3] Publish adoption/NO_ADOPTION/INCONCLUSIVE in `specs/009-pipeline-throughput/validation/phase-4-decision.md` using all five ≥10% gains, every mandatory stratum ≤5% regression, exact work/admission parity, bounded resources and no unexplained fixed-budget coverage loss; forbid extra favorable sampling and report subset scope explicitly.
- [ ] T075 [US3] Set the supported default only after T074 qualifies it and retain serial rollback in `configs/foundation/pipeline_validation.json` and `src/oeis_learn/fresh/bigint_config.py`; if adoption fails, retain the baseline and record investigation completion without claiming a speedup.
- [ ] T076 [US3] Review implementation/target evidence and update `specs/009-pipeline-throughput/quickstart.md` and `specs/009-pipeline-throughput/validation/task-evidence.json` with the chosen envelope, rollback and one consolidated next validation command; no separate profiler/repackaging handoff.

## Phase 5 — US4: Operations, qualification and cost admission (P2)

**Goal**: Close current-source engineering gates and exact-scope cost reports while preserving history and independent learning/research decisions.

**Independent test**: Gate fixtures distinguish compatible/stale/missing receipts, deferred 008:T093, negative 008:T096 and complete versus incomplete forecasts. Current-source engine/device bundles qualify only their demonstrated closures.

- [X] T077 [P] [US4] Add gate/carryover/cost rejection tests in `tests/unit/test_pipeline_qualification.py` for all thirteen 008 carryovers, source-sensitive requalification, T093 deferral, unchanged T096 7,730/14,400-second negative ledger, missing forecast components and diverse32-versus-full-transfer scope mismatch.
- [ ] T078 [P] [US4] Extend `tests/unit/test_container_validation.py` and `tests/unit/test_foundation_roles.py` with supported-command inventory, retired-command disposition, clean installed readers/no retired solver reachability, arbitrary-UID/rootless export, isolated CPU scorer and normal/interrupted complete-state operations coverage.
- [X] T079 [US4] Implement `QualificationReport` in `src/oeis_learn/pipeline_validation/qualification.py`, binding current gates to dependency-aware evidence, separating engineering/learning/research/long-run dispositions and recording every carryover closure; "A no-go is a completed report, not a promotion".
  <details>
  <summary>Required QualificationReport constraints (verbatim from data-model.md)</summary>

  > QualificationReport
  >
  > Map each gate and carryover task to current evidence, compatible historical evidence, missing prerequisites, disposition and closure condition. Record T093 as deferred, T096 as historical negative and separately named new attempts if ever admitted. Separate engineering readiness from learning success, research completion and permission for a longer run. A no-go is a completed report, not a promotion.

  </details>

- [X] T080 [US4] Complete reusable operations/retirement/resume/export adapters in `src/oeis_learn/pipeline_validation/adapters.py` and `src/oeis_learn/container_validation.py`, covering each supported command or explicit retirement with bounded fixtures; preserve historical numerical/read-only compatibility without acquiring/converting T093 weights.
- [X] T081 [US4] Implement exact-scope forecasts in `src/oeis_learn/fresh/pipeline_report.py` and `src/oeis_learn/fresh/cost_policy.py` for diverse32 and full128+128 transfer separately, including all periodic/end controls, updates/validation/native/reference/startup/checkpoint/export/cleanup and conservative variability; unknown components block the inherited ≤80%-of-budget admission gate.
- [X] T082 [US4] Integrate current-source qualification and cost stages in `src/oeis_learn/pipeline_validation/registry.py`, packaging/source checks in `Dockerfile` and existing CI only where needed; preserve early cached dependency/native/guest layers and register affected checks without adding manual operator commands.
- [X] T083 [US4] Run gate/forecast/operations fixture suites and record results in `specs/009-pipeline-throughput/validation/phase-5-local.md`; verify engineering execution cannot train/resume T096, expose final data or treat a positive cost estimate as a study decision.
- [ ] T084 [US4] Collect/review one combined current-source arithmetic/fitting/containment/HIP/operations run with Docker and Podman coverage, importing only compatible receipts where necessary; retain `specs/009-pipeline-throughput/validation/phase-5-qualification.json` and explicit missing-capability blockers rather than treating one engine as both.
- [ ] T085 [US4] Publish source-bound engineering/cost decisions and the retained small-value 13.29–13.40x gap/>5x review-trigger disposition in `specs/009-pipeline-throughput/validation/phase-5-report.md`; review 008:T089–092 dependencies, close only evidenced T094/T097/T098 portions and retain T095's old scoped decision and T096's original negative result.
- [ ] T086 [US4] Reconcile `specs/009-pipeline-throughput/carryover.md`, `specs/009-pipeline-throughput/validation/task-evidence.json` and `specs/009-pipeline-throughput/quickstart.md` against phase-5 evidence; give one next ready validation/study-preparation handoff with measured expected cost, finite limits and archive path while keeping unadmitted research pending.

## Phase 6 — US5: Admitted controlled research (P3)

**Goal**: Reuse the qualified application through one command per already admitted study batch; stop at actual scientific decisions and final-exposure authority.

**Independent test**: Fixture-only manifests reject missing/stale/over-budget authority before training or private mounts, preserve every arm and budget across interruption, and never infer final or long-run permission. Actual study tasks below remain gated separately from code completion.

### Tests and orchestration code — do not run research here

- [X] T087 [P] [US5] Add study-kind/authority/provenance/stop tests in `tests/unit/test_pipeline_studies.py`, including the noncircular new-learnability gate, representation's prior-learning requirement, unknown kinds, missing amounts, seed substitution, unreviewed assistance and independent study decisions within a combined batch.
- [X] T088 [P] [US5] Add final exposure/budget fixtures in `tests/integration/test_foundation_finalization.py` for explicit final-plan authority, exposure before prompt access, interrupted/unstarted arms, same-plan remaining-budget resume and changed-plan/freshness rejection.
- [X] T089 [US5] Implement `StudyDecision` parsing, typed gates and immutable lifecycle in `src/oeis_learn/pipeline_validation/studies.py` with finite per-arm/shared work/wall/disk limits and exact-scope receipts; "Missing gates yield BLOCKED, not a successful scientific result" and "A manifest may document an explicit prospective deferral; it cannot mark the unexecuted study complete".
  <details>
  <summary>Required StudyDecision constraints (verbatim from data-model.md)</summary>

  > StudyDecision — `pipeline-study/v1`
  >
  > Fields: kind (`learnability`, `representation`, `curriculum`, `replay`, `assistance`, `encoder`, `final`); prospective decision ID; frozen hypotheses/arm roster/checkpoints/configs/seeds; strict/assisted lineage and reviewed inputs; prior decision hashes; scope-specific cost receipt; mandatory qualification identities; fixed denominators and20/100 horizons; shared/per-arm budgets; selection/stop/promotion rules; authority and final-plan reference when applicable. Invalid/unreviewed authority fails before mounting data or starting training.
  >
  > States: `DRAFT → FROZEN → ADMITTED → RUNNING → COMPLETE | NEGATIVE | INCONCLUSIVE | INTERRUPTED`. Missing gates yield `BLOCKED`, not a successful scientific result. A manifest may document an explicit prospective deferral; it cannot mark the unexecuted study complete. Final cohort retains existing `sealed → authorized → exposed` lifecycle, with exposure recorded before prompt access and no transition back to sealed.

  </details>

- [X] T090 [US5] Wire phase6/study-manifest/final-plan routing in `scripts/foundation_local.py` and internal study adapters in `src/oeis_learn/pipeline_validation/adapters.py`, reusing existing learning/evaluation paths, one archive and all mandatory subordinate controls; reject engineering-manifest mixing and stop before undecided arms, final access or unauthorized long training.
- [X] T091 [US5] Implement matched-study reports in `src/oeis_learn/pipeline_validation/study_reports.py` using existing `src/oeis_learn/fresh/comparison.py`: full128+128 denominators, raw/rescue/equal-budget/correct/shuffled/absent/template, visible20/continuation100, seed spread/predeclared structure-family bootstrap (2,000 fixed-seed resamples; descriptive 95% intervals), stopped/unstarted costs and inherited one-endpoint transfer gate; no hidden result selects checkpoints/candidates.
- [X] T092 [US5] Run fixture-only research/finalization tests, register phase6 availability and document concrete manifest-driven batch examples in `specs/009-pipeline-throughput/quickstart.md`; record `specs/009-pipeline-throughput/validation/phase-6-local.md` without marking any actual study executed.

### Prospective decisions and gated execution — separate from code/local checks

- [ ] T093 [US5] Prepare a separately identified new-learnability decision, only if proposed after phase5, in `configs/foundation/pipeline_studies/learnability.json`; require explicit authority/current runtime/device/resource/provenance and exact-scope cost without demanding an earlier learning PASS, retain ≥32 prompts/four structures/16 non-affine/≤8 offset-only, 31/32 correct and32/32 complete, and preserve historical **008:T096** unchanged.
- [ ] T094 [US5] Execute/review only the admitted new-learnability batch through one study command and record `specs/009-pipeline-throughput/validation/phase-6-learnability.json`; report every raw/rescue/control outcome and budget, stopping dependent learning on negative/inconclusive results without automatic retries.
- [ ] T095 [US5] Prepare matched representation contracts and a fresh sealed final-cohort reference in `configs/foundation/pipeline_studies/representation.json` using existing `src/oeis_learn/fresh/input_sealing.py`; freeze generic data/initialization/vocabulary differences, exposed diagnostics, controls/cadence and paired seeds20261016/17/18 (or prospective documented replacement), completing **008:T099** preparation without final access.
- [ ] T096 [US5] After explicit representation admission, execute all six literal/parameterized arms and full transfer controls with one study command; record `specs/009-pipeline-throughput/validation/phase-6-representation.json` for **008:T100**, preserving negative/interrupted/unstarted arms, all costs and strict lineage without seed substitution.
- [ ] T097 [US5] Review **008:T101** in `specs/009-pipeline-throughput/validation/phase-6-transfer-decision.md`: ≥95% coverage per split/seed, every admission independently verified, structural advantage ≥8/128 against each shuffled/absent/template in ≥2/3 seeds and mean≥6.25pp, parameter loss≤4/128 mean, frozen family uncertainty/seed spread and full strata; explicitly stop or promote once at the registered endpoint.
- [ ] T098 [US5] Prepare and, only if admitted after a supported recipe decision, execute the curriculum-only batch in `configs/foundation/pipeline_studies/curriculum.json`; retain earlier-stage practice and publish `specs/009-pipeline-throughput/validation/phase-6-curriculum.json` with all arms/costs before deciding replay (**008:T102**, first intervention).
- [ ] T099 [US5] Prepare and, only after its separate decision/admission, execute replay in `configs/foundation/pipeline_studies/replay.json`; preserve training-only independent admission, complete checkpoint/cumulative budgets and publish `specs/009-pipeline-throughput/validation/phase-6-replay.json` (**008:T102**, second intervention).
- [ ] T100 [US5] Resolve assistance inputs/decision in `configs/foundation/pipeline_studies/assistance.json`: reviewed demonstrations, compatible generic foundation, overlap/ancestry/descendant audit, matched generic-only control and shared/branch cost; missing inputs produce BLOCKED or an explicit prospective deferral before final planning, never invented examples or a completed **008:T103** comparison.
- [ ] T101 [US5] If assistance is admitted, execute/review its full matched batch with one command and retain `specs/009-pipeline-throughput/validation/phase-6-assistance.json`; preserve strict/assisted lineage and all arms, or leave this execution task unchecked with the explicit T100 deferral.
- [ ] T102 [US5] Freeze/admit and execute the four post-training encoder component sensitivity ablations in `configs/foundation/pipeline_studies/encoder-sensitivity.json`; publish `specs/009-pipeline-throughput/validation/phase-6-encoder-sensitivity.json` with active/total parameters, throughput, size/length/complexity strata and interactions, explicitly not retrained utility (**008:T104**).
- [ ] T103 [US5] Freeze only prospectively selected matched encoder retrainings in `configs/foundation/pipeline_studies/encoder-retraining.json`, obtain their separate cost/decision admission, execute their combined batch and publish `specs/009-pipeline-throughput/validation/phase-6-encoder-retraining.json`; no automatic architecture sweep or unregistered combinations (**008:T104**).
- [ ] T104 [US5] Prepare immutable final arm/checkpoint/cohort/shared-and-per-arm-budget plan in `configs/foundation/pipeline_studies/final.json` only after all recipe/assistance/encoder decisions including deferrals are frozen; require the existing authorized final-plan reference and explicit invocation before access (**008:T105**).
- [ ] T105 [US5] Execute/review the explicitly authorized final batch and store `specs/009-pipeline-throughput/validation/phase-6-final.json`; record exposure before the first prompt, preserve all failed/unstarted arms and consumed budgets, and reject changed-plan retuning or freshness reset (**008:T105**).
- [ ] T106 [US5] Publish discovery review in `specs/009-pipeline-throughput/validation/phase-6-discovery.md`, linking alternatives/omissions and separating finite match, structural alternative, conjecture, bounded verification, proof and novelty; a limited archive audit while final evaluation is blocked does not close **008:T106**.
- [ ] T107 [US5] Publish evidence-backed longer-run go/no-go in `specs/009-pipeline-throughput/validation/phase-6-readiness.md`, including qualified settings, measured cost and all pending gates; a negative report may close **008:T107** reporting but never grants learning promotion or automatically launches12-hour-plus training.
- [X] T108 [US5] Reconcile executed versus deferred/blocked research and immutable evidence links in `specs/009-pipeline-throughput/carryover.md` and `specs/009-pipeline-throughput/validation/task-evidence.json`; close only actually evidenced carryovers, keeping engineering, learning, research completion and longer-run admission separate.

## Cross-cutting closeout

- [X] T109 [P] Review documentation/command examples in `specs/009-pipeline-throughput/quickstart.md` and `README.md` against implemented CLI behavior, preserving one all-ready engineering command and one command per admitted research batch; explain only genuine capability/decision boundaries that require another invocation.
- [ ] T110 Perform final affected-source contract/packaging checks through the consolidated registry in `src/oeis_learn/pipeline_validation/registry.py`; update `specs/009-pipeline-throughput/validation/task-evidence.json` for new changes only, avoid redundant hardware reruns when closures remain compatible, and do not wait for PR CI.
- [X] T111 Publish final implementation handoff in `specs/009-pipeline-throughput/validation/implementation-closeout.md` with code/test status, remaining target-machine and research tasks, the exact next copy-pastable command, measured expected range/hard limits and resulting archive; never describe pending gates as completed.

## Dependencies and execution order

| Boundary | Dependencies | What can proceed independently |
| --- | --- | --- |
| Setup/foundation | T001 then T002; identity tests before helpers; resource tests before resolver; closures after identity | T003 and T004 after T001; T006 can run with T004; T007 requires T005/T006 |
| Phase1 US1 MVP | T001–T008; tests T009–T011 before corresponding implementations; T012→T013/T014→T015; adapters/CLI/engines follow their contracts; archive/reuse follow receipts | Test files T009/T010/T011; later native CPU design need not wait for target HIP |
| Phase1 US2 baseline | Foundations and US1 receipt/runner interfaces; T027 before T028–T032; T033 before T034 | CPU comparison fixtures while real-engine/HIP evidence is pending |
| Phase2 US2 inference | Phase1 frozen comparison contracts; T035/T036 before edits; T040 before T041; calibration T034 before candidate timings | Phase3 CPU work after shared foundations, with separate files; never run competing benchmarks |
| Phase3 US3 native | Foundations and fixture/admission adapters; T043–T046 before their implementations; T047→T048→T049/T050→T051; slot/cache/dispatch integration T052–T058 before T059 | CPU-native work needs no HIP/weights; T060 requires qualified common v2 policy and phase1 CPU calibration |
| Phase4 US3 overlap | Common v2 policy, bounded native lifecycle, qualified selected components or explicitly declared baseline substitutes; T063–T065 before T066–T071; T072 before target T073 | Code/fixture work may proceed with pending HIP; adoption T074/T075 remains blocked until required target evidence |
| Phase5 US4 operations | Current applicable source/engine/device/operations receipts; phase4 decision for selected optimized defaults, or explicitly retained baseline | T077–T083 can develop before performance adoption; no-adoption does not block an honest qualification report |
| Phase6 US5 code | Stable host identities/registry and existing research contracts; T087/T088 before T089–T091 | Fixture-only code does not require a successful learning attempt |
| Phase6 actual studies | Phase5 relevant qualification/resource/cost + each explicit study decision; T093→T094; T095 before revised screens; T094 learning PASS before T096; T096→T097; supported recipe→T098→separate replay decision T099; assistance/encoder decisions before T104→T105→T106/T107 | Matched preparation can precede learning success but cannot execute studies. Already decided independent studies may share one admitted batch. Missing optional assistance can be prospectively deferred; mandatory gates cannot |
| Closeout | T109 after CLI is implemented; T110 after last relevant code changes; T111 records all remaining dispositions | Engineering closeout does not wait indefinitely for blocked research; unchecked execution tasks remain visible |

**Delivery ranges**: Phase 1 T001–T034 (including shared foundations); phase 2 T035–T042; phase 3 T043–T062; phase 4 T063–T076; phase 5 T077–T086; phase 6 T087–T108; cross-cutting closeout T109–T111. Use these ranges for future phase implementation requests.

**Integration ordering**: Read the relevant quoted data-model constraints and full contract before each implementation. Tasks touching the same file execute serially even across phases. Evidence reviews follow collection; collection never implies scientific PASS. Changes after a receipt invalidate affected descendants via the closure rules rather than silently reusing stale evidence.

### Parallel examples per story

These are development opportunities, not instructions to launch agents or concurrent hardware measurements.

| Story | Independent task pair/group | Common prerequisite |
| --- | --- | --- |
| US1 | T009 CLI/DAG tests; T010 reuse tests; T011 bundle tests | Shared foundation interfaces |
| US2 | T035 CPU cache/token tests; T036 HIP test implementation | Frozen inference contract; hardware execution remains serialized |
| US3 | T043 policy tests; T044 archive tests; T045 pool tests; T046 cache tests | Frozen v2/scheduling contract; no production edits in these parallel tasks |
| US3 overlap | T063 scheduler fixtures; T064 resume fixtures; T065 telemetry fixtures | Completed phase3 interfaces |
| US4 | T077 gate/cost fixtures; T078 engine/operations fixtures | Receipt/resource interfaces |
| US5 | T087 study admission tests; T088 final exposure tests | Existing research/final lifecycle and shared schemas |

## Requirement and carryover coverage

| Requirements / criteria | Task coverage |
| --- | --- |
| FR-001–005; SC-001/007 | T004–T026, T078/T080/T082/T084, T109–T111 |
| FR-006–011; SC-004/005 | T027–T034, T040–T042, T059–T061, T069–T075, T081/T085 |
| FR-012–013; SC-003/007 | T035–T042, T063/T067/T073 |
| FR-014–018; SC-003/006 | T006–T008, T043–T062, T063/T066/T070/T073 |
| FR-019–021; SC-002/003/006 | T043/T047–T050/T054–T058, T063/T064/T066/T068/T073/T080 |
| FR-022–026; SC-008/009/010 | T003, T077–T086, T089/T093/T094/T107/T108 |
| FR-027–030; SC-009/010 | T087–T108 |

| Spec008 carryover | Spec009 owner / closure tasks |
| --- | --- |
| 008:T093 | User-deferred; T003/T079/T085/T108 preserve disposition, no execution substitute |
| 008:T094 | T027–T042, T051/T059–T062, T069–T075, T081/T085; whole-pipeline evidence plus retained arithmetic gap review |
| 008:T097 | T009–T026, T064/T068, T078/T080/T082–T086 |
| 008:T098 | T008/T019/T079/T084–T086; immutable current-source qualification |
| 008:T099 | T095, reviewed contracts/cohort; sealing is not final authority |
| 008:T100 / T101 | T096 / T097 respectively; full matched execution and decision |
| 008:T102 | T098 then T099, separate intervention decisions |
| 008:T103 | T100 decision, T101 conditional actual comparison |
| 008:T104 | T102 sensitivity, T103 selected matched retraining |
| 008:T105 | T104 immutable plan, T105 explicit final execution |
| 008:T106 / T107 | T106 discovery / T107 readiness report |
| Checked 008:T089–092 | T018/T034/T041/T059/T073/T084 requalify affected current closures; retain historical receipts |
| Checked 008:T095 / T096 | T081/T085 new scoped cost; T077/T079/T085 preserve negative historical learning. T093/T094 are separately named future attempt tasks, not rewrites |

## Implementation strategy and operator checkpoints

**MVP**: Shared foundations plus phase1 US1 T009–T026: useful stdlib host orchestration, shared builds, explicit capability gaps and one failure-safe archive before optimization. Phase1 delivery additionally includes T027–T034's frozen measurement baseline. Do not claim phase1 target calibration complete from fake-engine tests.

**Incremental delivery**: Commit each coherent test/implementation group, push early and add reviewable commits. Complete nonblocked code and local verification while hardware runs are delegated to the operator. Phase2 and phase3 development can proceed independently, then common v2/qualified components feed phase4. Phase5 may close engineering with a no-adoption or blocked-research report. Phase6 has real decision boundaries; manifest preparation is not execution authority.

**Single-command handoff**: At each implementation checkpoint, prefer the [quickstart all-ready command](quickstart.md#default-handoff-one-combined-command), including all ready CPU/native/HIP/engine checks, bounded measurements and one archive. Replace placeholder run/input paths with actual compatible paths when known. Phase-specific selection is for targeted reruns, not mandatory operator sequencing. The engineering ceiling remains 86,400 seconds, with 84,000 seconds maximum declared reservations; actual expected time comes from calibration, never from that ceiling. No separate build/profiler/packaging commands should be necessary. Phase3 fixed-candidate validation can run without HIP; GPU claims require actual HIP. Research batches use the same entry point with admitted study manifests and their own finite budgets.

**Stop conditions**: Unavailable device/engine blocks only affected gates; failures block descendants and independent safe checks continue. Global interruption/resource/audit failures stop admission while preserving evidence. Negative optimization stops expansion; negative research stops unsupported promotion. Do not wait for PR CI after pushing; report the commit and remaining one-command hardware handoff.

## Supplemental measurement refinement — pipeline-replay-viability/v1

These items track the accepted narrowed follow-up separately from the original 111 tasks; they do not close missing original performance/portability/research evidence. Contract: [replay.md](contracts/replay.md).

- [X] 009:R001 Register a separate closed replay manifest, scope and static graph; preserve original validator/research authority and all existing ceilings.
- [X] 009:R002 Reuse visible parent/weight preparation and freeze CPU full-prefix/mixed-cache reference tensors before GPU timing, without reading hidden truth.
- [X] 009:R003 Execute one cold, one excluded warmup and three bounded baseline roles, with immutable reference/resource checks, complete missing-slot reporting and no candidate/research admission.
- [X] 009:R004 Add stdlib host contract/orchestration/privacy regressions and pinned CPU/actual-HIP role fixtures; verify the available host CLI and archive locally. Authored runtime fixtures are not claimed executed locally; their target execution is now reviewed under R005.
- [X] 009:R005 Execute/review the single combined replay command on the target; retain actual CPU/HIP fixture and five-trial results, including non-PASS/partial records. No automatic retry or promotion.

## Supplemental phase-2 investigation — cache-bookkeeping replay comparison v1

This separately registered component scope follows accepted replay viability; see [contract](contracts/replay-comparison.md) and [handoff](validation/replay-comparison-implementation.md). Original full-generation, whole-pipeline and research gates remain unchanged. Current original-task counts are77 complete/19 runtime pending/15 research blocked; T110 reopens for the new affected source while its prior exact-source closure remains historical.

- [X] 009:C001 Freeze the separate closed manifest, static graph and exploratory AB/BA contract before candidate timing; preserve existing manifests and denied authorities.
- [X] 009:C002 Bind current source/reference/weights, complete baselines, resources/threads, immutable child/slot identities and finite setup/startup allowances; execute one candidate warmup and two pairs with independent missing-slot reporting.
- [X] 009:C003 Run host contract/orchestration/privacy/partial-work checks and the actual public stdlib CLI; author/register real CPU/HIP comparison role fixtures and record the verified non-admitting local archive.
- [X] 009:C004 Execute/review the one combined target command, including current pinned CPU/actual-HIP roles, all five comparison slots, source/images/resources and archive; preserve failures without adaptive retries or adoption.

### Comparison qualification fixture follow-up

The [first comparison archive review](validation/replay-comparison-operator-20261024T190425Z.md) retains a mandatory HIP fixture failure before measured work:621 CPU tests and15 other HIP tests passed.009:C004 and T110 remain pending. The fixture now establishes real interop=1 in a fresh bounded subprocess and checks actual2/1 observations without weakening the production gate.106 available host tests pass. No schedule, budget, workload or scientific threshold changed, and no incomplete pair became a performance result.

### Accepted component evidence and current priority

The [second comparison archive](validation/replay-comparison-operator-20261024T193940Z.md) passes at exact producer ac41e4d:622 CPU/16 actual-HIP tests and all10 replay roles complete. Two exploratory component reductions are10.54/11.28%; no production adoption.009:C004 and affected-source T110 close for that source; original counts78 complete/18 runtime pending/15 research held or blocked. The user now prioritizes CPU/GPU/memory throughput and capacity. Hold learning-performance/curriculum investigations; a later separately admitted multi-day training run will inform organic/stochastic versus curriculum-driven learning after hardware qualification.

## Supplemental hardware screen — equal-work batch8/16 replay v1

The user's hardware-first instruction follows the accepted bookkeeping comparison. [Contract](contracts/replay-batching.md) and [handoff](validation/replay-batching-implementation.md) register a separate equal-work screen, preserving all prior evidence and training holds. Original counts77 complete/19 runtime pending/15 research held or blocked reflect reopened current-sourceT110, not revocation of its accepted older-source closure.

- [X] 009:B001 Register a separate closed batch8/16 manifest, fixed32 visible rows/32768 supplied tokens/160 row boundaries and no learning/default authority.
- [X] 009:B002 Reuse isolated source-bound reference/role/ledger/telemetry machinery; freeze complete baseline calibration, actual resource/thread envelope, candidate capacity/headroom and AB/BA schedule before timing.
- [X] 009:B003 Run complete host orchestration/authority/privacy/capacity/partial-work tests and the public stdlib CLI; author/register actual CPU/HIP32-row oracle and both batch-role fixtures and record verified non-admitting local evidence.
- [X] 009:B004 Collect/review one combined target batching run, including current CPU/HIP fixtures, all fixed-work roles, memory/utilization observations and source-bound archive. Preserve missing/failed arms without retries or adoption.

### Accepted batch8/16 screen

[Target evidence](validation/replay-batching-operator-20261024T203537Z.md) closes009:B004 and affected-sourceT110 at d74d30e:629 CPU/17 HIP tests, all rows/roles complete,48.98% median component interval reduction. Counts78 complete/18 runtime pending/15 research held. Further hardware experiments should be combined under finite prospective contracts to minimize operator handoffs; isolated timing measurements stay serial.

## Supplemental combined hardware experiments — finite capacity ladder v1

Per the user's request to minimize handoffs, [contract](contracts/hardware-ladder.md) and [handoff](validation/hardware-ladder-implementation.md) combine ordinary/telemetry overhead and conditional32/64/128 comparisons against16. Inputs are32 visible rows cyclically mapped into128 physical cache rows, clearly labelled as capacity diagnostics. Original manifest identities/results and learning hold remain. Counts77 complete/19 runtime pending/15 research held reflect reopening current-sourceT110.

- [X] 009:H001 Freeze the separate manifest, prior batch16 evidence basis,128-row/131072-token/640-boundary work, mode/progression/selection rules, finite candidate roster and unchanged phase/global/resource bounds.
- [X] 009:H002 Implement shared reference/preparation, ordinary/telemetry mode calibration, complete baseline-only duration/capacity admission and per-branch five-role ledgers with strict source/mode/weight/child custody, no retries and explicit stopped/unstarted work.
- [X] 009:H003 Run host authority/DAG/negative/failure/receipt/global-stop checks and actual public stdlib export; author/register real CPU/HIP128-row/all-size/ordinary-mode oracle fixtures; retain local evidence and one combined command.
- [X] 009:H004 Collect/review the combined target archive, including current CPU/HIP kernels, frozen mode, every admitted/stopped/blocked branch, CPU/GPU/memory observations and package/source/audit/cleanup identities. No production or learning admission inferred.

### Accepted combined hardware ladder

[Target review](validation/hardware-ladder-operator-20261024T220811Z.md) closes009:H004 and affected-sourceT110 at adda1b6:639 CPU/18 HIP tests and all22 GPU roles complete;0.62% overhead selects aggregate telemetry. Batch128 is the experimental finite-roster winner,61.13% median interval reduction versus16, about90.5% sampled GPU busy and2.84GiB role memory. Original counts78 complete/18 runtime pending/15 research held. No duplicate run or production/learning admission. Native CPU scheduling, deliberate overlap and useful full-pipeline/sustained operations remain.

## Supplemental native CPU capacity screen — slot waves v1

After the [accepted GPU ladder](validation/hardware-ladder-operator-20261024T220811Z.md), the next hardware investigation isolates native CPU admission/scaling. [Contract](contracts/native-capacity.md) and [handoff](validation/native-capacity-implementation.md) freeze a separate CPU-only scope, preserving original queued/default behavior and all accepted source-bound evidence. Counts77 complete/19 runtime pending/15 research held reflect reopening affected-sourceT110 for new native code. Deliberate overlap/full-pipeline/sustained work remains later; no learning investigation.

- [X] 009:N001 Freeze the separate closed CPU-only manifest/registry, full18-row corpus, slot-wave admission with immutable four-second request/parent clocks, independent1/2/4 hypotheses and ordinary/observed roles under unchanged limits.
- [X] 009:N002 Implement fresh clients/caches, full baseline-only calibration and source-bound child, durable18-slot/job plans/events/terminal debits, strict complete semantic/worker/clock parity, no favorable partial ratios and global containment stops.
- [X] 009:N003 Run134 host tests and actual public stdlib export with all18 undispatched slots explicit; author/register three real packaged-native1/2/4 slot/assignment/clock fixtures, retain source-bound local evidence and one combined command.
- [X] 009:N004 Collect/review the combined CPU target archive, current qualification image/fixtures, complete/inconclusive/blocked trials, actual CPU/memory/native service observations, all immutable clocks/debits and cleanup. No GPU ladder repetition or production/learning admission inferred.

### Native qualification repair after first target collection

[Reviewed archive](validation/native-capacity-operator-20261025T032351Z.md) matches source and retains652 CPU passes/one serial-worker metrics assertion failure; all18 performance slots remain UNSTARTED. The fixture now checks the retained serial versus2/4 pooled metric shapes without weakening actual slot activity/assignment/clock/debit requirements. Already-completed CI also exposed a wrong serializer in the007 disposition binding, now refreshed with its actual canonical functions and all68 dispositions preserved. Two targeted host regressions reproduce both failures before repair;136 host tests and verified export now pass. Runtime Python bytes, workload, bounds, schedule and thresholds are unchanged.009:N004/T110 remain pending corrected-source CPU collection; original counts77 complete/19 runtime pending/15 research held. No HIP replay or learning admission.

### Accepted full18-row native screen

[Corrected target review](validation/native-capacity-operator-20261025T040709Z.md) closes009:N004/T110 at176ecc9:655 CPU tests, all18 native slots/324 result rows and strict306 wave debits complete. Four workers win experimentally with68.63–70.71% reductions; no defaults or learning admission. Counts78 complete/18 runtime pending/15 research held. No duplicate native/GPU screen is requested; deliberate co-scheduling, useful pipeline throughput and stability remain hardware work.

## Supplemental GPU/native co-scheduling and repeatability v1

The [accepted native screen](validation/native-capacity-operator-20261025T040709Z.md) and earlier GPU128 screen support this separate [contract](contracts/coschedule.md)/[handoff](validation/coschedule-implementation.md). One container shares the CPU/memory envelope, one fresh GPU child runs at a time, and native4 deliberately overlaps only in its named arm. Fixed eight-cycle/144-native-row work retains complete evidence inside existing artifact bounds. Original policies, scientific requirements and learning hold remain.

- [X] 009:O001 Register the separate closed manifest/graph, accepted evidence basis, GPU128/native144 equal work, fixed20-slot serial/overlap/repeat roster, immutable clocks and unchanged global/phase/resource/artifact bounds before timing.
- [X] 009:O002 Implement CPU/native references, source-bound baseline-only child and schedule decision, one quota/GPU owner, fresh supervised GPU children, unique native cycle jobs, strict full-work parity/clock/debit/resource gates and bounded shared cancellation/drain with complete partial/unstarted evidence.
- [X] 009:O003 Run154 host tests and actual public stdlib export with all20 unstarted slots explicit; author/register packaged-native cycles and real HIP/native serial/overlap fixtures; record exact source/manifest/archive and one target command without host dependencies.
- [X] 009:O004 Collect/review current CPU/native/HIP fixtures and every admitted/failed/blocked ordinary/repeatability slot from one target archive, including source/child/decision/boundaries/jobs/clock/debit/resource/device/cleanup identities. No original joint adoption, persistent-trainer, learning or multi-day admission inferred.


### Accepted co-scheduling and configuration convergence

[Target review](validation/coschedule-operator-20261025T151306Z.md) closes009:O004 andT110 at5014881. All20 trials verify12800 GPU boundaries/2880 native debits; overlap reduces component elapsed21.72/24.99%. Ten observed repeats span39.019–39.504s with4.42GiB sampled role upper peak. Counts78 complete/18 runtime pending/15 research held. Do not repeat isolated component screens. Real generation currently admits at most32 rows, so GPU128 capacity is not a drop-in training setting; useful generated-program throughput, persistent state/checkpoint/resource qualification and original adoption/cost gates remain. See [hardware convergence](validation/hardware-convergence.md).


## Supplemental model-size hardware screen v1

The user requested larger model capacity before the long-run choice. This prospectively revises the earlier no-model-change convergence direction without changing any accepted result or original default. [Contract](contracts/model-sizing.md)/[handoff](validation/model-sizing-implementation.md) freeze fresh8.12M/31.90M/104.50M profiles, actual update/generation/native work and finite15-slot roster. Original qualification/adoption/research requirements remain.

- [X] 009:M001 Freeze explicit hardware-only profiles/initializations, fixed real-data workload, complete/partial denominators, CPU oracle, finite15-slot/phase/resource/artifact rules and separate authority before timing.
- [X] 009:M002 Implement the exact sized constructor while preserving frozen default weights/shapes; shared source/data/reference/weight/resource binding, real disposable AdamW/validation/grammar/native observations, all-row/clock/debit evidence, bounded telemetry and no automatic model/learning selection.
- [X] 009:M003 Run165 host tests, actual public stdlib CLI and verified partial export; author/register real pinned CPU/HIP shape/count/baseline/cache/update/generation/native fixtures and one target command without installing host dependencies.
- [ ] 009:M004 Collect/review current CPU/HIP fixtures and all15 admitted/blocked/failed profile roles, actual counts/numerical parity/updates, generation denominators, native clocks/debits, memory categories, complete timing/source/device/cleanup and archive evidence. No quality, persistent-trainer, full-pipeline/default or multi-day admission inferred.
