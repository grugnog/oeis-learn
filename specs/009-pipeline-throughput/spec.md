# Feature Specification: Bounded Pipeline Throughput and Research Completion

**Feature Branch**: `009-pipeline-throughput`

**Created**: 2026-10-23

**Status**: Draft — specification, not implementation or experiment admission

**Input**: User requests Spec 009 for measured GPU/CPU/memory optimization and parallelization, consolidated operator validation commands (one per phase or one across all ready phases; long runs are acceptable), incorporating and revising the unfinished work from Spec 008.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Validate a phase with one command (Priority: P1)

As the workstation operator, I can run one documented command for the current phase and receive one reviewable result bundle, without assembling a sequence of builds, tests, profilers and packaging commands.

**Why this priority**: Repeated manual runs have made qualification slow and error-prone. A reusable validation workflow is the first useful delivery, even before optimization.

**Independent Test**: Run a bounded baseline phase, including a deliberate failed prerequisite, using existing production behavior. Inspect successful and failed bundles without running a training experiment.

**Acceptance Scenarios**:

1. **Given** a supported engine and the phase's declared inputs, **when** its command runs, **then** it prepares or verifies the image, checks prerequisites, runs all applicable checks and comparisons, and exports a single integrity-checked bundle with a concise verdict.
2. **Given** missing HIP capability, an unavailable required engine, stale inputs or a failed preflight, **when** validation runs, **then** it stops before dependent expensive work and reports exactly which gates remain blocked; CPU evidence cannot satisfy a GPU gate.
3. **Given** interruption or a failed stage, **when** the same phase entry point is used again, **then** already verified compatible prerequisites may be reused, invalidated stages rerun, and both attempts and their consumed budgets remain recorded.
4. **Given** the normal supported workstation setup, **when** the operator follows a phase handoff, **then** no separate test, profiler, archive or cleanup command is required. Research decisions remain explicit boundaries rather than being silently crossed by a validation command.
5. **Given** several ready validation phases, **when** the operator selects the combined scope, **then** one invocation runs their prerequisite-ordered checks without intermediate manual launches, shares compatible preparation, reports progress and creates one combined archive even if the run is long. A failure blocks its dependants; independent safe checks may continue within the declared budget.

### User Story 2 - Improve throughput with trustworthy comparisons (Priority: P1)

As a researcher, I can tell whether an optimization produces more independently verified work per minute, which stages benefit, and whether correctness or completeness has changed.

**Why this priority**: High utilization alone is not a useful outcome, and faster partial work can hide dropped attempts or weaker verification.

**Independent Test**: Compare the existing path and one proposed change on identical frozen inputs and weights, then compare them under identical time budgets. Neither comparison requires retraining or final-cohort access.

**Acceptance Scenarios**:

1. **Given** the same fixed completed workload, **when** baseline and candidate are compared, **then** all expected jobs are accounted for, independent acceptance agrees, and elapsed time includes all required pipeline work. A capped arm is reported as incomplete, not as a faster equivalent run.
2. **Given** equal time budgets, **when** throughput is compared, **then** results report verified successes and completeness over the full frozen denominator, including unstarted jobs and unsuccessful outcomes.
3. **Given** a faster decoder or more simultaneous CPU work, **when** combined pipeline performance is assessed, **then** the report includes generation, fitting, verification, learning updates, checkpoint/export and startup costs separately; measured diagnostic shares are not labelled whole-training shares.
4. **Given** no material improvement or increased uncertainty, **when** the adoption gate is applied, **then** the existing default remains available and the result is recorded as negative or inconclusive without expanding the search automatically.

### User Story 3 - Use spare compute without losing control (Priority: P1)

As an operator, I can allow more native CPU work and overlap it with GPU generation while retaining predictable limits, candidate selection and recovery.

**Why this priority**: The reviewed workload spends substantial time in both generation and native work while using roughly one of sixteen allowed CPU cores. Coordinating these stages is a plausible gain, but changes ordering and budget risks.

**Independent Test**: Feed fixed candidate jobs with varied durations into sequential and concurrent execution. Vary completion order, inject failures and saturate resource limits before requesting hardware performance measurements.

**Acceptance Scenarios**:

1. **Given** identical completed per-job outcomes arriving in different orders, **when** candidates are selected, **then** the selected result and per-arm evidence are identical under the frozen selection rule; the fastest response cannot silently become the winner.
2. **Given** a saturated queue or exhausted parent deadline, **when** another job is submitted, **then** waiting is bounded and charged, admission can stop, and every job receives an explicit terminal disposition. More workers never multiply a prompt's allowance.
3. **Given** a worker crash, cancellation or memory pressure, **when** recovery occurs, **then** no orphan computation survives the cleanup allowance, another healthy request succeeds, and failures remain distinguishable from mathematical rejection.
4. **Given** inference batching or overlap changes, **when** fixed-work equivalence is checked, **then** prompt identity, random streams, row ordering, end-of-sequence handling, cache invalidation and independent verification retain their declared meaning.
5. **Given** available shared memory, **when** concurrency grows, **then** total memory and queued work remain bounded and a setting that worsens end-to-end throughput is not selected merely for higher utilization.

### User Story 4 - Finish qualification without rewriting history (Priority: P2)

As the maintainer, I can see what remains from Spec 008, validate the changed system once, and distinguish an engineering completion from permission to run another learning experiment.

**Why this priority**: Old source-bound successes and a negative learning attempt cannot automatically qualify an optimized scheduler or a new experiment.

**Independent Test**: Reconcile the carryover register against every unchecked Spec 008 task and exercise admission decisions using pass, fail, missing and stale receipts.

**Acceptance Scenarios**:

1. **Given** the prior 31/32-complete T096 result, **when** a later diagnostic scores 32/32, **then** the original negative result, thresholds and consumed budget remain unchanged.
2. **Given** changed runtime, scheduling or inference behavior, **when** qualification is assessed, **then** affected arithmetic, fitting, isolation, recovery, real-device and operations gates require evidence for the new identity; unrelated compatible evidence may be reused with a recorded dependency rationale.
3. **Given** a proposed new diverse32 or transfer screen, **when** cost admission is checked, **then** it uses the exact proposed workload and current pipeline, preserves full denominators and controls, and cannot borrow admission from a smaller or older workload.
4. **Given** the user's T093 deferral, **when** current qualification is evaluated, **then** historical-model conversion or missing historical weights do not block the from-scratch path; retained readers and numerical compatibility checks still run.

### User Story 5 - Complete controlled research through explicit gates (Priority: P3)

As a researcher, I can progress from qualified infrastructure to matched learning studies and a defensible longer-run decision, without turning hardware tuning into an uncontrolled algorithm sweep.

**Why this priority**: These are the remaining research goals of Spec 008. They depend on feasibility and qualification rather than being prerequisites for useful engineering improvements.

**Independent Test**: Validate frozen study manifests and admission/stop behavior with fixtures before executing any research. Each admitted study has one operator entry point and a complete report even if negative or interrupted.

**Acceptance Scenarios**:

1. **Given** passed prerequisites and a frozen admitted study, **when** its command runs, **then** it executes the registered arms within shared budgets and reports every seed, control and denominator; unsuccessful arms cannot be replaced opportunistically.
2. **Given** an unsupported recipe or failed promotion gate, **when** the next study is considered, **then** it is blocked with a reason; engineering completion does not become a learning-success claim.
3. **Given** absent reviewed demonstrations, **when** assistance is considered, **then** assistance remains explicitly deferred or blocked and no examples are invented. A decision to omit assistance must be frozen before final evaluation.
4. **Given** all intended recipe decisions are fixed, **when** final evaluation is explicitly requested, **then** only the immutable registered arms run, exposure and budgets survive interruption, and no result is used to retune the same final cohort.

### Edge Cases

- Mixed prompt lengths, finished rows, long token streams, reordered batches and repeated prompts with distinct random streams.
- Slow fitting, unsupported structures, independent-reference exhaustion, stale caches, late responses and a response arriving exactly at its deadline.
- Worker death during a fit or evidence write, double delivery, interrupted checkpoint boundaries, cancellation while producers are blocked, and a full output disk.
- Process-count or thread oversubscription, nested CPU parallelism, shared-memory bandwidth contention, worker replication of caches and memory exhaustion despite low GPU allocation counters.
- Missing sensors, overlapping CPU/GPU memory counters, asynchronous accelerator timing, profiler overhead and engine startup that is outside container lifetime.
- Partial workloads, incomplete controls, missing selected candidates and capped bundles; omitted verbose details must not erase per-prompt failure reasons.
- A changed image with an unchanged configuration, a changed comparison manifest with the same label, or an old receipt that passed for a different dependency closure.
- Host interruption before a container starts; temporary files or rootless-container permissions preventing artifact export.

## Requirements *(mandatory)*

### Functional Requirements

**Validation and evidence**

- **FR-001**: Each delivery phase MUST have one documented operator validation entry point that handles image preparation/verification, preflight, relevant regression checks, bounded comparisons, summary and archive creation. Normal operation MUST NOT require manually chaining subordinate commands. The same interface MUST support fresh execution and safe failed-stage reruns. It MUST also offer a combined scope for all ready validations across phases, sharing prerequisite work and producing one combined archive. Long runtime alone MUST NOT be a reason to split the operator handoff. Before expensive work, show the selected scope and finite time/resource allowances; during execution, emit stage progress and a liveness update at least every 60 seconds while the supervisor is running. Continue independent safe checks after a stage failure where their prerequisites hold, while blocking dependent work and preserving the overall non-pass status.
- **FR-002**: Validation MUST identify required capabilities before expensive work, including actual HIP execution for GPU claims and the selected Docker/Podman engine. Missing required capabilities MUST produce BLOCKED, not PASS. Both engines require coverage for the final portability gate; one invocation may exercise both when available, or combine compatible separately supplied engine receipts and report missing coverage explicitly.
- **FR-003**: Every invocation MUST emit a machine-readable stage ledger and human-readable summary with PASS, FAIL, BLOCKED, PARTIAL or INCONCLUSIVE outcomes, source/image/configuration/input identities, commands, durations, errors, exclusions and an integrity manifest. Cancellation or failure MUST retain a bounded partial bundle wherever storage remains writable; otherwise report the precise artifact failure and surviving paths.
- **FR-004**: Evidence reuse MUST be dependency-aware and hash-verified. Changed source dependencies, image, inputs, workload, hardware-sensitive settings or limits MUST invalidate affected stages. Reuse MUST NOT refund consumed experiment budgets, turn a weights snapshot into a resumable run, or overwrite prior evidence.
- **FR-005**: Validation MUST share compatible image builds and prerequisite checks across its stages. Expensive dependency build layers MUST remain reusable when only application source changes; exact source and packaged-artifact checks remain mandatory. Final affected-source qualification MUST not be replaced by previously passing unrelated tests.

**Measurement and adoption**

- **FR-006**: Before comparisons, freeze workload membership, weights, inputs, seeds, limits, controls, cold/warm conditions, repeated-run order, primary metric, material-gain threshold, acceptable regression limits and stopping rule. Use separate calibration and confirmation observations; a revised policy starts a new comparison identity. The initial adoption target is at least a 10% improvement in the chosen end-to-end primary metric, with no greater than 5% regression in completed-work latency on any mandatory workload stratum and no loss of fixed-work correctness or completeness. These are proposed engineering defaults, not measured promises; planning may revise them with a baseline-noise/cost rationale before candidate measurements, and MUST define finite repetitions and uncertainty rules.
- **FR-007**: Run two distinct assessments: fixed completed work for latency/semantic equivalence and fixed budgets for useful throughput/coverage. Count every scheduled job and every frozen prompt, including unstarted work; never present unlike partial work as an equivalent-work speedup.
- **FR-008**: Measure stage wall time, completed/verified work, CPU consumption, GPU activity, memory, queue wait, worker occupancy and end-to-end time. Disclose unavailable metrics and collection overhead. Distinguish fit/compile/execute/reference/transport where observable; a native request span MUST NOT be described as pure WASM time.
- **FR-009**: Measurement MUST include representative short and long generation, mixed lengths, difficult fits, legal large integers, batch-one and batched work, cold/warm caches, model updates and validation. Measure checkpoint/export and host startup separately. A training-cost projection MUST identify all scheduled components and remain explicitly incomplete when any material component is unmeasured.
- **FR-010**: Routine confirmation MUST use low-overhead aggregate telemetry. Detailed token, tensor, byte and memory-growth traces MUST be disabled by default and collected only in bounded separate diagnostic windows. Time overlapping stages without double-counting wall time; do not sum overlapping shared-memory counters as distinct physical usage.
- **FR-011**: Adopt a configuration only when it passes correctness/resource gates and the preregistered end-to-end material-gain and regression criteria. Retain a reproducible sequential baseline and a supported rollback configuration. Negative or inconclusive comparisons MUST stop or narrow the intervention rather than automatically expanding batch, concurrency, model size or experimental budget.

**Correct execution under concurrency**

- **FR-012**: Inference improvements MUST preserve model mathematics, precision, weights, grammar, token limits, prompt identities and random-stream policy unless a separately reviewed contract explicitly changes them. Define numerical tolerances before execution; changed completed candidates or admission outcomes MUST be reported and adjudicated, never concealed by aggregate scores. Optimization equivalence cannot pass with unexplained completed-candidate differences; any intentional change requires a separately identified policy and fresh qualification.
- **FR-013**: Batching MUST support heterogeneous lengths, finished-row removal/reordering and cache invalidation without cross-prompt contamination. Fixed-work deterministic and seeded stochastic comparisons MUST cover these cases on CPU and actual HIP; GPU performance claims require actual target-device evidence.
- **FR-014**: Native concurrency MUST be explicitly bounded by effective CPU capacity, aggregate memory, active work, pending work, caches, outputs and thread limits. Compare the sequential baseline and a small preregistered set of increasing concurrency settings; do not assume all sixteen cores are beneficial or launch an unbounded sweep.
- **FR-015**: CPU work and GPU generation MAY overlap only under a declared scheduling policy that specifies job ownership, stable identities, ordering, dispatch, admission, result collection, backpressure, drain and cancellation. Hardware utilization is supporting evidence; maximize independently verified useful work within the resource envelope rather than utilization percentages.
- **FR-016**: Selection MUST use frozen visible-only rules independent of response arrival order. For identical completed job outcomes, varying completion order MUST yield identical selections. Under wall deadlines, completed sets may differ: record these differences, give each missing outcome a reason, and never claim deterministic equality for unequal completed work. Waiting and late-result rules MUST be defined before comparisons.
- **FR-017**: Bound per-job, per-prompt, per-arm and whole-run work. Queue waiting and fitting/verification MUST consume their applicable parent allowances; parallel jobs MUST NOT each inherit a fresh copy of a shared budget. Cancellation MUST stop admission and terminate queued/active work within a declared cleanup allowance, including descendants.
- **FR-018**: Results MUST be associated with immutable job, candidate, model snapshot and policy identities, admitted at most once, and rejected if stale or incompatible. Caches MUST preserve profile/solver/engine semantics, bounded capacity and provenance; no hidden cross-arm cache advantage may enter a comparison without disclosure.
- **FR-019**: Preserve separate generation, fitting/admission and scoring dispositions for every arm and prompt. Unsupported, unsatisfiable-within-declared-bounds, unknown, timeout, unstarted, cancelled, invalid and verification failure MUST remain distinguishable. An unrelated unfinished control cannot invalidate completed evidence or supply missing evidence for another arm.
- **FR-020**: Preserve exact arithmetic, fresh candidate isolation, resource ceilings, independent final-source verification and known false-fit counterexamples. No performance optimization may replace verification with solver status, admit partial outputs, weaken hidden-data isolation, or reintroduce a retired solver.
- **FR-021**: Recovery and resumable learning MUST preserve cumulative budget, random/data ordering, optimizer, curriculum, replay and selection state at qualified boundaries. In-flight work from a different model snapshot MUST NOT be silently applied to a later update. No new asynchronous learning algorithm is implied by inference/native overlap.

**Qualification and inherited research**

- **FR-022**: Maintain the complete [Spec 008 carryover register](carryover.md), mapping all thirteen unchecked tasks to active ownership or explicit deferral, with the conditions for closure. Historical checked tasks retain their recorded source scope; document affected requalification without erasing those receipts.
- **FR-023**: Complete current-source operations qualification for supported commands, clean retirement, historical readers, normal/interrupted execution, resume, permission-safe exports and capped mandatory summaries. Consolidate these checks into phase validation rather than requiring separate user commands for each fixture.
- **FR-024**: Publish an immutable qualification report identifying every pass/fail/pending gate, evidence dependency and resource policy. Preserve T093 as user-deferred and the original T096 attempt as negative with its consumed budget. A later diagnostic success MUST NOT replace that attempt or automatically authorize another.
- **FR-025**: Recompute cost admission separately for any new diverse32 diagnostic and for the complete transfer screen using the qualified optimized pipeline. Include periodic and end-of-run controls, fitting, independent scoring, checkpoint/export, startup and conservative variability. Retain the inherited requirement that projected cost fit within 80% of the preregistered finite budget; an incomplete forecast cannot admit a run.
- **FR-026**: A further learning qualification, if proposed, MUST have an explicit prospective decision, new immutable identity and finite budget before execution, preserving the old ledger and the inherited diverse32 composition, at least 31/32 correct rescue and 32/32 completeness thresholds. Performance-validation commands MUST NOT launch this training, resume the old attempt or access a final cohort.
- **FR-027**: Carry forward matched literal/parameterized studies with three paired seeds, fixed generic data and complete raw/rescue/equal-budget/correct/shuffled/absent/template comparisons. Keep full 128+128 development denominators, visible-fit20 versus continuation100 distinction, selection isolation and the inherited transfer gate. Study schedules or budgets may be prospectively revised from measured cost; results-driven changes to thresholds, denominators or control strength are prohibited.
- **FR-028**: Carry forward separately preregistered curriculum then replay studies, reviewed generic-only/assisted comparisons, and four component sensitivity ablations followed by selected matched retrainings. Each has its own decision, admission, lineage and cost accounting; missing prerequisites remain blocked, and a prospective deferral is recorded rather than silently counted as a pass.
- **FR-029**: Freeze an unexposed final cohort before the revised research sequence and keep it sealed while selecting recipes, assistance and encoder settings. Final evaluation requires a post-decision immutable plan and explicit invocation, exposure recorded before work, cumulative budgets and all registered arms including failures. Final results cannot select candidates, checkpoints or recipes.
- **FR-030**: Publish discovery evidence with separate finite-match, structural-alternative, conjecture, bounded-verification, proof and novelty statuses, followed by a longer-run go/no-go report. Failure or inconclusive research is a valid reported outcome but MUST NOT satisfy a failed promotion gate. No phase command automatically launches a 12-hour-plus run.

### Key Entities

- **Comparison contract**: Frozen workload, baseline/candidate settings, seeds, metrics, limits, repetition and decision rules.
- **Resource policy**: Effective CPU/GPU/shared-memory capacity, concurrent and queued work ceilings, cache/disk limits, deadline hierarchy and cleanup allowance.
- **Job and arm evidence**: Stable prompt/candidate/snapshot identity, scheduled and realized work, per-stage outcomes, selection and independent verification.
- **Phase validation bundle**: Dependency-bound stage receipts, resource observations, aggregate results, compact per-prompt failures, omissions, integrity manifest and verdict.
- **Qualification and carryover register**: Historical evidence, current owners, deferred goals, affected gates and explicit permissions for subsequent studies.
- **Research decision manifest**: Registered hypotheses, arms/checkpoints, provenance, sealed cohort, shared/per-arm budgets and promote/stop interpretation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every engineering phase has one copy-pastable validation command and one resulting bundle; the combined scope runs all ready validations across phases with one invocation and one archive. A supported normal run requires zero additional manual build/test/profile/package commands. Active long runs report stage progress/liveness at least every 60 seconds; a failure never becomes a combined PASS because other checks succeeded. Deliberate research decisions and capability provisioning are documented exceptions, not hidden setup steps.
- **SC-002**: Every scheduled job and frozen prompt is accounted for in success, partial, failure and interruption cases. Zero incomplete jobs are labelled verified, zero missing controls disappear from denominators, and all required summary records survive permitted export caps.
- **SC-003**: All mandatory exactness, false-fit, isolation, ordering, cache, budget and recovery cases pass for the adopted source. There are zero independently admitted mathematical disagreements, duplicate admissions or cross-prompt/track leaks. Completed fixed-work candidate/selection differences are either eliminated or explicitly evaluated as a separate policy; they cannot pass as an equivalent optimization.
- **SC-004**: Before an optimized default is adopted, at least one setting achieves the preregistered material end-to-end improvement: initially at least 10% lower median complete-work elapsed time or 10% higher independently verified successes per minute, with the primary measure chosen in advance. Mandatory workload strata have no greater than 5% completed-work latency regression, and fixed-work correctness/completeness and all resource limits hold. Confirmation must meet the predeclared uncertainty rule; a favorable noisy point estimate alone is insufficient. If none does, publish a measured no-adoption decision; this completes the investigation, not the performance-improvement claim.
- **SC-005**: All confirmation reports distinguish equal completed work from equal budgets, contain the required stage/resource measurements or explicit unavailable reasons, and quantify observation overhead. No full-training speedup is claimed from diagnostic fractions or incomplete projections.
- **SC-006**: All injected exhaustion/crash/cancellation cases stop within their declared deadline and cleanup allowance, leave no active descendants, and permit a subsequent healthy request. Aggregate bounds hold under the maximum admitted concurrent workload.
- **SC-007**: Every phase bundle verifies its manifest and prerequisite identities; changing a relevant dependency prevents stale evidence reuse in every invalidation fixture. Actual accelerator and both supported-engine gates remain explicit.
- **SC-008**: All thirteen unchecked Spec 008 tasks have a current disposition and closure condition. T093 remains deferred, T096 remains negative, and none is checked complete merely because it was transferred.
- **SC-009**: No research starts with an absent, failed, stale or over-budget prerequisite. Every admitted study reports all frozen arms and denominators, its realized costs and a promote/stop decision, including negative and interrupted outcomes.
- **SC-010**: The final handoff separately states engineering readiness, learning qualification, research completion and longer-run admission. No final cohort is exposed before its immutable decision plan, and no long run starts automatically.

## Assumptions

- The target remains the user's ROCm workstation with sixteen CPU cores available and 128 GB shared system/GPU memory. These are available resources, not automatically granted container limits: the reviewed diagnostic used a sixteen-CPU, 16 GiB envelope. Effective limits must be measured; increasing them requires a new declared policy.
- Existing portable Docker/Podman operation, a standard-library-only host wrapper, workspace-local state, explicit role mounts and no host driver/service/account changes remain constraints. CPU-only development can precede user-run HIP validation but cannot satisfy it.
- The user explicitly accepts long validation runs to reduce manual handoffs. Combine already ready, already authorized checks under finite declared budgets; keep each phase separately selectable for targeted reruns. This preference does not remove resource limits or authorize new research decisions.
- The primary objective is verified throughput and reliable completion at bounded cost, not maximum GPU busy percentage, fully occupied memory, a larger model or speculative retraining. Keep the current small model and numerical semantics for engineering comparisons.
- Start with isolated inference and native-concurrency comparisons, then combine only supported improvements. One coordinated GPU owner and bounded CPU work are proportionate to this workstation; distributed execution, multiple competing learners, a general adaptive scheduler, new solver coverage and model architecture searches are out of scope.
- The current source baseline is PR #59 head `ef240bf8e5ad04df93555fa43097565250d9eed5`. Its reviewed diagnostic reports approximately 50.51% decoding and 45.99% native-request wall time, about one CPU core in use, sampled GPU busy of 31.44% during decoding, and 1.68 GiB peak container memory. These motivate experiments; partial work and timing differences prevent treating them as a matched-work speedup or whole-training forecast. See the [review receipt](../008-bigint-constant-synthesis/validation/phase-5-pipeline-profile-rerun-review.json).
- Constitution v2.0.0 and Spec 008 exactness/provenance/research contracts remain authoritative except for the explicit ownership and sequencing revisions in [carryover.md](carryover.md). No constitutional amendment, lowered learning threshold or historical-result rewrite is proposed.
- Frozen diagnostic weights may be reused for performance measurements without historical-model conversion. New strict training starts from random weights or a qualified generic-only descendant as declared; diagnostic weights-only state does not authorize resume.
- The 10% material-gain and 5% stratum-regression defaults are specification assumptions, not user-confirmed hardware performance. Planning must freeze or prospectively revise them with a rationale, together with finite sample counts, uncertainty rules and scheduling design, before candidate measurements. No unmeasured speedup is promised.

## Delivery and operator validation boundaries

These are required delivery outcomes for planning, not implementation tasks or commands that exist today. Planning may refine internal work without weakening the single-command handoff or combining independent research decisions. CPU-only implementation and fixture validation may proceed while HIP evidence is pending; dependent adoption gates remain blocked. Extend existing validation machinery where sufficient rather than requiring a new general orchestration framework.

| Phase | Independently useful outcome | One-command validation scope | Exit / next authority |
| --- | --- | --- | --- |
| 1 — Baseline and validation workflow | Reproducible baseline, representative workload and consolidated runner | Prerequisites, baseline checks, bounded measurements, failure/partial bundle behavior | Frozen comparison contract and usable evidence; no optimization claim |
| 2 — Inference efficiency | Qualified reduction of generation overhead and measured batching choices | Focused inference correctness, actual-HIP fixed-work and equal-budget comparisons, resource report | Adopt a supported improvement or record no adoption |
| 3 — Native concurrency | Bounded simultaneous CPU work with ordering and recovery guarantees | Sequential/concurrent fixed-candidate comparisons, limit/fault tests and scaling report | Qualified concurrency setting or retained sequential default |
| 4 — Pipeline overlap and joint tuning | Coordinated generation/native work within one resource policy | Combined pipeline equivalence, stress/recovery and end-to-end confirmation, including update/validation costs | Supported operating envelope and rollback choice |
| 5 — Operational qualification and admission | Current-source gates, complete carryover status and scope-correct cost forecasts | Relevant CPU/native/HIP and engine evidence, retirement/operations/export/resume, qualification bundle | Engineering pass/stop; any fresh learning attempt still needs its own admitted decision |
| 6 — Controlled studies and final decision | Explicitly admitted qualification/research, followed by a documented readiness outcome | One entry point per admitted study batch handles all subordinate validation, execution and packaging | Stop at recipe/assistance/retraining/final-exposure decisions; no automatic long run |

The operator interface must additionally provide a combined validation scope across all ready phases. It may run for hours within its declared finite budget, reuse compatible builds/prerequisites, and serialize measurements that would otherwise compete for the GPU or CPU. Phase boundaries alone do not require returning to the operator. Stop or mark dependent stages blocked only for actual prerequisites, unavailable capabilities, exhausted limits or unresolved decisions; continue independent safe validations where possible. Detailed phase receipts remain inside the single combined archive.

Phase 6 intentionally has decision boundaries: future arms depend on earlier findings and final-cohort exposure requires a separate explicit request. It MUST NOT be compressed into one unattended command that guesses those decisions. The same phase interface and manifest-driven batch execution should minimize operator invocations; no separate manual test commands are needed within an admitted batch. If a gate blocks research, report the blocked research separately from completed engineering rather than keeping completed optimization work indefinitely open.
