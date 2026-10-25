# Implementation Plan: Bounded Pipeline Throughput and Research Completion

**Branch**: `009-pipeline-throughput` | **Date**: 2026-10-23 | **Spec**: [spec.md](spec.md)

**Input**: Spec 009 and the user's clarification that validations should be combined wherever possible, even when the resulting command runs for a long time.

## Summary

Extend the existing portable foundation validator with a static dependency graph and a combined `all-ready` scope. It builds/checks compatible images once per engine/target, executes already authorized stages, continues independent checks after a failure, emits progress and exports one bounded archive. It does not create a general workflow framework or automatically launch learning studies.

Optimize in measured increments: cached-decoder bookkeeping at unchanged batch eight; an explicitly versioned deterministic selection/retention policy; bounded 1/2/4 native workers; then GPU-generation/CPU-fitting overlap and a small batch/queue comparison. Preserve exact arithmetic, independent admission, hidden-data isolation and complete per-arm outcomes. A single coordinator owns job identities, shared capacity, cache publication and terminal evidence. One GPU owner uses a frozen snapshot during each synthesis block; weight updates never race candidate work.

Measure fixed completed work and fixed budgets separately. Adopt only a prospectively declared configuration that meets the 10% end-to-end gain and 5% stratum-regression limits with the conservative five-pair rule in [comparisons.md](contracts/comparisons.md). A useful negative investigation is an allowed outcome, but does not establish a speedup or pass a learning gate.

All thirteen unchecked Spec 008 tasks are owned by the [carryover register](carryover.md). Complete the engineering report before admitted research; T093 stays deferred and the original negative T096 stays unchanged. A fresh learning attempt, matched screens and final-cohort access require distinct prospective manifests and their existing gates.

## Technical Context

**Language/Version**: Standard-library-only host Python; Python 3.12 in the pinned containers; existing Rust 1.90.0 native/guest builds. No toolchain change.

**Primary Dependencies**: Current pinned ROCm 7.2.1 / PyTorch 2.9.1 GPU image, Wasmtime 20.0.2, Z3 Rust binding 0.21.1 and committed native/WASM locks. Retain existing decoder/grammar, independent Python reference and container engines. No new service, scheduler framework or mandatory host package.

**Storage**: Workspace-local canonical manifests, append-only per-invocation stage/job ledgers, hashed receipts, existing complete learning checkpoints and one combined analysis ZIP. Source/image/policy identities bind each stage. Stage reruns have fresh output roots; existing completed learning runs are read-only inputs. See [data-model.md](data-model.md).

**Testing**: Existing Python unit/contract tests, Rust exactness/fitting cases, clean-image packaging/retirement, Docker and Podman role/permission/containment tests, actual-HIP cache/token parity and finite parameter updates. Add deterministic fake-clock/out-of-order/fault tests and standard-library-only full-host tests before scheduling changes. Measurement stages serialize competing resource use.

**Target Platform**: User's AMD Ryzen AI Max+ 395 / gfx1151 shared-memory workstation, portable Linux Docker/Podman. Sixteen CPU equivalents and a user-authorized aggregate memory ceiling of 90% of measured physical RAM for the complete active GPU/native role, not per worker. Preserve the historical 16 GiB comparison profile. Shared-memory capacity trials use named16 GiB/64 GiB/workstation90 profiles where available, with identical limits on both sides of each paired comparison. Missing hardware can block adoption evidence without blocking independent code/tests.

**Project Type**: Containerized research CLI with a GPU learner/generator, supervised native synthesis service and separate independent scoring roles.

**Performance Goals**: Lower complete pipeline wall time by at least 10%, with no more than 5% completed-work latency regression in any mandatory workload stratum, zero semantic/admission regressions and bounded resources. Five paired confirmation runs must satisfy the conservative rule; no extra sampling until favorable. Equal-budget verified throughput and coverage are also reported. High utilization is not itself a success criterion.

**Constraints**: One GPU owner; no asynchronous learning; no broader solver/model/precision changes; per-process native requests stay serial; exact values and current candidate resource profiles retained; aggregate queues/caches/memory bounded; deadlines include waiting; hidden truth only reaches an isolated scorer after visible selection. Existing solver retirement and historical readers remain supported. Validation duration may be long but is always finite and declared.

**Scale/Scope**: Initial workers 1/2/4; decoder batch eight for first intervention, later named policy compares 1/8/16 with only one factor changed at a time. Eight pending native jobs / 32 MiB globally, at most two decoder batches of lookahead, four GiB aggregate native RSS watchdog, one GiB per native slot for the initial pool. These native fault-containment limits are distinct from the much larger shared CPU/GPU role budget. Start role caches at existing global values; a separately identified larger-cache candidate and capacity-supported inference batches are measured in phase4. Initial engineering command has a 24-hour external ceiling including at most two engine builds; phase-specific caps and archive reservations are in [operator.md](contracts/operator.md). These are ceilings, not predicted runtimes or training authorization.

## Constitution Check

Constitution v2.0.0 remains adopted. No amendment or exception is proposed. User-directed portable Podman/Docker support remains applicable. The old historical model is not a prerequisite for from-scratch work.

| Principle / gate | Before research | After design | Required implementation evidence |
| --- | --- | --- | --- |
| I — Exact data and declared semantics | Pass: existing numerical/precision contracts retained | Pass: policy changes affect ordering, not arithmetic; exact assignments and hashes preserved | Independent arithmetic corpus, large inputs, unchanged failure semantics |
| II — Independent acceptance | Pass: solver success is provisional | Pass: raw/fit/reference stages remain separate; cache hits still admitted independently; v1/v2 seals explicit | False-fit counterexamples, final-source checks, stale/duplicate rejection |
| III — Bounded recovery | Pass: one aggregate resource envelope | Pass: queue/active/result/cache budgets, shared deadlines, five-second total cleanup, external role watchdog | Oversubscription, queue/backpressure, crashes, cancellation, full-disk and healthy-after-failure tests |
| IV — Reproducibility / hardware | Pass: pinned containers and source-bound evidence | Pass: stable jobs, finite preregistration, actual-HIP gates and complete update-boundary resume; no host changes | Real device updates, CPU/HIP parity, both-engine current-source isolation and resource receipts |
| V — Provenance / isolation | Pass: strict generic lineage retained | Pass: inference/native work sees visible data only; separate final-plan authority and exposure ledger | Mount/leakage tests, research-admission rejection, checkpoint/track lineage |
| VI — Honest discovery | Pass: no new novelty/proof claims | Pass: partial results and historical negatives preserved, final discovery labels distinct | Complete denominator/failure exports and scoped final report |

Both planning gates pass **in design**. No hardware performance, runtime qualification or scientific promotion is claimed by this document. If implementation changes mathematical/fitting semantics, reopen the affected Spec 008 algorithm gate with a new identity; do not label it routine integration.

## Project Structure

### Documentation (this feature)

| Document | Purpose |
| --- | --- |
| [spec.md](spec.md) | User outcomes and requirements |
| [research.md](research.md) | Source-backed decisions and alternatives |
| [data-model.md](data-model.md) | Manifests, identities, ledgers, state transitions |
| [contracts/operator.md](contracts/operator.md) | Combined CLI, stages, budgets, reuse and archive behavior |
| [contracts/scheduling.md](contracts/scheduling.md) | Native slots, overlap, deterministic policies, deadlines and cancellation |
| [contracts/comparisons.md](contracts/comparisons.md) | Fixed workloads, numerical parity, finite performance decisions |
| [contracts/research-gates.md](contracts/research-gates.md) | Inherited studies, admission, final exposure and go/no-go |
| [quickstart.md](quickstart.md) | Planned single-command operator handoffs and expected results |
| [carryover.md](carryover.md) | Every remaining Spec 008 task and closure condition |

The subsequent `/speckit:tasks` step derived [tasks.md](tasks.md) from this design. It preserves these six delivery phase numbers and separates code/local checks from hardware validation and admitted research execution.

### Source Code (repository root)

| Existing or proposed location | Responsibility |
| --- | --- |
| `scripts/foundation_run.py`, `scripts/foundation_local.py` | Preserve public stdlib wrapper; add pipeline validate/study routing and flags |
| `src/oeis_learn/container_validation.py` | Reuse stage adapters and current engine/device/installed-package qualification |
| `src/oeis_learn/pipeline_validation/` (new focused package) | Static stage registry, stdlib host coordinator, manifest validation, receipt/reuse graph and combined archive; no Torch/YAML import in host path |
| `scripts/qualify_bigint_phase5.py`, `scripts/profile_bigint_pipeline.py` | Thin compatible legacy entry points over factored stage operations; no duplicate nested archives |
| `src/oeis_learn/decoder/inference_cache.py`, `sampler.py`, `bigint_generation.py` | Cache metadata, transfer reduction and versioned generation-batch policy |
| `src/oeis_learn/synthesis/worker_pool.py`, `service_policy.py`, `client.py` | Slot lifecycle, one role-wide capacity/cache owner, bounded transport and completeness-aware reuse |
| `src/oeis_learn/fresh/bigint_diagnostics.py`, `synthesis_policy.py`, `candidate_archive.py` | Typed decode/native work boundary, stable jobs, v2 visible selection and deterministic alternatives |
| `src/oeis_learn/fresh/bigint_worker.py`, `bigint_state.py` | Integrate policy identities and drain barriers into the current worker and CheckpointStore; no second checkpoint implementation |
| `src/oeis_learn/fresh/pipeline_profile.py`, `pipeline_report.py`, `tracking/pipeline_profile.py`, `tracking/foundation_metrics.py` | Reuse workload execution/reporting/resource collection; explicit ordinary versus synchronized/trace instrumentation |
| `src/oeis_learn/experiments/source_snapshot.py`, build inspector, Dockerfile and existing CI | Bind manifests/new files into source/package inspection while preserving early dependency build-cache layers |
| `configs/foundation/pipeline_validation.json` (new), `configs/foundation/pipeline_studies/` (new) | Versioned finite validation defaults and separately authorized study manifests; no hidden local defaults |
| `tests/unit/test_pipeline_validation.py` (new), existing cache/synthesis/profile suites | Full-host fake-engine orchestration, deterministic scheduling/fault contracts, mandatory archive and identity coverage |
| Existing `tests/integration/test_bigint_decoding.py`, `test_bigint_gpu.py`, `test_bigint_workflow.py` | CPU/HIP parity, workflow/resume and actual device evidence |

**Structure Decision**: Extend the current application. The new host package coordinates existing capabilities; it does not duplicate fitting, checkpoint logic or profiling. Keep the native protocol unchanged initially by assigning globally unique worker generations across slots and recording slot IDs in coordinator receipts.

## Implementation Phases and gates

Spec Kit's research/design workflow is complete with these documents. The six delivery phases below are the implementation phases; they are not additional planning phases.

| Phase | Code/design work | Validation and exit | Dependency / research boundary |
| --- | --- | --- | --- |
| **1 — Baseline and combined validator** | Static stage graph; manifest/receipt/archive contracts; safe factored adapters; baseline/calibration workloads; progress, cap and reuse handling | Stdlib host/fault tests; source/package/role checks; bounded baseline and resource calibration on target. Complete commands and archive exist even before optimization. | CPU preparation first; HIP evidence may follow. Freeze comparison inputs and finite decisions before candidate timing. |
| **2 — Inference efficiency** | Hoist per-step metadata, bounded cache scratch, safe transfer reductions. Keep batch eight and old selection policy for first paired comparison. | Cached/full, reordered/finished/long-row CPU and HIP tests; unchanged completed candidates; isolated comparison; end-to-end adoption confirmation waits for the common v2 policy | No attention/precision/model change. A candidate may be retained experimentally without claiming production adoption until combined confirmation. |
| **3 — Deterministic policy and bounded native slots** | v2 policy-only serial comparison; stable identities; global slot/capacity/cache ownership; 1/2/4 concurrency and typed native jobs | Out-of-order/cost-invariance and v1 compatibility; reference/cache/admission tests; worker fault, cancellation and scaling measurements | Keep policy-only effects separate. Serial and concurrent timing arms both use v2; no solver redesign. |
| **4 — Overlap and joint confirmation** | Within-group/mode GPU/native overlap, reserved lookahead, fair dispatch, drain barriers; versioned batch/queue choices; nonintrusive instrumentation | Exact completed-work comparisons, fixed-budget coverage, aggregate stress/cleanup, joint five-pair end-to-end confirmation and rollback | Reuse qualified isolated candidates; compare only the bounded preregistered set. No candidate work survives a model-update boundary. |
| **5 — Operations, requalification and cost admission** | Finish command/retirement/resume/export coverage and dependency-bound qualification report; complete schedule cost projections | Current-source CPU/native/HIP, both-engine isolation/permissions, complete operations report, separate diverse32/transfer forecasts | Engineering report may pass while learning/research remains blocked. T093 deferred; negative T096 preserved. |
| **6 — Admitted controlled studies** | Wire consolidated study orchestration to existing qualified learning/evaluation paths; prepare contracts, not new algorithms | New learning qualification only if separately decided/admitted; representation, curriculum/replay, conditional assistance, encoder, final and discovery/readiness reports | Each decision freezes the next allowed batch. No auto-retry, final exposure or 12-hour-plus training merely because validation passed. |

Phase 2 and Phase 3 CPU-only development can proceed independently after the Phase 1 contract foundations; their hardware comparisons must not compete for resources. Phase 4 can evaluate a qualified candidate with the other component still at its baseline if its manifest explicitly declares that combination. Failed/inconclusive optimization does not force unrelated components to stop; failed correctness blocks affected adoption.

The combined validator selects all ready, explicitly authorized stages and deduplicates prerequisites across these phases. It does not stop solely at a phase heading or to keep the command short. It stops admission on global interruption/exhaustion, and blocks descendants of a failed prerequisite while independent safe checks continue.

## Validation and requirement traceability

| Spec requirements | Design / phases | Required evidence |
| --- | --- | --- |
| FR-001–005 | Operator contract, phases 1/5 | One-command/DAG deduplication, stdlib host, both-engine capability and stale-reuse tests; readable combined archive |
| FR-006–011 | Comparison contract, phases 1–5 | Frozen workloads, complete/partial pairing, uncertainty/adoption decision, telemetry overhead and component-complete projection |
| FR-012–013 | Inference design, phase 2/4 | Exact generated tokens/sources plus declared logit tolerances; cache invalidation/positions/finished-row tests and HIP updates |
| FR-014–018 | Scheduling contract, phases 3/4 | Aggregate capacities, stable dispatch/publication, budget/cancellation, stale/duplicate/cache tests and source-bound resource reports |
| FR-019–021 | Data model/scheduling, phases 3–5 | Every arm/job terminal, independent reference, all-model-state drain/resume and hidden-role separation |
| FR-022–026 | Carryover/research contracts, phase 5 | T093 deferred; T094/T097/T098 reports; T095 re-admission rules; negative T096 and old budgets intact |
| FR-027–030 | Research contract, phase 6 | T099–T107 manifests, full controls/denominators, stopped/negative arms, final exposure and longer-run decision |

Every success criterion SC-001–010 is exercised by the corresponding operator, comparison, scheduling or research gate; requirements-quality review is not a substitute for those runs. Keep code completion, local test completion, actual-HIP evidence and scientific outcome separate in task status.

## Complexity Tracking

No constitutional violations are proposed. The single host coordinator, bounded native slots and separate hidden scorer are justified by existing role isolation and measurable spare capacity. The v2 selection/retention change is necessary because elapsed-time ranking is scheduler-dependent; it is explicitly qualified, not disguised as an optimization. No distributed scheduler, persistent orchestration database, new optimizer, reference service or automatic tuning framework is added.

## Completion of this planning step

Research choices and interfaces are resolved in the linked artifacts. Numerical/runtime behavior remains unmodified. Planning checks must validate links, phase/requirement/carryover coverage, finite budgets, manifest/state consistency and Spec Kit prerequisites. Task generation is now recorded in [tasks.md](tasks.md) and [its review](validation/tasks-review.md). The next implementation increment is phase 1; actual qualification and CI execution were not performed by planning/task generation.

## Registered measurement refinement after the fourth operator archive

The original measurement scope retains its no-adoption/inconclusive outcome. The user requested continuing the proposed narrowed follow-up. [Baseline replay viability](contracts/replay.md) adds a separately versioned static manifest/registry for forced decoder replay only: existing prerequisites, one visible-only CPU oracle, exactly five baseline GPU roles and one archive. This executes no candidate optimization or research. Trial/phase/global caps, numerical tolerances, original default manifest, full-generation denominators and complete-cost requirements remain unchanged. New source requires pinned-runtime/actual-HIP validation; historical passes are not reassigned to it. This refinement is subordinate to the six-phase plan and does not close missing full comparisons or research gates.

The refinement is now executed/reviewed at producer `c9be16a`, with [registered replay PASS](validation/replay-operator-20261024T172754Z.md). Its derived60-second bound is component-only, not candidate admission or whole-pipeline/cost evidence. Original full-comparison and research gates remain unchanged; no repeat baseline collection is needed.

## Registered phase-2 component comparison after replay viability

The user requested continuing all phases after accepted baseline replay. The next isolated investigation is [cache bookkeeping replay comparison v1](contracts/replay-comparison.md), selected only by `configs/foundation/pipeline_replay_comparison.json`. It collects new-source baselines/reference, freezes a child and finite setup/startup allowances before any candidate observation, then runs one excluded hoisted/v1 warmup and exactly two AB/BA exploratory pairs. All roles are fresh and use identical eager FP32 forced inputs, reference, resources and aggregate1Hz observation. A failure blocks later arms; reporting retains every slot. No candidate/default/research promotion is granted.

[Implementation and single-command handoff](validation/replay-comparison-implementation.md) records105 passing host tests, verified local missing-engine export, and authored CPU/actual-HIP role fixtures awaiting their pinned-runtime collection. The six original phases and their full-generation, complete-cost, five-pair whole-pipeline and research requirements remain. Existing manifests/results and accepted baseline evidence retain their identities; T110 is reopened only for the newly affected source.

## User-directed hardware-first priority after accepted component comparison

[Source-matching comparison evidence](validation/replay-comparison-operator-20261024T193940Z.md) passes with10.54–11.28% lower complete forced-replay intervals for hoisted bookkeeping. Carry it forward experimentally while retaining all full-pipeline adoption requirements. Approximate1-of16 CPU-equivalent use, sampled GPU busy around30% and1.66GiB role memory against112.45GiB allowed capacity motivate a finite larger-batch screen and bounded native/overlap work, not a model enlargement or an accuracy experiment. Busy percentages and allocator/RSS counters are not substitutes for verified throughput.

The user explicitly prioritizes CPU/GPU/memory usage/performance and holds learning-performance work. Phases1–5 hardware work remains active; phase6 orchestration remains available but scientific execution is held. After a hardware configuration is qualified for effective throughput, containment, recovery and sustained operation, prepare a separate finite multi-day training decision with frozen costs/checkpoints/evaluation/stop rules. That later experiment will inform continuing organic/stochastic learning versus switching to curriculum-driven learning, with pre007 work as context. It is not launched now; no curriculum intervention or historical-result rewrite is implied. Numerical/provenance/final-data protections remain.

## Registered hardware-only equal-work batching screen

The [batch8/16 contract](contracts/replay-batching.md) and separate `pipeline_replay_batching.json` manifest implement the next finite capacity/throughput investigation. Both arms use experimental hoisted/v1 with the same32 frozen visible rows/32768 supplied BOS tokens; batch8 has four groups and batch16 two. A CPU full-prefix oracle covers160 row boundaries. Complete batch8 cold/warmup/three calibrations freeze the child, memory prediction/headroom and deadlines before one excluded batch16 warmup and two AB/BA pairs. Reporting retains every partial/unstarted row/slot and denies adoption/research admission. This is a scoped hardware investigation, not original phase4 production adoption or a training-batch/model change.

The [handoff](validation/replay-batching-implementation.md) records112 host tests and verified public CLI export, plus authored CPU/HIP real-role fixtures awaiting target execution. Existing manifests, six-phase structure, historical outcomes and user-directed learning hold remain. Batch32/native/overlap/sustained operation remain later bounded hardware work after relevant evidence; no automatic sweep or multi-day run.

## Accepted batching evidence and combined hardware experiment priority

The [batch8/16 screen](validation/replay-batching-operator-20261024T203537Z.md) passes at exact source, with48.98% median complete replay interval reduction, GPU busy about44% and1.71GiB sampled role memory. Use batch16 experimentally in the next prospective capacity scope. The user's request to combine multiple useful experiments favors one finite hardware invocation sharing build/reference/preparation and archive. Serialize isolated timing to avoid contention; freeze any capacity branches and stop rules before observing them. Include ordinary/telemetry overhead calibration before primary timing. Larger-capacity work remains diagnostic, with no model, training-batch, learning-quality or research-policy change.

## Registered combined hardware scope after the user's handoff clarification

The [finite hardware ladder](contracts/hardware-ladder.md) implements one shared invocation/archive for observation-overhead calibration and batch32/64/128 hypotheses against16. This is a prospective scope revision: the earlier manifest allowed a later32 branch only; the new identity explicitly preregisters64/128 conditional on supported preceding gains and capacity, with no further expansion. Preserve the accepted batch8/16 result and all original scopes.

Freeze32 visible inputs cyclically mapped into128 independent cache rows, not128 distinct prompts. Use the CPU full-prefix oracle/mapping for640 boundaries and131072 supplied tokens per arm. One ordinary/observed pair chooses a common mode before complete batch16 calibration. Each branch seals the same baseline-only bounds, exact weights and scaled memory/headroom before one excluded warmup and two AB/BA pairs. Stop on incomplete/parity/resource failure or unsupported progression; unexecuted roles remain NOT_SELECTED/UNSTARTED, never executed PASS. The full declared phase2 reservation is9810s within10800s; global/build/cleanup/disk/authority limits remain.

[Available checks and handoff](validation/hardware-ladder-implementation.md) record123 host tests, full simulated branch/fallback/failure/interrupt/receipt verification, public missing-engine export, and authored CPU/HIP128-row/all-size fixtures awaiting target collection. Experimental size selection does not adopt production defaults or qualify full-pipeline/research/learning performance. Native scaling, deliberate overlap and sustained operation remain later hardware work; learning stays held for the user's later multi-day decision.

## Accepted finite capacity ladder

[Actual target evidence](validation/hardware-ladder-operator-20261024T220811Z.md) verifies all22 GPU roles and selects128 experimentally:61.13% median interval reduction versus16, about90.5% sampled GPU busy,2.84GiB role memory, and0.62% overhead in the one observation pair. The finite roster ends at128. This capacity fixture uses32 unique visible rows replicated into128 physical rows and does not qualify generated-program throughput. Next hardware work concerns CPU/native queue/service behavior, deliberate overlap and sustained useful pipeline work. No automatic larger-size sweep or learning experiment.

## Native CPU capacity follow-up

The [native-capacity contract](contracts/native-capacity.md) registers a CPU-only phase1/3 scope after the accepted GPU128 screen. It isolates a prospective slot-wave admission policy from the old queued diagnostic and keeps the same immutable four-second request deadline once admitted. Full18-row baselines, two independent worker hypotheses and separately observed roles share one image/prerequisite/archive invocation. There is no cold-selected subset, production scheduler change, HIP repeat, automatic overlap or scientific authority.134 host tests and local export pass; three packaged-native fixtures and the actual screen await target execution. See [handoff](validation/native-capacity-implementation.md).

## First native-capacity collection: validation repair

The [source-matched target review](validation/native-capacity-operator-20261025T032351Z.md) records652 CPU passes and a serial-worker fixture metrics assertion failure before all performance work. The fixture now respects the existing serial/pooled metric shapes; separate completed CI failures from a wrong007 binding serializer are repaired with the actual foundation canonical functions.136 host tests and verified partial export pass. Runtime Python, full18-row work, immutable clocks, candidate roster and thresholds remain unchanged. Corrected-source CPU validation remains pending under the same single-command scope; no duplicate GPU scope or learning work is introduced.

## Accepted native component and co-scheduling boundary

[Actual native-capacity evidence](validation/native-capacity-operator-20261025T040709Z.md) accepts the full18-row screen and selects4 workers experimentally after68.63–70.71% paired interval reductions. All306 wave clocks/debits verify; no default changes. Alongside accepted GPU128 capacity this supports a new prospective component co-scheduling/parity/resource and repeatability scope. Isolated screens stay accepted at their sources; useful generated-program/full-pipeline confirmation, persistent-trainer stability and learning admission remain separate.

## Registered component co-scheduling and repeatability follow-up

The [co-scheduling contract](contracts/coschedule.md) registers phases1/2/4 after accepted native4/GPU128 evidence. One shared container/quota/watchdog owns fresh single GPU children and four native slots; serial and deliberate overlap arms process identical GPU128/native144 work. Complete references and five serial baselines freeze a child before two AB/BA exploratory pairs; their complete decision chooses the fixed ten observed repeats. Unique cycle jobs, original request/parent clocks, shared five-second error drain and existing artifact caps remain.154 host tests/export pass; new actual native/HIP fixtures and the complete target scope remain pending. See [handoff](validation/coschedule-implementation.md). This component scope does not close original queued/v2 native, useful generated-program/full-pipeline confirmation, persistent trainer or held learning gates.


## Accepted co-scheduling and converged hardware direction

[Target review](validation/coschedule-operator-20261025T151306Z.md) accepts all20 slots at5014881:21.72/24.99% component elapsed reductions, ten complete repeats and qualified shared resources. The isolated GPU/native/overlap search is complete for this finite roster. [Hardware convergence](validation/hardware-convergence.md) freezes the working direction and identifies one combined useful real-pipeline/persistent-state qualification before a separately decided multi-day run. Actual generated batch128 is not currently admitted; original full-work/five-pair/cost requirements and user-held learning investigations remain.


## User-requested model sizing before the persistent-state soak

The user requested greater learned-detail capacity and authorized a compact hardware-only8M/~32M/~100M screen before the integrated soak. [Contract](contracts/model-sizing.md) explicitly versions that architecture change; frozen production constructors, accepted component evidence and learning strategy remain. Fifteen fresh roles measure actual updates/validation/grammar sampling/native verification with the same real data and immutable budgets. [Handoff](validation/model-sizing-implementation.md) records165 host tests, verified export and authored current CPU/HIP fixtures awaiting target. Model choice and persistent-state qualification depend on those observations; no automatic size expansion, learned-quality conclusion, curriculum change or long training. Original counts77/19/15 reflect current-sourceT110 reopening.


## Corrected explicit sizing execution policy

[First target collection](validation/model-sizing-operator-20261025T170755Z.md) exposes an unconstrained fused encoder path in the declared eager sizing scope:687CPU/30subtests and all CPU references pass; first m8 HIP encoder parity fails and all15 timings remain unstarted. [Repair/handoff](validation/model-sizing-execution-repair.md) applies one audited/restoring FP32 mathematical-attention context to CPU references, GPU roles and fixtures, keeps tolerances/work/budgets unchanged and versions input/reference policy evidence while retaining historical readers.170host tests/export pass; corrected actual HIP is required before interpreting model capacity or proceeding to persistent-state qualification.
