**Current preparation:** [Sequence-learning pilot plan](validation/learning-pilot-plan.md) and [L001–L015 tasks](tasks.md#long-run-learning-pilot-preparation) now include actual held-out100-term learning/control curves and cautious further-training estimates. The proposed profile is48 hours inclusive with up to24 learner-role hours, subject to corpus feasibility, implementation and measured cost/admission. A short test validates plumbing without any learning-success threshold. No executable new HIP command or long-run admission exists yet; accepted hardware evidence is preserved.

**Current accepted result:** [Primary engineering soak PASS](validation/soak-operator-20261025T232558Z.md): both hour-long windows and full-state restart pass at unchanged runtime source d5c6de02. The workable experimental setting is104.5M/training16/inference16/native4/ordinary. Independent Docker remains pending; no repeat primary soak or size/batch ladder is requested. [Remaining work and next preparation](validation/hardware-readiness-next.md) retains the7.36% between-leg slowdown trigger and7.546/8GiB storage watchpoint. Original tasks87 complete/9 engineering pending/15 research held; no default, learning or multi-day admission.

The records below are historical handoffs and source-specific decisions; use the current disposition above for this implementation.

# Spec 008 carryover and scope revisions

**Recorded**: 2026-10-23  
**Source baseline**: `ef240bf8e5ad04df93555fa43097565250d9eed5` (PR #59)  
**Owning specification**: [Spec 009](spec.md)

This register accounts for all thirteen unchecked tasks in [Spec 008 tasks](../008-bigint-constant-synthesis/tasks.md), plus completed tasks whose evidence or admission must be reconsidered after a pipeline change. Transfer is not completion. Historical task checkboxes and source-bound receipts retain their meanings. Spec 009 owns the future work below; [tasks.md](tasks.md#requirement-and-carryover-coverage) maps these rows to concrete tasks and acceptance evidence before implementation.

## Outstanding task disposition

| Spec 008 task | Disposition in Spec 009 | Requirement / delivery boundary | Closure evidence and revised scope |
| --- | --- | --- | --- |
| **T093** historical retained-model rescue | **Deferred by user**, outside current completion gates | FR-022/024; all phases | No pre-007 model conversion, historical weights acquisition or historical model comparison is required. Keep read-only readers, historical numerical controls and compatibility tests. Revisit only through a separate explicit decision; no pass is claimed. |
| **T094** performance qualification | **Active; broadened to the whole pipeline** | FR-006–021/025; phases 1–5 | Preserve the existing small-value/large-integer corpus and retained-profile comparison. Add source-bound inference, CPU concurrency, overlap and end-to-end reports with declared resources. Investigate the retained small-value gap and batch-one uncertainty in context of total cost. The >5x slowdown rule remains a documented review trigger, not a requirement to match a narrower arithmetic profile or weaken semantics. Record whether further local runtime work is justified; do not require every hotspot to disappear. |
| **T097** operations and exports | **Active; consolidated validation** | FR-001–005/021/023; phases 1 and 5 | Validate every supported command or explicitly document its retirement; clean installation, no old solver reachability, retained readers, actual-engine input/role behavior, complete resume state, interruption and capped exports. Include permission-safe artifact collection. One phase command coordinates these checks; a combined scope can run all ready validation phases in one invocation and archive. Exclude the deferred historical-model experiment; do not remove its compatibility checks. |
| **T098** qualification identity and gate ledger | **Active; separate engineering from research admission** | FR-022–026; phase 5 | Publish immutable current-source identities and all pass/fail/pending gates, with dependency-aware reuse. Report engineering readiness separately from the still-negative learning gate and unadmitted full transfer screen. A blocked study is not evidence that completed engineering failed, nor permission to promote it. |
| **T099** matched contracts and sealed final cohort | **Active, gated research preparation** | FR-027/029; phase 6 before studies | Freeze matched data, initialization, vocab differences, seeds, controls, coverage, cost and selection rules, exposed diagnostic cohorts and a genuinely unexposed final cohort. Refresh budgets/cadence only prospectively from the optimized pipeline. Sealing does not authorize final access. |
| **T100** three paired literal/parameterized seeds | **Active, gated execution** | FR-027; phase 6 representation study | Execute all six registered arms only after runtime/device/learning and exact-scope cost admission. Preserve all completed, negative, interrupted and unstarted arms and their costs; no seed substitution. Existing seed defaults may be retained, but a changed roster requires a prospective identified contract rather than results-driven replacement. |
| **T101** complete transfer/control comparison | **Active, paired with T100 reporting** | FR-019/027; phase 6 representation decision | Report full 128+128 denominators, raw/rescue/equal-budget controls, correct/shuffled/absent/template results, visible-fit20 versus continuation100, size/length/complexity strata and seed/family uncertainty. Apply the inherited transfer gate and explicit stop/promotion decision; fitting a visible prefix is not structural transfer. |
| **T102** curriculum then replay | **Active, conditional on a supported recipe** | FR-021/028; phase 6 subsequent study | Freeze curriculum and replay as separate interventions, preserve earlier-stage retention and training-only admission, record realized cost and all outcomes. A negative representation decision must state whether a scientifically supported continuation exists; otherwise stop rather than quietly selecting a winner. |
| **T103** assistance | **Conditional; reviewed inputs required** | FR-028; phase 6 assistance decision | Require a compatible generic foundation, separately reviewed demonstration manifest, overlap audit, lineage and matched generic-only control. Missing inputs remain BLOCKED. A decision to omit or defer assistance may be frozen before final planning, explicitly restricting final claims; it does not complete the original assisted comparison. |
| **T104** encoder sensitivity and retraining | **Active, conditional bounded study** | FR-028; phase 6 before final decisions | Retain four post-training component ablations and prospectively selected matched retrainings; report active/total parameters, throughput, size/complexity strata and interactions. Distinguish sensitivity of existing weights from retrained utility. No full architecture sweep or requirement to retrain every combination. |
| **T105** immutable final study | **Active, separately authorized after decisions** | FR-029; phase 6 final-evaluation boundary | Freeze all intended study decisions, including explicit deferrals, arm/checkpoint identities and shared/per-arm budgets. Preserve sealed → authorized → exposed behavior; record exposure before first final-prompt access, retain failed/unstarted arms and reject changed-plan retries claiming freshness. No tuning on the exposed cohort. |
| **T106** discovery review | **Active; claims limited to supplied evidence** | FR-030; phase 6 final reporting | Review retained alternatives and omissions; distinguish finite matches, structural alternatives, conjecture, bounded verification, proof and novelty. If final evaluation is blocked, a limited archive audit may be delivered but cannot count as a completed final discovery study. |
| **T107** longer-run readiness | **Active; go/no-go is the deliverable** | FR-025/030; phases 5–6 | Publish measured resource settings, complete/pending gates, all study evidence and a go/no-go decision. A negative or inconclusive outcome is useful completion of the report, not qualification for long training. Require a subsequent explicit run decision; no automatic 12-hour-plus launch. |

## Completed work whose scope must remain explicit

| Existing evidence | Retained meaning | Spec 009 obligation |
| --- | --- | --- |
| T001–T088 implemented core, integration and retirement | Completed for their recorded implementation/evidence scopes | Preserve working behavior and known regression fixtures. Reopen affected semantic/design gates if a later change alters them; do not reimplement the entire feature merely to change its owner. |
| T089 arithmetic, T090 fitting, T091 engine containment, T092 actual HIP | Their historical source/image-bound receipts remain valid evidence of those sources | Requalify affected behavior and current packaged artifacts. Consolidate applicable checks into phase commands and final qualification; old receipt existence alone is insufficient. |
| T095 cost preflight | Completed scoped decision: diverse32 admitted; full transfer screen not admitted | Recompute both scopes as relevant on the changed pipeline. Old scope-specific success cannot admit a larger workload or a new source without dependency review. |
| T096 bounded learning attempt | **Complete NEGATIVE**: 31/32 correct meets the success count, but 31/32 completeness fails required 32/32. Consumed 7,730 seconds of a 14,400-second allowance. | Preserve original thresholds, identity, archive and ledger. No refunds, relabelling, implicit resume or automatic retry. A prospective new qualification attempt requires its own decision/identity/budget and keeps the original result visible. |
| T094 adoption/profile follow-ups, including W005 | Completed component evidence, not closure of all T094 | Reuse the reviewed baseline and bounded profiling harness where compatible. No need to repeat the entire historical investigation simply to begin Spec 009. |

The [latest pipeline review](../008-bigint-constant-synthesis/validation/phase-5-pipeline-profile-rerun-review.json) covers a frozen-weight diagnostic, not a replacement T096 attempt: each arm started 560/576 attempts and completed 545 generations, although isolated rescue/equal-budget scoring returned 32/32. Partial generation, changed deadline outcomes and diagnostic input scope remain explicit.

## Sequencing revisions and preserved gates

1. **Finish a reusable validation workflow first.** Establish fixed-work and fixed-budget comparisons before selecting optimization targets. Do not require another full profile replay when its source-bound findings already answer the baseline question; gather the missing representative long-output, difficult-fit and complete schedule costs in bounded scopes.
2. **Separate isolated improvements from their combination.** Inference, native concurrency and overlapped execution need independent evidence before joint tuning. The implementation plan must choose a small finite comparison set and admission criteria, avoiding a Cartesian sweep.
3. **Prioritize total verified throughput.** The existing 13.29–13.40x retained small-value execution gap and uncertain batch-one regressions must remain in the report. They motivate a documented decision about further work; they are not an unconditional demand to optimize WASM before GPU or scheduling work. Native wall spans include more than WASM execution.
4. **Separate engineering completion from learning promotion.** Phase 5 can produce a complete current-source engineering report and an honest blocked learning/research decision. It cannot mark the learning gate passed. Any new diverse32 attempt retains at least 31/32 independently correct rescue continuations and 32/32 completeness, with required controls and the declared 100-index horizon.
5. **Keep cost admission scope-correct.** Every proposed learning scope includes all scheduled updates, validations, diagnostics/controls, scoring, checkpoint/export, startup and cleanup. The inherited 80%-of-budget ceiling remains. An unmeasured component yields a pending forecast, not free work.
6. **Retain the transfer gate.** Per seed, require at least 95% complete coverage on each development split and independent verification of every admission. On structural-dev, require at least eight additional successes out of 128 against each shuffled, absent and template comparator in at least two of three seeds, and a mean paired advantage of at least 6.25 percentage points against each. Preserve the parameter-dev limit of at most four lost successes out of 128 versus the declared literal/full-input control mean, the predeclared family uncertainty method and the one end-of-screen gate decision. See the normative [experiment contract](../008-bigint-constant-synthesis/contracts/experiments.md).
7. **Do not make optional missing inputs an indefinite infrastructure blocker.** Assistance remains conditional on reviewed data. A recorded prospective decision to defer it can narrow the final-study roster, with claims restricted accordingly; the assistance comparison stays uncompleted. The same principle applies to an explicitly deferred research question, never to mandatory correctness, independence, learning or cost gates.
8. **Keep final exposure and long-run decisions separate.** A consolidated command may execute an already admitted batch and stop cleanly at the next decision boundary. It cannot invent demonstrations, choose future arms using hidden results, approve a fresh learning attempt or authorize a long run simply to finish a task list.

## Planning handoff requirements

The implementation plan and subsequent tasks must trace every row above and every Spec 009 requirement to a phase gate or explicit deferral. They must define finite run/repetition budgets, numerical adoption/regression thresholds, scheduler/deadline semantics, aggregate resources, evidence invalidation and the consolidated operator interface before implementation. The plan should reuse existing public entry points and evidence machinery where sufficient; this specification does not require a new profiler or a new standalone command for every comparison.

Research execution tasks must be distinct from implementation and local verification tasks, so unavailable HIP hardware or blocked research does not hide completed code work. A handoff must include exactly what one command the operator can run next, its expected cost and limits, its output archive, and the specific decision it can resolve. The user explicitly accepts long runs: combine all ready validation phases when possible, with finite limits, regular progress, shared prerequisites and one archive. Do not split commands merely to shorten an invocation or because a document has separate phase headings. If more than one invocation is unavoidable, explain the capability or decision boundary; do not offload routine orchestration to the operator.

## Implementation checkpoint (PR #60)

Code for phases 1–6 is wired to the combined entry point; current local evidence is 80 passing stdlib tests and a verified non-PASS public CLI archive. This does not close any remaining008 runtime/research carryover. See [implementation closeout](validation/implementation-closeout.md), [local evidence](validation/local-evidence.json) and the reconciled [009 task index](validation/task-evidence.json).

| Carryover | Current disposition | Required next evidence |
| --- | --- | --- |
| 008:T093 | User-deferred | No acquisition/conversion requested |
| 008:T094 | Open | Current-source whole-pipeline confirmation and retained 13.29–13.40x gap review; no speedup claimed |
| 008:T097 | Open | Target supported-command/operations, permission, interruption/resume and both-engine evidence |
| 008:T098 | Open | Review current-source runtime/device/resource/operations qualification |
| 008:T099 | Blocked preparation | Reviewed matched contracts and fresh sealed cohort; no final authority implied |
| 008:T100 | Not executed | Learning PASS, exact full-transfer cost and separately admitted six-arm batch |
| 008:T101 | Blocked | One registered transfer endpoint from the complete batch |
| 008:T102 | Not executed | Supported recipe, curriculum decision, then separate replay decision |
| 008:T103 | Blocked; no reviewed assistance inputs | Reviewed lineage and matched comparison or explicit prospective deferral; neither fabricated here |
| 008:T104 | Not executed | Four component sensitivities, then separately selected retraining decisions |
| 008:T105 | No final exposure | Immutable post-decision plan and explicit invocation |
| 008:T106 | Blocked | Actual final results and honest discovery review |
| 008:T107 | No longer-run authority | Evidence-backed readiness decision after the required gates |

Original008:T096 remains NEGATIVE with 7,730/14,400 seconds consumed. No new learning attempt was proposed/admitted, no assistance deferral was silently adopted, and no historical result was rewritten.

## Fourth operator review

The [source-matching fourth archive](validation/operator-20261024T141450Z-review.md) passes selected Podman CPU/HIP/operations prerequisites and ends with overall PARTIAL, no optimization adoption, incomplete costs and missing Docker portability. This supplies scoped evidence toward008:T094/T097/T098, but closes none of those full carryovers. T093 remains deferred and T096 retains its original negative ledger. No study/final authority is inferred.

## Replay viability closure

The [registered baseline replay scope](validation/replay-operator-20261024T172754Z.md) now passes with exact-source CPU/HIP and five complete baseline trials. It adds component viability and affected-source checks, closing009:T110 and supplemental009:R005 only. It closes none of the thirteen full008 carryovers or missing full-pipeline/cost/research gates; original008:T093/T096 remain unchanged.

### Current comparison implementation: separate phase-2 scope

The [new component comparison](validation/replay-comparison-implementation.md) implements the next isolated bookkeeping investigation with its own prospective child/order/limits. It is target-pending and closes no008 carryover. Accepted baseline source69807d38 and original negative/inconclusive measurements remain unchanged. Missing full-generation, native/joint, operations/portability, full-control costs and scientific gates remain explicit. Current111-task counts are77 complete/19 runtime pending/15 research blocked after reopening affected-source T110; no historical result is revoked or reassigned.


## Learning-pilot implementation/local handoff

The [new preflight](validation/learning-pilot-implementation.md) adds the user-requested sequence-learning evaluation and conditional further-compute analysis without closing any full008 carryover or changing008:T096 NEGATIVE. All unblocked code/local work is delivered; actual corpus feasibility/current-source CPU/HIP/restart/full-control costs remain target-pending. T011 closes and009:T110 reopens for changed runtime source, preserving87/9/15 original counts. The one finite command returns a single ZIP; L013 reviews it before any explicitly admitted long study. Old-source hardware evidence is retained; no size/batch/soak repeat.
