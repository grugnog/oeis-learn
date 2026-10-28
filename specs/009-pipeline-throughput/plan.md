**Current state —2026-10-27:** [Calibration PASS](validation/pilot-calibration-operator-20261027T141405Z.md).
The prospective35h13m warm-run projection fits the38h24m admission target in48h.
T110/L012/G005/K006 close; original counts88/8/15. No repeat calibration/soak.
Next is [J001–J005 versioned long-run admission/profile wiring](validation/long-run-admission-next.md).
The legacy long-run schema rejects the candidate cost; execution remains unadmitted.

# Integrated remaining-work plan: long-run sequence-learning pilot preparation

**Branch:** `009-pipeline-throughput` · **Updated:** 2026-10-25
**Status:** Hardware soak accepted; current next work is [learning-pilot preparation](validation/learning-pilot-plan.md). User now requests sequence-learning evaluation and cautious further-compute estimates. Proposed48-hour inclusive/24-hour learner profile, larger strict corpus, six full-development checkpoints and one short plumbing/cost validation remain prospective; no long run is launched.
**Specification:** [Spec 009](spec.md) · **Execution checklist:** [tasks.md](tasks.md#integrated-soak-work-packages)

The linked learning-pilot plan and [L001–L015 checklist](tasks.md#long-run-learning-pilot-preparation) are the current preparation sequence. The following sections retain the accepted engineering-soak design and its historical execution sequence. They do not request another identical soak. The original soak replaced the earlier isolated screens. The [previous plan at2291446](https://github.com/grugnog/oeis-learn/blob/22914464b6ff21d342265e61004bd0dc0f2534d7/specs/009-pipeline-throughput/plan.md)
remains historical design context. The original six phase numbers,111 task IDs,
[contracts](contracts/) and [008 carryovers](carryover.md) keep their meanings.
A work package below may satisfy several original tasks; it never changes their
closure conditions merely to finish a checklist.

## Outcome and starting evidence

Deliver one public CLI invocation and one integrity-checked archive that establish
whether the104,496,662-parameter model can run the actual training pipeline under
sustained, bounded load, including validation, generation, native fitting,
checkpoint publication, a controlled process restart and exact-contract recovery.
Produce independent verdicts for correctness, containment/recovery, sustained
operation, throughput/default adoption, portability and cost readiness.

The [accepted sizing collection](validation/model-sizing-operator-20261025T180134Z.md)
passes170 host,692 CPU plus31 subtests,20 actual-HIP tests and all15 roles at
producer56f6156/source735ede… (720 source entries). Its104.5M model gives2,092
median training tokens/s and5.89GiB observed role upper memory on short work.
Earlier bookkeeping, GPU capacity, native slot-wave and co-scheduling evidence
remains accepted at its own producers. None needs a duplicate isolated ladder.
Current original-task counts are87 complete /9 engineering pending /15 research
held after reviewing actual current-source evidence. T011 retains a test-catalog gap;
seven original runtime gates and the T086 prospective costed handoff remain separate. T110 closes for unchanged
affected runtime source; documentation changes do not require hardware requalification.
Accepted older-source evidence remains historical. The execution sequence below
records the implemented soak design; the next action is the linked preparation package.

At the planning baseline, the practical gaps were specific: normal `fresh.bigint_worker.models` used
the fixed8.1M constructor; the sizing harness uses128 training/32 validation rows
and short disposable updates; normal learning has an update/early-stop endpoint;
and fresh sizing processes do not establish persistent state, long-run ledger,
checkpoint/disk or memory behavior. Production AdamW uses `foreach=False`, unlike
the sizing harness's default dispatch. Integrate and measure the actual trainer
rather than extending the sizing loop into a second trainer.

## Frozen direction and meaning of full scale

| Item | Direction to implement | Boundary |
| --- | --- | --- |
| Model | Explicit104.5M profile: width768, heads12, six encoder/six decoder layers, FF3072 | Fresh seeded initialization; no resizing or importing8M trained weights.31.9M remains a separately measured fallback, not an automatic retry after failure |
| Numerics | Audited FP32/highest matmul, TF32 off, deterministic algorithms, MHA fastpath off, SDPA math | Same validated execution policy in references, updates, generation and resumed worker; retain original parity tolerances |
| Data | Entire approved generic training partition and complete declared development-validation partitions | Freeze all IDs, counts, token/length strata, data order and hashes. Do not truncate to sizing's first128/32 rows or silently admit unqualified inputs |
| Learning behavior | Existing AdamW, LR, input groups, codec and target representation; curriculum/replay remain as currently declared | Engineering termination is explicitly time/work based; no loss/success-driven extension or early completion. No learning-quality comparison |
| Training batches | Finite hardware-only8/16/32 preflight; one choice frozen before the soak | Same fixed token workload and initialization per arm. Different optimizer trajectories are disclosed, not called numerical equivalence or a learning-recipe win |
| Real generation | Existing admitted1/8/16 and conditional32 policies; up to32 real eligible rows | Preserve prompt/RNG/mode/group semantics and capacity guards. GPU128 replay is capacity evidence, not permission to pad prompts or raise production limits |
| Native execution | Four bounded workers with common v2 policy, admitted only after queued-policy/fault qualification | Four-second service clocks include queue/owner wait; retain serial rollback and existing native thread/RSS limits |
| Scheduling | One GPU owner, bounded real-generation/native overlap and drain barriers | No updates against an in-flight synthesis snapshot; no competing measurement arms in parallel |
| Resources | At most16 CPU equivalents and existing90%-physical/ancestor/engine aggregate memory ceiling | Reservation/headroom and measured availability required; allocator, driver and cgroup counters are not separate additive pools |

Full scale means the selected full model, full approved data partitions, ordinary
operation mix and actual persistence costs. It does not require filling all RAM,
forcing100% device duty, manufacturing longer programs or executing a research
transfer/final-cohort study. If the approved inputs are incomplete, finish any
already-authorized preparation before handoff or name the exact input blocker;
a diagnostic subset cannot silently substitute for the full-scale claim.

## Execution sequence and implementation owners

### 1. Freeze one new engineering scope — S001

Register an explicit versioned soak manifest/registry/contract through the current
`pipeline_validation.contracts`, `registry`, adapters and stdlib host CLI. Record
fresh-engineering authority, model/data/initialization, execution and scheduling
policies, fixed batch roster, phase ownership, workloads, reporting schemas and all
bounds before target timing. Reject unknown/missing fields and research/final
references before mounting inputs. Preserve old manifest and receipt readers.

The new scope permits bounded diagnostic stateful updates and recovery of its own
engineering state. Its checkpoints are not research-resume authority and cannot
resume008:T096. Old learning/default contracts remain unchanged. Specify exact
work/update/job limits and maximum diagnostic/checkpoint counts from the frozen
cadence; validate their worst-case memory/disk/ledger bounds before admission.

**Exit:** Host schema/authority/dependency/reservation tests pass; complete planned
stage and trial inventories exist, including blocked/unstarted terminal records.

### 2. Integrate the selected model and actual persistent trainer — S002–S003

Extend the shared model/profile boundary, `fresh.bootstrap_contract`,
`fresh.bigint_config`, `experiments.bootstrap_models` and
`fresh.bigint_worker.models` with a closed versioned104.5M opt-in. Preserve old
v1 shapes, seeds, historical readers and default8M behavior. Model construction,
recipe identity, cost/shape identity and checkpoint authority must agree; validate
actual encoder/decoder/active/total counts and reject cross-shape resume.

Wrap complete CPU/HIP worker execution in the audited execution-policy context.
Reuse the current training/data permutation, validation and diagnostic code,
including actual optimizer settings. Add a bounded engineering continuation mode
to that worker; do not add another optimizer/update loop in pipeline validation.
The normal learning stopping rules remain unchanged. Engineering work has fixed
finite update and wall ceilings; reaching a work ceiling early is reported as
partial sustained coverage, never permission to issue another allowance.

Use the full admitted training partition. Keep the real configured validation and
periodic diagnostic cadence, all declared panels/modes/attempts and full row
accounting. Freeze cadence in consumed-training-token units as well as resulting
update counts when comparing batch sizes, so a larger batch cannot appear faster
by silently doing less validation/checkpoint work. Training loss and native
successes are diagnostic outputs; neither selects an extra experiment. Checkpoint
selection state is preserved, while continued training restores the latest full
state rather than silently switching to an older selected checkpoint.

**Exit:** Pinned CPU and actual-HIP fixtures demonstrate actual counts, original
CPU/full/cached parity and finite changed parameters in the real worker; the
full partition and actual operation schedule are visible in the run contract.

### 3. Complete state, bounded evidence and recovery — S004

Use `fresh.bigint_state.CheckpointStore`, existing ledger and pipeline boundary
helpers. Persist weights, optimizer, Python/NumPy/Torch/device RNG, data-order RNG
and position, selection/stopping, curriculum, replay, policy hashes, logical job
frontier, cumulative reservations/debits and engineering work-window position.
Every update, validation/mode transition, save and restart requires zero
unaccounted active/reserved synthesis work. Cycle/snapshot identities must prevent
repeated prompt work from colliding across checkpoints or renewing allowances.

Test normal roundtrip, controlled stop, interrupted update/save/worker, stale or
changed policy/profile/source, corrupt/latest-pointer state, duplicate or late
results, missing completion and insufficient remaining budget. In a small
controlled fixture compare the next resumed update/RNG/data/job behavior with its
uninterrupted reference; retain numerical tolerances without requiring unrelated
fresh HIP processes to have bitwise-identical floating-point weights.

Keep checkpoint storage bounded using the existing prune/publication mechanism:
latest and selected plus at most one in-progress generation. Account for peak
three-generation full104.5M weights/AdamW state, serialization buffers, input
links/copies, native caches and report growth before starting. Prune only this
invocation's owned state, never the engine cache. Keep checkpoints/private payloads
out of the public ZIP. Bound metrics, job evidence and accumulated frontiers at
the frozen maximum work count; compact interval aggregates retain complete counts,
source hashes and required fault/clock evidence. Do not drop bad windows or
truncate mandatory evidence to claim completion.

**Exit:** Complete-state/fault fixtures pass and prove recovery without a budget
refund, duplicated debit, leaked worker or unbounded retained state.

### 4. Close native, engine and operator prerequisites — S005–S006

Finish the original queued1/2/4/common-v2 exactness, false-fit, policy, cache,
transport, owner-observation and containment tests. Accepted slot-wave scaling is
evidence for that diagnostic only; it cannot waive T059/T060/T062. Preserve fixed
negative and difficult-fit cases. Classify mathematical rejection separately from
worker failure, timeout, cancellation and incomplete evidence. After injected
faults verify positive worker quiescence and a healthy subsequent request.

Combine current-source clean installed-package/source checks, supported-command
inventory or explicit retirement, historical readers, no retired solver
reachability, arbitrary-UID/rootless exports, failed prerequisites, startup/build
interruption, disk exhaustion, final-isolation and isolated CPU scorer tests.
Exercise Docker and Podman where available. Missing secondary-engine evidence
stays a portability blocker while independent safe primary-engine work can run;
one engine is never reported as both. Mandatory primary correctness/device/
containment failures block the soak. The deliberately injected failure must be an
expected fixture outcome, not a FAIL prerequisite passed through as success.

**Exit:** Applicable current-source CPU/native/HIP and operations receipts exist
with explicit per-engine availability and all injected-failure cleanup outcomes.

### 5. Calibrate once, qualify the real joint path and freeze one candidate — S007

Freeze the same initial104.5M state, complete training-work row/token order and
precision/optimizer settings for all training-batch arms. Each8/16/32 arm has one
excluded warmup and exactly three complete fixed-work measurements. Token counts,
validation work and setup-inclusive costs are reported; partial arms have no
favorable-subset aggregate. Select the highest median useful training-token rate
among complete resource-qualified arms; if within5% of the best, choose the
smallest batch within that band. Freeze this as the engineering batch choice
before sustained work; it does not change production defaults or LR automatically.

The manifest must declare the exact finite work quantity and per-arm reservation
before timing. It must fit the phase budget and include representative full-data
length strata. If the longest approved examples are far below1024, add separately
labelled valid-program stress fixtures at the declared capacity, with their own
finite roster. These establish allocation/long-cache safety, not useful throughput
or training-corpus coverage; do not add them to the learning corpus.

Run required long/ragged/finished/reordered/cache-invalidation CPU/HIP checks,
actual generated-token/source/admission mapping, v2 scheduling/cancellation and
representative difficult-native tests. Qualify real batch32 only through the
existing capacity/behavior proof; use the largest qualified real batch otherwise,
with the prospective fallback order recorded. Do not weaken its current positive
batch16/headroom guard on the strength of replay128 or sizing32.

Check ordinary versus aggregate1Hz overhead on identical representative complete
work before primary timing. Retain the5% overhead fallback: ordinary mode is
primary if exceeded, with a separately reserved observational segment. Mandatory
resource watchdogs stay on. Use bounded device events at existing drains rather
than synchronizing every ordinary stage. Emit phase/time-window CPU, GPU duty,
queue/native wait, memory categories, model update counts, training tokens,
validation rows, generation attempts/tokens/completions and verified programs.

Any equivalent-work optimization comparison uses the same104.5M architecture,
training batch, weights, data, v2 policy and matched work on both sides. Differences
between8M/104M or training batches are never optimization speedups. Carry the old
empty-complete-subset/inconclusive baseline forward; do not rerun an unchanged
negative scope seeking a favorable outcome. If a newly identified exact workload
has a nonempty complete subset and a full registered confirmation fits its
reserved budget, the original five fresh pairs per cold/warm scope may run. Freeze
all pair work/order/strata before candidate timing; use AB/BA/AB/BA/AB and the
original every-pair10%/stratum5%/coverage rules. Otherwise record INCONCLUSIVE or
NO_ADOPTION and retain the baseline. The engineering candidate can still be tested
under its explicit experimental profile. Do not expand the roster after results.

**Exit:** Frozen candidate, actual schedule/capacity/overhead receipts and explicit
adoption eligibility/disposition; missing correctness blocks the soak, whereas a
missing optimization speedup does not become a false hardware failure.

### 6. Execute the bounded persistent soak — S008

Use two predeclared3600-second workload windows with a single600-second reserve
for initial/final checkpoint publication, controlled stop/restart, drain and role
cleanup: a7800-second inclusive soak parent. The two windows provide up to two
hours of actual operation; report completed active duration and any admission/
checkpoint gaps rather than calling the entire parent productive training time.
A parent reservation is never renewed on restart. External launch/cleanup bounds
must fit that parent or an explicitly separate declared outer setup reservation.

Begin fresh with the frozen full-size model, full approved data and selected
engineering batch. Within each window run the existing actual operation schedule,
including representative update, validation, periodic generation/native fitting
and checkpoints. At the midpoint drain, save, terminate the worker/container and
verify quiescence. Restart a fresh worker from the latest full checkpoint under
the identical contract and remaining ledger/window allocation. Do not reset data
order, job/frontier identity, optimizer/RNG, cadence or remaining budget. Keep all
worker/client state alive between ordinary boundaries within each leg; repeated
fresh sizing roles are not a persistent soak.

Require at least one full training-partition traversal, one complete configured
validation/diagnostic/checkpoint cycle in each leg, and a real post-resume update
and subsequent cycle. If preflight predicts those minima cannot fit, report the
proposed schedule infeasible before the soak; do not shrink panels/attempts or
renew prompt clocks. At runtime missing minima are PARTIAL. A frozen full attempt
roster may include legitimate grammar dead ends, mathematical rejects and original
deadline outcomes, but each needs its correct terminal state. Unstarted or lost
scheduled work remains incomplete, never a verified program. Positive native
success fixtures remain labelled correctness work separate from generated work.

To exercise native coexistence even when fresh-model generation yields no usable
program, preregister one bounded18-case native health/stress pulse per ten minutes
of each workload window (at most12 pulses total), under the same four-worker/
four-second/shared-role limits. Run pulses only at declared frozen-snapshot
synthesis points and drain them before updates/checkpoints. Use the retained exact
corpus; never substitute probe programs for model outputs or feed them into
training. Track probe IDs, times, debits and correctness independently from real
generated-program fitting. Their cost remains in total mixed-soak wall time;
production-work cost forecasts identify this artificial load separately. If no
generated program reaches native fitting, explicitly mark that natural path
NOT_EXERCISED while reporting the separately stressed hardware/worker path. This
avoids making a positive learning outcome a prerequisite for the hardware soak,
or needing a second HIP run simply to exercise otherwise idle native workers.

Use five-minute windows and per-component counts/timings. Reserve initial
settling separately; compare the first two and last two complete matched windows
within each leg, and equivalent work across restart. The registration fixes
minimum window count and workload matching before execution. Report thermal/
frequency and sensor availability when exposed. No mandatory sensor is invented.

**Exit:** Independent sustained-operation verdict and all checkpoint/job/resource/
window receipts; no learning-success threshold or final-cohort access.

### 7. Report, reconcile and stop at the next decision — S009–S012

The archive/report must distinguish:

- **Correctness/containment/recovery:** finite updates, declared numerical/token/
  source/admission semantics, clean engine roles and exact cumulative state.
- **Sustained operation:** complete required windows/cycles, bounded resources,
  no deadlocks/leaks of active jobs, restart success, checkpoint/disk containment.
- **Performance:** absolute useful work per whole minute and phase/window, full
  denominators and variability; adoption uses its original separate rules.
- **Portability:** demonstrated primary/secondary engine coverage and blockers.
- **Cost/readiness:** complete scoped costs or explicit unknown components and
  an evidence-backed next-run decision, without automatic learning admission.

Record5% or greater sustained matched-window throughput decline and anonymous/
file/device growth as investigation triggers, not proof of a leak or permission
to rerun selectively. To classify sustained readiness PASS, all preregistered
matched steady windows must exist, show no more than10% rate decline at the
registered endpoint, and settled anonymous/device residency must not grow beyond
its predeclared live-state/cache bound. S001 defines the exact byte ceiling from
known live allocations and maximum cache/checkpoint/job state before target
observation; an undefined bound makes readiness INCONCLUSIVE. Account for new
optimizer/cache populations and file cache explicitly. A short rate rebound,
missing sensor or incomplete window cannot erase an earlier breach. Immutable
safety breaches remain failures even if the last sample looks normal.

Measure whole invocation and startup/stop/restart/checkpoint/export costs, not
training-only72h extrapolation. Publish diverse32 and full128+128 transfer forecasts
separately with every declared control/diagnostic/scoring cost. Reuse compatible
evidence, and measure missing components only under their admitted finite scope;
unknown or incomplete costs remain unknown. The inherited80%-of-allowance rule
still governs a proposed learning scope. This engineering soak does not need a
full scientific transfer screen merely to establish stable hardware operation.
Retain the small-value13.29–13.40x gap and >5x review-trigger disposition.

Finish all unblocked code and local fault checks, update source/bindings with the
correct serializers, commit/push periodically, then supply one exact command
with expected cost, hard limits and one ZIP path. Build, test, run, recover, score
where authorized, report and archive are subordinate stages, not manual operator
steps. Do not issue another HIP handoff while integration remains unfinished.
After the target archive arrives, verify member/source/receipt/job/state hashes,
close only evidenced tasks, publish PASS/FAIL/PARTIAL/INCONCLUSIVE and stop without
waiting for newly queued PR CI. A failed soak produces one concrete failure
investigation, not an automatic retry or fallback-model run.

## Budgets and stage ordering

Keep the existing operator caps; S001 must validate each static subreservation
before launch. Selected scope owns a new registry, so it must not blindly append
the old18000-second joint stage to another full-size soak.

| Owner | Hard reservation ceiling | Contents / subdivision rule |
| --- | ---: | --- |
| Per-engine shared builds, at most two engines | 14,400s each | Existing pinned image targets, shared once; no nested fresh allowance |
| Phase1 | 3,600s | Host/engine/CPU readiness, input/source checks and frozen baseline coverage |
| Phase2 | 10,800s | Actual HIP600s, selected-model CPU reference, finite training-batch/long-sequence/inference qualification; exact arms subdivided before freeze |
| Phase3 | 7,200s | Queued-v2 native policy/exactness/scaling and fault/cleanup receipts |
| Phase4 | 18,000s | Joint safety3,600s; overhead/ordinary-observational qualification1,200s; inclusive soak7,800s; at most5,400s for eligible original-rule confirmation/reporting |
| Phase5 | 14,400s | Both-engine command/retirement/permissions/operations, isolated scorer, missing admitted scoped-cost measurements and qualification reports |
| Archive | 900s | One verified public bundle, mandatory evidence first |
| Emergency cleanup/reporting reserve | 300s | Retained invocation-wide; never borrowed for more work |
| Total declared worst case | 84,000s | Within unchanged86,400s global ceiling; unused time does not extend a role/prompt/window |

A separate observational segment, if overhead forces ordinary-primary fallback,
comes from the stated phase4 qualification reservation, not an extra soak. The
conditional confirmation branch cannot start unless its entire cold/warm roster
fits5,400s after accounting for reports. Otherwise it is explicitly unstarted/
inconclusive, with no default adoption. Any new required substage must fit its
owning phase by a prospective revision before timing; no borrowing or cap renewal.

Planning estimate: approximately3–6 hours with warm caches for the combined
qualification, preflight and two-hour-work-window soak, dependent on operations
and conditional confirmation. This is uncalibrated, not a promised duration.
Cold builds can use their larger explicit allowances. The final implementation
handoff supplies the measured/prospective stage estimate and exact reservations.
Before admitting a stage, reserve its worst case plus cleanup/archive; an exhausted
budget still exports blocked/partial evidence.

Preserve5s role cleanup,4s native service,60s production generation versus the
separately named180s diagnostic,1024-token cap,8MiB streams,8GiB transient disk,
128MiB traces,512MiB public ZIP and16MiB report reserve. Check archive and transient
upper bounds using the maximum admitted work count before dispatch; mandatory
proof/job evidence cannot rely on compression or optional truncation. Use compact
bounded telemetry and data links, keep tensors/checkpoints/raw native payloads
private, and retain all counts/receipts/faults within those limits.

Static order is shared readiness → selected-model/reference and native/operator
prerequisites → fixed efficiency/capacity/measurement freeze → joint safety →
soak with one controlled restart → conditional eligible confirmation and scoped
cost/qualification reports → archive. Independent safe checks can continue after
a failure. Hardware timing arms are serial; only named GPU/native overlap shares
an active role. A missing secondary engine does not prevent an independent primary
soak, but it prevents full portability closure. Global resource breach or failed
quiescence stops further compute admission.

## Minimize HIP handoffs: one combined collection is the target

The operator target is one initial HIP command and one result ZIP for this whole
engineering investigation. Reference creation, finite batch arms, eligibility and
overhead decisions, policy freeze, joint checks, both soak legs, controlled restart,
conditional confirmation, cost/reporting and export are orchestrated within that
invocation. The operator does not stop after preflight to choose a batch, start a
second soak command, manually resume, run a profiler or repackage results. Fixed
safe decisions follow their preregistered rules; no new experiment is invented
from intermediate results. Negative adoption results do not pause an otherwise
qualified engineering soak. Independent safe checks continue after a blocked
branch and every planned slot is exported.

Before handoff, finish all unblocked integration, host/CPU/fake-engine/fault tests
available here and inspect already-available relevant CPU/packaging CI evidence.
Include the real pinned CPU and actual-HIP prerequisites in the combined command
where this workspace cannot run them. Do not wait for newly queued PR CI, and do
not split authored but ready checks into a separate user trial just to get faster
feedback. Check input/profile/architecture/deadline/ledger/artifact bounds and
failure exports as a whole before publishing the command.

A further HIP loop is justified by a concrete failed mandatory check, demonstrated
runtime defect, missing capability or an explicit new decision. Diagnose the
existing ZIP and finish all unblocked repairs first. Then rerun only the affected
scope plus required prerequisites; reuse prior PASS receipts only when verified
source/image/dependency/workload/resource compatibility proves they apply. Do not
weaken a conservative closure or hide failed/negative work to avoid a rerun.
Cold/warm confirmation cannot reuse exploratory or previous favorable pairs.
Retain one archive per invocation and link its producer/consumed budget to any
necessary follow-up. No repeated isolated size/capacity ladder or routine manual
build/test/soak/result sequence is planned.

## Every remaining original engineering task has an endpoint

The following18 original engineering tasks, plus reopened affected-source T110, are currently pending runtime. These mappings define what
the combined implementation/collection must deliver; a successful soak alone
cannot tick every row. Keep a row open with its exact blocker if its complete
original scope remains unexecuted or incompatible.

| Original task | Work packages | Required closure evidence |
| --- | --- | --- |
| T011 | S006,S008,S010–S011 | Real permission/full-disk/unsafe-member/interruption export fixtures and verified failure archive |
| T025 | S006,S008,S011 | Real-engine DAG, expected failed prerequisite, interruption and receipt index in `validation/phase-1-operator.json` |
| T026 | S009,S011–S012 | Reviewed one-command US1 acceptance matrix in `validation/phase-1-operator-review.md`, with missing capability explicit |
| T031 | S007–S008,S011 | Current representative overhead/ordinary fallback, bounded sensor/device/event attribution and window reports |
| T034 | S001,S007,S011 | `validation/phase-1-baseline.json` freezes current exact workload and gaps before candidates; preserves original negative/empty-subset result |
| T041 | S002,S007,S011 | `validation/phase-2-inference.json`: actual-HIP long/cache/generated-token/source/admission equivalence at the declared profile and scope |
| T059 | S005,S010–S011 | Current-source common-v2 serial/queued1/2/4 exactness/cache/policy/fault tests and finite comparisons, `validation/phase-3-local.md` |
| T060 | S005,S011 | `validation/phase-3-native.json`: exact queued-scope occupancy/transport/reference/strict clock/debit and engine cleanup evidence |
| T062 | S005,S009,S011–S012 | Source-bound scheduling/admission/cache counterexample acceptance matrix; arithmetic changes reopen their semantic gates |
| T064 | S004,S010–S011 | Complete actual CPU/HIP state/barrier/resume/fault fixtures plus full-size controlled restart and cumulative ledger |
| T069 | S007–S008,S011 | Ordinary unsynchronized overlap/resource interval observations and separate CPU scorer launch after GPU-role drain |
| T073 | S005,S007–S008,S011 | `validation/phase-4-joint.json`: actual HIP joint parity/stress/cancel/update/validation/complete-pipeline results, every failed/unstarted job and excluded stress case |
| T075 | S007,S009,S012 | Original-rule adoption evidence or explicit retained baseline/NO_ADOPTION/INCONCLUSIVE; experimental soak settings are not silently made supported defaults |
| T076 | S009–S012 | Reviewed envelope/rollback, updated task ledger and one consolidated handoff in `quickstart.md` |
| T078 | S004,S006,S010–S011 | Supported/retired inventory, installed readers/no retired reachability, arbitrary UID/rootless/scorer and complete-state interrupted operations |
| T084 | S005–S006,S009,S011 | `validation/phase-5-qualification.json`: actual current-source arithmetic/fitting/containment/HIP/operations per engine; missing second engine remains pending |
| T085 | S009,S011–S012 | `validation/phase-5-report.md`: engineering/cost/adoption decisions, retained small-value gap and strictly scoped008:T094/T097/T098 disposition |
| T086 | S009,S011–S012 | Reconcile `carryover.md`, task evidence and quickstart; measured next-step cost/limits/archive and all scientific holds |

T110 is reopened for the new runtime/test/contract closure; its accepted older-
source completion remains historical. It closes only after final affected-source
packaging and actual target evidence. T111 is reconciled again at closeout; prior completion
is not erased. SupplementalS tasks track new integration, not replacements for
original tasks or a change to the original111-task denominator.

## Research tasks remain integrated at their decision boundaries

These15 original tasks stay held during the engineering work. The engineering
report must list each group below with its actual prerequisites and next decision;
it cannot quietly remove them or execute them to fill the remaining task list.

| Original tasks | Later work and dependency | Disposition after this soak |
| --- | --- | --- |
| T093,T094 | Freeze/admit a new learnability scope, then execute/review its full endpoint | Await engineering/runtime/provenance/exact-cost evidence and the user's separate learning decision; no retry/resume of008:T096 |
| T095,T096,T097 | Matched representation contracts/cohort, six literal/parameterized arms, transfer decision | Prepare only when appropriate; execution retains prior-learning, full-control, cost and separate admission gates |
| T098,T099 | Curriculum intervention, then separately decided replay | Held for later evidence-backed learning-strategy choice; no intervention in hardware soak |
| T100,T101 | Reviewed assistance inputs/decision, then matched execution | Missing reviewed inputs stay blocked or prospectively deferred; no invented demonstrations |
| T102,T103 | Encoder sensitivity and separately selected retraining | Held; size feasibility does not qualify component utility or authorize an architecture sweep |
| T104,T105 | Immutable final plan and explicit final evaluation | Final cohort remains sealed; no exposure during engineering |
| T106 | Discovery/claims review | Full task waits for required scientific evidence; engineering summary is not discovery completion |
| T107 | Full evidence-backed longer-run readiness report | Publish hardware readiness now separately; do not mark the full research readiness task complete solely from soak stability |

A later multi-day training proposal uses the appropriate fresh learning admission,
exact whole-schedule costs and explicit decision. It need not automatically run
all downstream studies first, and a positive soak does not itself admit it.
Original008:T093 stays user-deferred;008:T096 stays NEGATIVE with7,730/14,400s
consumed. All thirteen008 carryovers remain accounted for by the original register.

## Technical context, governance and completion boundary

Keep the standard-library host, pinned Dockerfile/locks, Python3.12 container and
existing Rust/native/guest stack. No host Torch/toolchain/service/VM/SSH or new
scheduler/database dependency. Preserve explicit mounts, isolated scorer, existing
role supervision, portable Docker/Podman support and one source-bound archive.
Use the current source snapshot and actual foundation canonical serializer for
implementation bindings; do not substitute pipeline identity serialization.

Constitutionv2.0.0 remains adopted. This plan preserves exact arithmetic and
independent acceptance, finite global/job/storage budgets, actual HIP updates,
complete recovery, strict generic provenance and hidden/final isolation. The
user-directed explicit larger model and hardware-only batch screen are new
versioned engineering choices, not changes to historical recipes. Design satisfies
those constraints; runtime satisfaction still requires the target evidence.

Preserve requirement coverage: FR-001–005 through S001/S006/S008; FR-006–011
through S007–S009; FR-012–013 through S002/S007; FR-014–018 through S004/S005/S008;
FR-019–021 through S003/S004/S006/S008; FR-022–026 through S009/S011/S012; and
FR-027–030 through the explicit research decision table. Success criteria remain
scoped to their original acceptance tests, not replaced by a generic soak PASS.

The planning deliverable is complete when the checklist and machine-readable
mapping cover all18 pending engineering tasks and15 held research tasks, budgets
fit unchanged parent caps, links/source identities verify and no task is falsely
closed. The next implementation starts atS001 and proceeds through all unblocked
code/local verification before a user-run target command. Completion of the
engineering investigation means a reviewed positive, negative or inconclusive
report plus explicit remaining blockers; it does not assert whole-Spec009 research
completion or launch a multi-day run.

## Implementation checkpoint before the first target collection

The [closed soak contract](contracts/soak.md), [single handoff](validation/soak-implementation.md) and [local evidence](validation/soak-local-evidence.json) implement the sequence above. The38-stage both-engine roster fits all existing phase reservations; the actual public CLI and190 host tests pass available checks and export explicit missing-engine evidence. The existing trainer now owns explicit m100 FP32 updates, full data, token cadence, in-place seven-state recovery and immutable7800s parent. No second trainer or default change was introduced.

Preflight reserves270s worker work/300s role/330s external per batch trial, with two full training passes and full validation. Joint fixtures test the full model/long context and a labelled small-row recovery fixture before the actual full-data legs. All96 periodic prompts must dispatch to count as a complete cycle; legitimate grammar/deadline outcomes remain distinct from scientific success. Native pulse intervals never catch up after long generation. The mandatory1Hz resource audit stays active if optional instrumentation exceeds5% overhead. Arbitrary-UID/export/interruption fixtures run per engine, and a public CPU scorer runs after GPU drain.

Original full-work/mandatory-strata confirmation eligibility is explicitly false for this new model/batch scope. Its20 cold/warm slots remain UNSTARTED and defaults remain baseline; no favorable exploratory subset is promoted. Original diverse32/transfer costs remain unknown. At this initial checkpoint S002/S004–S006 and S011/S012 awaited target evidence; the first target review below now records S002 closure. Original runtime and held research tasks do not close from host simulations.

## First target review and bounded permissions repair

[The first collection](validation/soak-operator-20261025T211131Z.md) matches sourcef35e2441 and verifies716 CPU tests plus31 subtests,21 actual-HIP tests and all12 full-data calibration roles. Batch32 freezes experimentally at2543 training tokens/s and7.41GiB maximum calibration role upper memory. S002 gains actual scoped model/count/parity/update evidence at d80bf576. No persistent window or7800s parent starts: the foreign-UID fixture cannot write device attestation. Docker is a Podman alias, so portability remains pending without blocking primary work.

[Repair evidence](validation/soak-uid-repair-local-evidence.json) records192 passing host tests and verified public export. Both pinned images explicitly provide searchable0755 mount parents. Only the disposable UID fixture sets022 to publish host-readable proof and logs path modes before writing; restrictive077 export/recovery is reproduced and tested locally. Primary UID operations now gate preparation/candidate timing. Private/pytest symlink names no longer become public optional omissions; mandatory/public unsafe-member rules remain strict. These changes leave model/data/work/selection/caps/scientific gates unchanged.

Use [one corrected combined command](validation/soak-implementation.md). Changed image/source closure requires fresh compatible-source qualification before the previously unstarted joint/overhead/soak/scorer stages; earlier successful preflight remains accepted at its producer, not relabelled current. S004–S006/S011/S012 and original runtime tasks remain open. This concrete mandatory failure justifies one repair collection under the original handoff policy; it is not a new isolated size search or automatic fallback. No new CI wait or learning admission.

## Second target review: owner mapping and container-path repair

[The second collection](validation/soak-operator-20261025T222026Z.md) verifies source1f3bba26 and21 HIP passes, but one new CPU fixture fails and UID operations still block every soak leg. Diagnostics show /work0755/output0777, disproving the previous mode-only hypothesis. The fixture had changed --user without mapping the host owner to that numeric container UID; the exact filesystem rejection mechanism is not identified. Its CPU test also corrupted explicit input paths through sequential string replacement when TMPDIR was under /work/output.

The [new local repair](validation/soak-uid-map-local-evidence.json) passes194 host tests and verified public export. Fixture paths are explicit arguments; rootless Podman uses documented keep-id UID/GID owner mapping with real ownership assertions; public modes are explicit even under private file defaults; missing device evidence cannot mask launch/fatal resource failures. No host accounts/ACLs/ownership are changed. This scopes arbitrary numeric container users rather than foreign subordinate-host-ID filesystem access. Model/data/work/selection/global/phase/prompt rules are unchanged. All repairs finish before the [one combined follow-up](validation/soak-implementation.md). No current runtime task closes from host tests; original counts77/19/15 and scientific holds remain.

## Accepted primary target and next preparation — 2026-10-25

[The completed archive](validation/soak-operator-20261025T232558Z.md) now establishes current-source primary qualification, full104.5M training-batch16, both hour windows, seven-state restart, joint stress, operations and separate CPU scorer. Source d5c6de02 is unchanged by this reconciliation. Original task counts are87 complete/9 engineering pending/15 research held. The remaining engineering tasks are T011/T034/T041/T059/T060/T062/T073/T084/T086. S005 retains original queued native scope; S004/S006/S011/S012 close for applicable evidenced scope with independent Docker explicitly pending. Full008 carryovers remain unclosed.

Do not repeat the accepted soak or size/batch screens. The next ready work is [prospective longer-run preparation](validation/hardware-readiness-next.md), including remaining host-side fault tests, a new finite recipe/budget/storage design and complete-schedule cost/admission. T086 remains open until that costed command/archive handoff exists. The accepted7.36% between-leg decline and7.546/8GiB transient peak are watchpoints for that preparation, not an automatic additional HIP loop. No new target command or learning/final/multi-day execution is requested now.

## Current extension: measure learning over a finite longer run

The user now authorizes preparation of sequence-learning evaluation and further-training estimates. [Learning-pilot plan](validation/learning-pilot-plan.md) freezes the question, existing-data warning, proposed larger generic corpus, full-checkpoint control roster, prospective48-hour inclusive/24-hour learner ceiling, hidden-isolated independent scoring, conservative cost gates and one30-minute learner/restart plumbing test with no learning-efficiency criterion. Actual preparation/calibration may take hours; its whole-schedule cost must be measured. At the planning checkpoint this design was documentation-only and runtime remained d5c6de02. The implementation handoff below supersedes that preparation status while retaining accepted hardware evidence. L003–L015 distinguish implementation/validation/execution tasks. T086 and all unexecuted original scientific studies remain open.


## Learning-pilot code/local handoff

The [implementation and one target command](validation/learning-pilot-implementation.md) now realize the [prospective learning plan](validation/learning-pilot-plan.md) through all unblocked code/local work. The separate [contract](contracts/learning-pilot.md) preserves old profiles/defaults and hardware evidence while adding finite strict32768-example preparation, actual-checkpoint full-denominator learning/control curves and conditional further-compute reports. L004 actual corpus feasibility, L012 measured-cost clause, L013 target review and L014/L015 subsequent admitted execution/analysis remain open. Pinned CPU/HIP fixtures are authored, not claimed run. T011 closes; new-source T110 reopens; original counts87/9/15 remain. Do not repeat the accepted size/batch/two-hour soak. No long run is automatically admitted.


### First pilot validation collection

The [review and repaired consolidated handoff](validation/learning-pilot-operator-20261026T124145Z.md) retain23 HIP passes and750 CPU passes/one cost-probe failure before any corpus or session work. The scoped scoring-cost v2 separates six required full100-index positive paths and the retained4096-bit fuel stress, with explicit unknown/mismatch failures and complete diagnostics. It changes no learner/scientific scorer/resource/clock policy.221 local host tests pass; corrected target qualification, corpus feasibility and full costs remain pending before L013 can admit a long study.

### Full-clock long-run admission delivered (2026-10-27)

The next-work J001–J005 implementation is now [delivered through the main CLI](validation/long-run-implementation.md),
with a [separate v3 profile/v2 authority contract](contracts/long-run-pilot.md).
Legacy profiles/readers retain their identities. Accepted preparation/calibration
is imported with explicit compatible-source/native/runtime closures; current
CPU/HIP/package/operations gates are combined into the admission invocation.
Its explicit approval can produce an exact decision, followed by separately
invoked phase6 full-state fresh learning. Actual target qualification (T110),
long execution (L014) and scientific review (L015) remain distinct.

### Completed pilot review

Actual execution L014 is complete; its detailed audit remains private. The next
strategy decision L015 is still pending. No original research comparison, default
adoption or final-exposure gate closes, and no new experiment starts automatically.
The public review utility can independently audit an operator-supplied archive.

### Admission evidence reconciled

Independent review of the supplied admission evidence now closes009:T086 and
009:T110 at their applicable scopes. Original task counts are89complete/7engineering
pending/15research held. Detailed audit metadata remain private. Supplemental
L014 execution is complete and L015 strategy decision remains open; independent
Docker and all original research/adoption/final restrictions are unchanged.

### Post-run diagnostic implementation

Implement [read-only pilot diagnostics](contracts/pilot-diagnostics.md) through the
main CLI. Deliver stdlib collection first, then verified saved-weight import,
bounded per-example/matched-conditioning/weight/activation probes, integrated
current-package CPU/HIP fixtures and one operator archive. Preserve the completed
study and qualified computational dependencies; no training, fresh corpus,
semantic reclassification, expired-attempt retry or strategy switch occurs.
