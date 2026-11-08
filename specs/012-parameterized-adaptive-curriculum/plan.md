# Implementation Plan: Parameterized Targets and Adaptive Curriculum

**Branch**: `012-parameterized-adaptive-curriculum` | **Date**: 2026-11-04
**Spec**: [spec.md](spec.md)
**Input**: `specs/012-parameterized-adaptive-curriculum/spec.md`
**Status**: A–D, F001–F015 and all mandatory Phase E software/evidence tasks complete. Original -004 acceptance and the separate focused actual HIP crash/debit result verify PASS; seven E tasks are checked and only optional T073 research analysis is deferred. Producer source/image identities remain separately scoped; see [closeout](validation/closeout.md).

## Summary

Implement a controlled parameterized-target and adaptive-family/review experiment
on the merged 011 foundation. First complete the inherited structural-only scope
and old-control audit. Register a new PAC policy, reuse paired targets/native fitting
and the installed learner. Add the named Graves EXP3.S kernel, block-attributable
training-only reward, bounded per-example Leitner review and exact whole-state
resume. The temporary operator wrapper composes canonical endpoints and
packages evidence in one normal session. It must not become a second learner,
fitter, acceptance authority or general workflow framework.

The default screen is one seed and 25M target tokens per new B/C arm. An optional
bandit-only D is supported but off by default; its full budget must be frozen before
outcomes. C–B alone measures the package, not the isolated review contribution.
Several hours of training are acceptable. Actual performance and quota readiness remain measured
gates; see [research](research.md), [experiment](contracts/experiment.md),
[curriculum](contracts/curriculum.md) and [execution](contracts/execution.md).

## Technical Context

**Language/Version**: Existing Python >=3.11 project; stdlib-only host orchestration.
**Primary Dependencies**: Existing pinned PyTorch/ROCm model image, native bigint
and Wasmtime execution, parser/codec, canonical container and sealed-evidence APIs.
No new host package, service, driver or external scheduling dependency.
**Storage**: Existing immutable manifests/JSON ledgers, private prepared shards,
full tensor snapshots and verified report archives under workspace run-state.
**Testing**: pytest unit/contract/integration suites, independent arithmetic and
policy fixtures, installed tiny real-tensor CPU path, existing Docker/Podman
construction/portability gates, and one consolidated actual-HIP operator session.
**Target Platform**: NixOS workstation, AMD Ryzen AI Max+ 395, 128 GB shared RAM;
explicit Docker/Podman engine via portable CLI; CPU workers and one serial learner.
**Project Type**: Research CLI/container pipeline, not a hosted service.
**Performance Goals**: No gain promised; bound diagnostic overhead and preserve
hourly snapshots/whole-run progress. Use real representative calibration and generous
finite time envelopes, including a 12-hour learner planning floor and 24-hour
preparation floor; full rules are in execution contract.
**Constraints**: Exact integer semantics; no hidden evaluation feedback; finite
resource graph; installed full-state resume; historical contracts unchanged; no
HIP work in this design pass; no arbitrary inherited global time ceiling.
**Scale/Scope**: 32 training families/32,768 gradient rows, bounded 256-example
probe panel, 128 structural tasks/23 families under the explicit v2 amendment (historical v1: 27 families), 128 review anchors, bounded 4,096-value reward
reservoir and two new arms by default. Optional fresh literal control or bandit-only
D only if frozen before execution. No online generation,
extra seeds, support expansion, RL/replay teaching or assessment adoption.

## Constitution Check

Pre-research and post-design checks both PASS at design level; execution evidence
remains outstanding. No constitutional amendment or exception is requested.

| Gate | Design evidence / runtime obligation |
| --- | --- |
| I: exact authority/numerical semantics | Existing paired-source grounding, integer profile and independent reference checks; no numeric-law change |
| II: supported grammar/independent acceptance | Existing safe fallback and fitter limits; independent control-taint/grounding counterexamples; final-source execution |
| III: bounded recoverable operation | Full nested wall/disk/work graph, bounded queues/caches/logs; atomic full-state snapshots and fault tests |
| IV: feasibility/reproducibility | Pinned containers, source/image identities, no-Lt policy; actual HIP updates and exact resume before research |
| V: strict provenance/isolation | Fresh random B/C, original mechanical pool, train-only probes, sealed structural/assessment custodian; control audit |
| VI: scoped evidence | Finite success, unresolved/ambiguous classes and censoring remain distinct; one-seed results exploratory |
| Workflow | Freeze hypotheses/metrics/limits/optional arms before scores; equation and queue counterexamples with critical work; distinguish published kernel from project adaptations; no positive-result software gate |

## Project Structure

### Documentation (this feature)

- `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`
- `contracts/experiment.md`, `contracts/curriculum.md`, `contracts/execution.md`
- `reliability-review.md`, `checklists/requirements.md`
- `tasks.md`: 74 dependency-ordered tasks across A–E; 66 local implementation tasks and seven E evidence tasks complete; conditional T073 research analysis deferred.
- `validation/` for the later temporary wrapper, frozen handoff and actual evidence.

### Source Code (repository root)

Reuse these existing boundaries; final file splits should remain small:

- `src/oeis_learn/generalization/`: execution/contracts, target proofs, budgets,
  checkpoint/state and new pure EXP3.S kernel, reward scaler and anchor queue records.
  Do not reuse the legacy class solely because its name contains Exp3S; reject old
  `visit-scheduler/v1` state at the new authority boundary.
- `src/oeis_learn/pipeline_validation/generalization_{preparation,training,runtime,persistence,execution}.py`:
  family sidecar, probe preparation, installed batches/updates, role and resume binding.
- Existing `pipeline_validation` resources/supervision/disk/diagnostics/finalization
  modules: extend coverage/registered policy, do not duplicate controllers.
- `scripts/foundation_run.py` and `scripts/generalization_inputs.py`: canonical
  entrypoints through their existing dispatch/assembly modules.
- `specs/011-controlled-generalization/validation/prospective_evaluation_policy.py`,
  `prospective_evaluation.py` and post-pilot analysis helpers: reuse with an explicit
  new structural scope/adapter, preserving old 256-task guards and evidence.
- `specs/012-parameterized-adaptive-curriculum/validation/`: temporary serial helper,
  calibration/contract fixtures and human handoff, not production algorithms.
- `tests/unit`, `tests/contract`, `tests/integration`: policy, isolation, installed
  path, failure exports and portability regressions near existing related suites.

**Structure Decision**: extend established policy and installed-runtime boundaries.
No parallel training stack, generic scheduler framework or broad infrastructure
rewrite. Exact new schema/recipe dispatch must be canonical before wrapper wiring.

## Implementation phases and model allocation

Exactly **two** critical phases are reserved for a more powerful model. Keep only
invariant-bearing work there. Routine integration is in D; correctness-critical
state/provenance integration is explicitly in C. Counterexample tests belong with
their critical change; broad suite execution does not consume a separate critical
phase. Commit and push each coherent subheader task group during implementation.

### A — Routine contracts and scaffolding

Create closed record/schema shells, independent equation/queue fixtures, proposed
command surfaces and deterministic inventories. Include selected-block/probability,
reward-scaling reservoir and per-anchor state; keep the optional D manifest explicit.
Map every reliability row to task ownership. Establish legacy regression
coverage and prepare the old-control audit inputs. Do not implement scheduler,
quota semantics, deadline arithmetic or transaction policy in scaffolding.

Exit: closed interfaces/fixtures are ready for B; old studies remain readable;
no acceptance or hardware completion is claimed.

### B — Critical algorithms and state invariants

Implement together: structural scope/denominators and immutable draw/exclusion
identity; registered PAC comparison/control policy; Graves fixed-share/importance
update and exact first-update indexing, signed
block reward and bounded quantile/reservoir policy; per-anchor Leitner acquisition/
recall/incidental exposure and service bounds; event/state transitions, missing
feedback and discarded-work rules; full nested budget and peak-live-storage arithmetic. Reuse target/fitter policy;
only necessary invariant changes belong here. Include independently expected
counterexamples for every new mathematical/admission/state rule.

Exit: pure policies are deterministic and tested for leakage, infeasibility,
censoring, selection-probability floors, reward sign/scaling, review fairness,
identity mismatch and recovery transitions. No runtime
adapter may reinterpret these rules later. FR-001–020/027–028 and resource-policy
parts of FR-021–023 are owned here.

### C — Critical installed execution and provenance integration

Bind prepared family metadata and probe roles to the real batch stream; route actual
matched pre/block/post reward observations and real review optimizer updates;
preserve RNG/model modes and exact credit attribution;
atomically bind weights/optimizer/debits with pending inner-block progress,
selection-time probabilities, scaler history/RNG, kernel counters and anchor grades.
Integrate
new recipe authority, source/image/environment compatibility and nested effective
budgets through canonical handlers. Preserve hourly snapshot and terminal publication
semantics. Use installed tiny real tensors for pause/resume/crash counterexamples;
actual HIP proof remains phase E.

Exit: local installed path demonstrates exact clean resume and honest discarded
crash accounting, without a parallel learner or per-update full snapshots. All
critical source changes are complete before broad delivery and handoff.
FR-008–026/030 correctness boundaries are owned here.

### D — Routine wiring, diagnostics and complete local validation

Compose the temporary serial helper and existing canonical commands. Add scoped
post-pilot case analysis, family/cost reports, all terminal packer projections,
control/feasibility/calibration summaries and exact operator instructions. Validate
all rows of the reliability matrix, Docker/Podman command construction, stdlib host
entry, old/new schemas, the complete budget graph and success/failure recovery using
actual installed handlers where dependencies permit. Run the relevant broad
non-HIP suite once after changes stabilize, then only tests justified by new changes.

Exit: one copy-pastable command is verified against real help/fixtures; every missing
workstation artifact is named before launch; no hardware is required to discover a
packaging/schema/deadline graph bug. FR-024–031 and all NFR delivery obligations.

### E — One consolidated operator validation and optional research session

Supply one invocation that performs full actual-input feasibility, required
qualification, representative calibration, tiny HIP acceptance and automatic export.
If the plan already authorizes research, continue serially to retained structural
baselines, new arms and final analysis without an intermediate manual command.
Otherwise acceptance ends with a report and a ready frozen research plan. Do not
rerun compatible successful stages. An unexpected failure produces a minimal rerun
boundary and evidence; one invocation is a normal-path objective, not a promise of
zero hardware failures. Actual T180 replacement evidence stays unchecked until run.

Exit: verified evidence separately states software readiness, completed scientific
scope and remaining limitations. No automatic additional campaign or winner.

## Requirement coverage and validation strategy

| Requirement group | Primary proof | Phases |
| --- | --- | --- |
| FR-001–007 | Old P3 preserved; new arms/factors; control mismatch; all original exclusions and exact structural quotas | A/B/C/D, actual admission E |
| FR-008–011 | Independent grounding/fallback and probe/gradient/evaluation role isolation | B/C, native E |
| FR-012–018 | Published-equation numeric fixtures; all 32 queue-cycle offsets; real charged review; block-attribution and censored-feedback cases | B/C/D, actual path E |
| FR-019–023 | Transaction fault matrix, model+policy/scaler/RNG/pending-block equality, source/image compatibility, deadline/storage graph | B/C/D, actual device E |
| FR-024–026 | First-error preservation, stage reuse, real packer/verifier and read-only recovery | C/D/E |
| FR-027–031 | Denominators/censoring, deterministic cases, new acceptance authority and scope gating | B/D/E |
| NFR-001–008; SC-001–010 | Phase boundaries, reliability matrix, measured budgets, one normal invocation and honest cost/report claims | All; hardware claims only E |

## Risks and explicit gates

Probe availability against exact duplicates and final-evaluation exclusions remains
unmeasured; fail before training if quotas are impossible. Parameterization can be
heavily diluted by control-sensitive fallback; report coverage before launching.
Adaptive feedback may cost more than it saves; calibrate two two-example probe
evaluations per bandit block, anchor grading and all common panels, charge them and
retain a fixed B. Proxy progress can reward relearning or miss cross-family transfer;
report raw/scaled rewards and independent retention. Target recall is not semantic
mastery. No published regret or spacing claim is inherited by the hybrid wrapper.
Historical A may fail matching; default to a labeled reference unless fresh A is
frozen. New source invalidates affected 011 qualification, while old private data
and evidence may still be reusable by identity. Inspect every legacy timeout cap;
large outer allowances do not fix short inner ones. No elapsed-time guarantee is
possible before actual calibration. None of these requires another prior chat now.

## Research revision validation

The primary-source review supersedes the custom heuristic draft. Recheck all
requirements and document links, calculate the three independent kernel examples,
and enumerate all 32 service-cycle offsets before task generation. These are
planning checks, not production tests or HIP qualification. Implementation must
add the independent tests to the real code path in phases B/C and use phase D for
combined regression execution. The existing relaxed floors, bottom-up deadline
arithmetic, one-hour finalization reserve and two critical phases are unchanged.

Planning checks executed on 2026-11-04: all three independent kernel fixtures
matched to 1e-12; an independent queue simulation ran 8,192 updates for each of
32 cycle offsets with 128 simultaneously due anchors repeatedly returned to the
minimum interval. Maximum observed due delay was 511 updates, within the stated
512 bound; every 512-update service window contained 128 opportunities. This checks
the documented arithmetic, not production behavior or every future implementation
fault. Spec prerequisites, local links, 31 FR / 8 NFR / 10 SC numbering, template
checks and whitespace checks passed. The two focused design reviews found and
resolved observation-count and missing resume-state ambiguities. No HIP or runtime
code was executed/changed and no CI wait was performed.


## Phase D completion and stop boundary

The temporary prepare/run/resume/recover helper composes canonical installed roles;
source/helper identities remain separate. Structural admission gates research,
qualification/calibration/actual HIP acceptance precede optional learners, and all
normal, failed, interrupted and incomplete scopes have independent archive checks.
The public report carries quota/representation/allocation/cost/case evidence with
private state and truth omitted. The ledger reconciles every FR/NFR/SC and all 18
reliability rows against the final 640-test/2-subtest CPU/contract selection.

No HIP or engine operation was executed in A–D and CI was not awaited. Phase E
requires the original private roots, a usable explicit engine and current pinned
images, compatible actual AMD/ROCm device access, sufficient peak disk capacity,
full original native quotas and representative calibration. The quickstart's single
operator invocation is ready for that session; research, fresh A/D and any later
seeds remain separate frozen choices. Historical 011 failures/tasks are unchanged.


### 2026-11-05 structural feasibility amendment

The operator authorized repair after actual post-exclusion quota failure. New runs
use the explicitly versioned 128-task/23-family population (68 certified/15 families)
in [the amendment](validation/structural-amendment.md). It freezes revised quotas from conservative source capacity
before model outcomes, preserving all 2,288 draws and every original exclusion.
Historical 27/18-family scope stays readable and failed; training, numeric law,
inference budgets and the 65/68 threshold remain unchanged. No actual new native/HIP
acceptance or research success is claimed by this implementation.


### Repair validation boundary

The first operator attempt supplied actual original-input availability and failed
admission before qualification/calibration/HIP acceptance. Its evidence remains
failed. Six post-D follow-ups freeze the revised population, retain every exclusion,
bind its identity through all reports, preserve nested causes across negative CLI
exits, and test old-run read-only recovery. The capacity proof covers worst-case
candidate-source duplication for the reported counts; new native/device proof is
still required. Pull the repair and use the fresh `-002` quickstart output, without
resuming the old scientific plan. No HIP or research run is launched by these fixes.
