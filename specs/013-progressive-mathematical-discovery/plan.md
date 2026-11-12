# Implementation Plan: Progressive Mathematical Discovery

**Branch**: `013-progressive-mathematical-discovery` | **Date**: 2026-11-08 | **Spec**: [spec.md](spec.md)

**Input**: `specs/013-progressive-mathematical-discovery/spec.md`

**Status**: Phase 0 research and Phase 1 design complete; dependency-ordered [tasks.md](tasks.md) generated, including optional contingencies and [consolidated hardware validation](hardware-validation.md). Runtime implementation, feature acceptance, source qualification and research execution remain pending. This document does not launch a campaign.

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

**Performance Goals**: Progress or explicit stalled/unavailable state within 60 seconds; every training task <=86,400 active seconds including startup/online search/checkpoint/finalization. Confirmed capability gates use prospective +2 pp gain, >=95% completion and <=2 pp retention loss. Classify these separately from invariant correctness and binding execution caps; soft targets and bounded exploratory/efficiency routes are defined in the contingency contract. No throughput/accuracy speedup is promised. Preserve fixed 1,024-token, four-second fitting-service and 60-second portfolio budgets in the base evaluation profile.

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
- `contracts/contingencies.md`: target classes, sensitivity, failure alternatives and bounded diagnostic reservations.
- `checklists/requirements.md`: existing specification quality review.
- `tasks.md`: implementation/acceptance work by user story, followed by explicitly optional research and contingency execution.
- `hardware-validation.md`: consolidated qualification and scoped post-freeze acceptance, original-producer reuse and sustained-run evidence.

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

Implement typed training-source extraction and bounded macro expansion with independent semantics, library-guided search with expansion into the existing codec, checkpoint/library binding and matched no-library evaluation. Add FindStat-derived source admission and bounded terms-only combinatorial-short/v1 pilot; other database groups get suitability records. Build discovery dossiers, proof/novelty separation and transitive refutation quarantine.

Exit: at least one actual learned component is used in generation in acceptance, not merely counted in a report. Source and component negative cases and a refuted claim traverse the same installed path as research.

### F — Consolidated qualification and staged research

Run CPU/native, installed-wheel, isolation, both-engine portability where required by the existing foundation, actual HIP and shortened end-to-end acceptance; collect one complete bounded package. Only then execute the frozen research DAG, serially, with milestone stops and reserved continuation slots. Reuse compatible evidence. Research stage failure prevents its confirmed-expansion path, not completion of remaining software acceptance work. Independently valid baseline paths and pre-reserved diagnostic actions follow their own frozen rules; they never upgrade the failed result.

Tasks must include concrete work in A–F and acceptance for all five stages. No phase is replaced with a vague future-work placeholder. Commit coherent groups and push without waiting for CI jobs; fix failures identified during ordinary review.

## Contingency and target review

The numeric research defaults are not equally hard. Use the four classes in [contingencies.md](contracts/contingencies.md) and freeze a target registry before execution. Correctness/isolation/accounting and the user’s 24-hour cap stay hard. Capability confirmation, utility adoption and information-gathering actions have distinct evidence/action records. Reserve limited diagnostic work within existing totals, never by automatically adding another campaign. Perform outcome-blind sensitivity checks before treating a 2 pp effect as detectable. Add software counterexamples for an attractive-but-unconfirmed model, a cheaper noninferior model, a late learner, a failed seed, a sparse cohort and an attempted integrity waiver.

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

### G — Reusable full-corpus preparation

Use a separate versioned corpus cache and CPU-only canonical data-fetch command. Pin a shallow bare OEIS repository, enumerate the complete Git tree through native batch reads, and fetch all current b-file objects in one native `git lfs fetch` invocation per retry. Do not use `--all` (history), thousands of subprocess batches, or per-sequence HTTP. Native transfer retries and bounded command retries preserve LFS objects. Preflight declared object bytes, disk and inodes; retain per-invocation elapsed costs and failure ZIPs.

Maintain a transactional per-entry SQLite inventory with source hashes, exact metadata, bounded indexed projections, full raw b-file locations, eligibility reasons and processing fingerprints. Reuse verified compatible stripped-cache values; conflicting releases remain explicit. Retain every entry in the inventory and expose eligible 100-term projections through the existing indexed-source contract. Never ingest names/formulas/programs into training targets. Retries process only unfinished or changed members, preserve successes and do not create a fresh experimental panel.

The full-corpus recovery policy `sha256-content-bounded-actual-size/v2` treats pointer sizes as metadata, never substitutes them for pinned SHA-256 content identity, and checks aggregate actual retained bytes before parsing. A bounded, unchanged, hash-matching object may carry an explicit declared-size warning. Missing/corrupt objects and exhausted resources remain blocking. Hash-verified disagreement between the main entry and b-file is an explicit `EXCLUDED` source disposition, never a choice of which values to train on. Complete preparation means every entry has a disposition and every object is verified; it does not mean every entry is eligible. Retain the existing indexed-corpus-entry/v1 cache binding because the exact projections of previously successful entries are unchanged; retry prior blocked entries and independently recheck all object fingerprints. The new verification policy and producer source are recorded in each result. This CPU producer change does not confer or transfer HIP authority.

Provide explicit cross-session reuse of a complete hash-bound exposure index with original input identities, producer source and spent costs. Re-audit retained inputs and require current-source hardware qualification independently. Full-corpus acquisition itself performs no GPU work, panel scoring or training.

### H — Independent OEIS pilot observer

Keep the first pilot's existing admission/selection/training policies fixed. Add `tools/oeis_diagnostics/` as a separately invoked analysis tool, never imported by production roles. It runs against a retained immutable CPU image with no build, GPU, network or experiment write mounts. Its own tool/plan/image/input identities are recorded separately. Existing source-closure rules are unchanged; these observer-only files and their own tests are outside the production qualification closure. This is not an exemption for runtime changes or authority to relabel older HIP evidence.

Before search, freeze up to 32 groups from the canonical OEIS training pool by deterministic outcome-blind ordering, one member per group, with unavailable members retained. Bind the corpus revision/raw hashes and exactly the next 100 source indices after the admitted 100. After original yield publication, reconcile every committed admission against that publication and report concrete/parameterised token lengths, literal digits/bits, branch/direct-index-equality indicators, and separate reference/native continuation results. Additional truth never enters a learner, selector or automatic refutation gate. Bound CPU work and preserve partial diagnostics. Future adaptive use of these diagnostic results requires declaring their exposure and a revised study.


### Registered-data handoff follow-through (2026-11-11)

A new preparation session may reuse a complete, audited registration session and
its reviewed immutable exposure index. A bounded CPU/native census measures
support and independently reserved panel supply before model outcomes. Synthetic
sensitivity cases are isolated and collected independently. Neither data reuse
nor the census transfers historical HIP authority or publishes a scientific freeze.
The operator preference is one unattended preparation-to-research invocation when
preparation may exceed 30 minutes; `--run-after-prepare` may transition only after
canonical freeze and campaign-bound acceptance, with final assessment separate.
T157 remains the explicit implementation dependency for the concrete registered-
data campaign builder, calibration connection and prospectively selected tuning
scope; the intermediate census is not completion of that workflow.

### Prospective census revision after unscored preparation (2026-11-12)

Retain preparation v2 and all its costs/exclusions. Explicit v3 data reuse is
limited to the reviewed original producer and an unscored, pre-HIP preparation.
The new session owns the remainder of the original six-hour census allowance;
one donor-derived durable owner prevents another directory resetting it. Reuse
all 40,960 original native outcomes and their exact input artifacts. Do not
repeat the tiny rehearsal, corpus acquisition or exposure import. Current-source
model qualification/calibration/acceptance are still required independently.

Commit 32,768 additional evaluation draws, alternating literal variants of
common initial gradient-support programs and binary compositions of their seen
pure bigint subexpressions. Use [-65,536,65,536] arithmetic literals, [1,4] control
counts and [1,99] branch thresholds, with unchanged native/reference, encoding,
exposure and finite-window validation. Every attempted counter remains charged;
there is no redraw-until-quota policy. This explicitly changes the evaluation
sampling prior, not the synthetic training distribution or primary endpoint.

Permit structure sharing for parameter/retention variants across panels while
retaining structural statistical clusters within each panel. Exact values,
shift/prefix windows, declared relationships and composition relations remain
indivisible. Primary composition/unseen structural grouping is unchanged.
Screening v2 preserves 32 tasks/stratum and at least 16 groups, with at most two
members each, selecting one from each group before second members. Singleton
OEIS groups therefore remain eligible without invented relationships. Full
confirmation targets and scientific gates remain unchanged.

Support projection v2 retains the original training rows/shards/probes and
chooses an eight-update-aligned interval separately for each effective batch.
Each interval must both cover 25M tokens without exhausting support and guarantee
a refresh before that milestone under the conservative maximum target size.
The map is part of learning compatibility and follows exact trial materialization
through calibration and execution. Reusing a fixed update interval across batches
would confound the intended token-scale intervention.

Exact original OEIS values outside the 4096-bit execution profile receive a
recorded INELIGIBLE disposition; malformed values/indices remain integrity errors.
No source is padded, truncated or substituted. Publish cumulative and new draw
counts, rejection reasons, original producer references and source dispositions.
An actual workstation supply deficit still blocks model work honestly.

### Training-only evaluation timing correction (2026-11-12)

The round-002 run published complete materials and passed its 48 original selected
HIP cases. All fourteen timing subprocesses then rejected terminal decoder token
caps as incomplete measurement. The supplementary diagnostic projections show
`token_cap_without_eos` at exactly 1,024 tokens, with all selected eight-attempt
rosters terminal. This is measured generation work but remains censored science.

Evaluation-case v2 verifies the selected portfolio clocks and attempt details,
allowing only the registered terminal token cap. Missing attempts, other decoder
failures, incomplete native fitting and incomplete selected-candidate scoring
still fail. The global four-method clock inventory may include unselected methods;
their absence cannot invalidate an explicitly one-method timing request. Historical
v1 case semantics remain unchanged. Coverage and phase timing are exported before
the case gate, including on failure.

A missing candidate did not exercise fitting or full-horizon scoring. For resource
estimates, reserve the unexercised four-second fit/raw services and thirty seconds
per native/reference scoring executor, separately from measured time, before the
existing fourfold margin. These are conservative reserves, not measured execution
or scientific results. Future direct service calibration may reduce uncertainty
only through original measurements under a prospectively declared policy.

The measured generation alone already contradicts the current 250-second screening
allocations and 4,200-second full-method generation roles. The fourteen timing
records must not be used to waive feasibility, shorten panels or relax scientific
completion gates. T089 remains pending a prospectively feasible design; another
hardware run is unnecessary merely to rediscover this deficit. Original census,
materials, exposure, model timing and qualification remain intact. A production
source correction does not transfer the old HIP authority to the new source.

The real 376-role roster is 6,162,183 bytes. Its readers share a 64 MiB document
bound, including execution-contract and reservation verification; short fixture
identifiers must not hide this integration requirement.

### Training-only service continuation and runtime review

Before revising the scientific schedule, `measure-services` reads the original
round-002 census, support, journal and fourteen generation/scoring publications.
It does not repeat generation, native draws, sensitivity, rehearsal or selected
HIP qualification. It rejects changed protected generation/service code, foreign
originals, and preparation journals containing later research or assessment work.
Historical measurements remain planning evidence for their original source.

Select four distinct native-verified gradient-training rows, excluding training
probes, by target length, visible integer size, full-horizon integer size and source
length, with deterministic identity ordering. Run two cold repetitions of canonical
fitting, independent visible admission and native/reference 100-term scoring.
Known training programs establish service timings, never learned capability.
Fresh native owners isolate the three services and each case runs in its own
process. Independent cases continue after failures; failed/interrupted originals
are retained, not automatically repeated. Reinvocation reuses checked completed
publications and consumes only the original remaining allowance.

The CPU-only installed role uses the current qualification image. Eight cases each
have 120 seconds, including four-second fitting/admission deadlines and a 65-second
scoring envelope around the existing two thirty-second executors. Its 1,200-second
inner and 1,320-second host allowances leave startup/collection/cleanup slack and
fit inside the original failed timing stage's remaining four-hour reservation.
A donor-derived owner claim prevents moving the continuation to reset allowances.
Image construction has its existing separate allowance. Case logs, original first
failures, process cleanup, service measurements and budget review enter the support
ZIP even when measurement is incomplete; private training rows remain local.

The report is explicitly non-executable. It compares existing task allocations
with measured generation, positive-witness scoring and unexercised service reserves,
then shows the fourfold margin and finalization reserve separately. Roles exceeding
24 hours need partitioning or a prospectively revised design. Screening, panel,
method, attempt and token budgets do not change automatically.

Expected runtime is distinct from reservations. Report unpadded generation/scoring
projections by stage, cumulative-target-correct incremental training tokens, and
explicitly missing costs. All eligible training in the selected round totals 364M
tokens; early stopping may reduce that total but has no measured probability yet.
Qualification elapsed time bundles initialization, updates, checkpointing and other
work and is not representative tokens/second. Leave expected training and total
elapsed time unknown until representative HIP update throughput, probes, checkpoint
publication/restore and setup are measured separately. Neither the fourfold safety
margin nor a hard cap is an ETA. Review the measurements with the operator before
changing scientific budgets or freezing a new schedule.

### Reference training timing and short-study sizing

The operator selected staged measurement before committing the full tuning budget
and asked whether 500k–1M screening and 2M confirmation can be informative. Add
`measure-training`, consuming the completed service diagnostic and unchanged
native training support. It builds/reuses only the current HIP image and measures
the reference recipe at batch 16 with the canonical FP32 update path. Sixteen
fixed coverage batches include initial familiar/frontier and later frontier data;
two additional stress batches cover longest targets and largest visible integers.
Exclude probes from gradients, record every debit before work, and report cold
initialization/Adam allocation separately from steady update speed. This fixed
coverage sample is not a prediction of learned curriculum allocation.

Measure the canonical training probes, full progressive checkpoint capture/save,
five verified checkpoint reads and fresh-model restore. The present production
learner checkpoints every update and reads complete state in its loop restore,
completion check, reservation, commit check and save verification. Omitting those
costs would materially understate training time. Report these component projections
separately from GPU update-only throughput; growing journal/quarantine history,
retention scans and sustained filesystem pressure remain unmeasured. This short
diagnostic cannot establish full-learner throughput or sustained execution.

The installed role verifies current source/image and actual HIP with the existing
numerical startup policy and exclusive GPU owner. A finite 1,800-second inner /
1,920-second host allowance includes cleanup; image construction remains separate.
A donor-derived ownership claim prevents sibling resets. Successful stages reuse
original checked publications; failed or interrupted model work is retained and
not automatically restarted. Original progress, debits, hardware facts, failures,
checkpoint hash inventory and partial reports enter the existing ZIP infrastructure;
training rows and checkpoint tensors stay local. This is diagnostic timing, never
qualification, a scientific checkpoint or a frozen campaign.

Project costs at 500k, 1M, 2M, 4M and 25M tokens without safety multipliers. Also
report the original first-refresh bounds and an initial-bin token-size projection.
Round-002's first-refresh upper bound is approximately 8.2M tokens; merely stopping
its current policy at 2M cannot guarantee that refreshed training ever differs
from static. A shorter diversity experiment requires a prospectively revised
refresh horizon that exposes the intervention. Learning-rate warmup/decay must
likewise be chosen for the intended horizon, or explicitly preserved when studying
early checkpoints of a longer schedule. These are scientific design decisions,
not automatic timeout adjustments.

Useful short-run sizing requires a separate registered learning-curve pilot:
measure executable development capability and retention at declared early cuts,
check whether gains/rankings stabilize or remain near the floor, and preserve all
data exposure. Teacher-forced loss and this 18-update timing diagnostic cannot
establish a useful training horizon. Repeated development scoring has a substantial
fixed cost, so token reductions alone do not produce proportional wall-time savings.
A short independent two-seed confirmation can support a claim at its own horizon;
it cannot establish superiority at 25M/100M/400M. Keep its unused confirmation panel
and the final assessment isolated. Decide the finite pilot/study budgets after
timing review; existing scientific rosters and budgets remain unchanged.

## Post-measurement persistence correction

Reuse the post-012 research `hourly/v1` cadence through the shared persistence
policy. New research learner packages explicitly bind that policy into their
compatibility identity; historical packages retain `per-update/v1`. Complete
model/Adam/RNG/controller cuts are saved at initialization, hourly complete-update
boundaries and clean pause/task completion. The existing progressive budget
journal remains durable before every update. On a crash, restore the last full
model/policy and count all later charges as discarded work, without adding those
charges twice. Unknown elapsed intervals retain existing conservative supervisor
rules. Live progress is informational and cannot authorize evaluation/continuation.

The owned learner uses verified checkpoint metadata for ordinary update counters;
full tensor verification remains mandatory on publication and restore. Hourly
windows retain the union of consumed target ancestry, exact per-task diversity
multiplicities and the latest 64 update metrics. Retention runs when the published
checkpoint changes. The budget reader indexes its own verified appends; any
external file change invalidates that index and forces full chain verification.
Real optimizer, failed-publication, failed-update, rollback, quarantine and installed
pause/resume/successor tests cover the change. Actual HIP throughput and the
scientific choice of shorter study horizons remain separate hardware/research work.

Shared journal storage for a new research freeze reserves 2 GiB, sufficient to
avoid the former 64 MiB implementation stop during ordinary multi-task training.
The runtime binds this to the frozen campaign's allowance; old campaigns keep
their original limits. Replay reconciliation reads only the unseen indexed suffix.
Progress distinguishes completed optimizer work from recoverable snapshot exposure;
cumulative loss/token counts survive pruning of recent per-update metrics.

Resource policy v2 prices hourly snapshots from the bounded task duration, plus
initial/final and crossed milestone cuts, instead of multiplying checkpoint cost
by every update. It retains the existing fourfold feasibility margin and scientific
caps. Historical v1 remains readable with its original per-update semantics. Runtime
reports preserve the historical comparator and add explicitly unmeasured hourly
component projections; they do not rename old HIP results as current validation.

For a prospective shorter study, `prepared_support.recadence` derives an immutable
schedule from the original rows and complete trajectory horizon. It preserves
all shards, probe exclusions, ordering, preparation omissions and provenance;
returns explicit first-look deficits; and never uses model scores. All successive
screening looks must share that schedule. A 500k/1M screening or 2M confirmation
choice still requires its own prospective design; this repair selects none of them.
