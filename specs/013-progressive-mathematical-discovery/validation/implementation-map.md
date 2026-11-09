# Spec 013 implementation map

Scope: T001–T059 software/specification tasks, with T018 acceptance pending. Read spec.md, plan.md, tasks.md, research.md, data-model.md,
quickstart.md, hardware-validation.md, planning-review.md, all five contracts,
the requirements checklist and adopted constitution in full before implementation.
Requirements checklist: 16 checked, zero unchecked. No extension hooks registered.

## Reused owners and preserved guards

| Boundary | Existing owner | 013 adapter |
| --- | --- | --- |
| Portable host | scripts/foundation_run.py → foundation_local.py::parser/main | progressive_cli.py → progressive_commands.py; stdlib host |
| Installed role | generalization_roles.py::invoke / project_and_invoke → generalization_role.py | progressive_roles.py; reuse process supervision and engine isolation |
| Real learning | generalization_training.py::run_torch → torch_update → rl/sft_trainer.py::supervised_step | progressive_training.PolicyUpdate → shared logical_update; no second learner |
| Full checkpoint | fresh/bigint_state.py::GeneralizationCheckpointStore.save/load | later progressive state adapter, preserving atomic generations |
| Actual checkpoint evaluation | generalization_evaluation.py::load_retained | progressive/baselines.py; no invented checkpoint weights |
| Native and reference | synthesis/client.py, reference.py, reference_parser.py; decoder/bigint_codec.py | progressive/coverage.py with independent witness checks |
| Qualification | generalization_qualification_io.py::produce/consume_original/current_facts | discovery runtime binds original producers and exact scope; actual new qualification later |
| Evidence | generalization evidence/reporting and bundle.py; supervision.py::launch/reclaim | progressive_evidence.py; full independent bundle reconciliation T079–T080 |
| Source closure | experiments/source_snapshot.py → pipeline_validation/closures.py | include all 013 code/tests/contracts/assets; no new exclusions |

Preserve generic 21,600-second guards in generalization_training.py and
generalization_roles.py, the PAC envelope validator, PAC fresh-random/no-replay
checks in run_torch, execution.py::_pac_policy's fixed 25M policy, mechanical
replay_admission.py's oracle/shorter-target restrictions and legacy interval
requirements. Discovery authority uses its own versions. The existing
rl/progressive.py and legacy discovery commands are unrelated and unchanged.

Proposed new files are the progressive package modules and
pipeline_validation/progressive_{cli,commands,roles,preparation,evidence}.py listed
by T002–T023. Canonical seams above resolve; generalization_bundle.py does not
exist: the actual shared archive owner is pipeline_validation/bundle.py.

## Validation

Source/test implementation is complete for T001–T017 and T019–T059; T053 now supplies independently recomputed third-seed routing. T018 now has mandatory Docker/Podman CI acceptance and remains unchecked pending its first successful receipt. The initial local validation environment has no Docker or Podman binary
and no packaged native worker or HIP device. Local CPU/stdlib checks cannot
establish container/native/HIP qualification. No research or HIP training is launched.


## Delivered boundaries

- T001–T005: reuse/guard inventory, model-free namespace, exact Optuna 4.5.0
  dependency and lock closure, installed schemas/policies/contracts, hand-checked
  diagnostic fixtures with separate public and private files. The pinned uv 0.8.22
  resolver adds six dependencies; no existing locked package version changes.
- T006–T013: closed envelopes/references, non-executable blueprints, finite task
  plans/DAG and storage graph, exact original producer matching, append-only
  exposure and finite-window/structure relations, typed projections, classified
  targets, durable reserve/commit/reconcile and boot-bound active clocks. Unknown
  supervisor termination consumes the unproven remaining task clock. Policy caps
  keep the 24h task limit and the non-additive 168h/24h scaling split.
- T014–T017: installed operation registry, canonical campaign accounting and task
  leases, existing watchdog/reclamation, conservative runtime authority, complete
  input assembly and immutable freeze, all 12 command argument surfaces, initial
  independent-status diagnostics and local original tracebacks. Working host
  commands: init/freeze/status. Commands owned by later tasks fail explicitly
  with exit 2. Full-state continuation intentionally rejects until T049/T050.
- T019–T023: independent metric/gate specifications, fixed evaluation profile,
  four-stratum census and separate public/truth publication, private witness audit,
  retained/random/earlier checkpoint adapters, deterministic conditioning controls,
  existing generic templates and bounded exact polynomial/linear-recurrence
  baselines with one shared portfolio and per-method measured costs.

Task plans exclude the campaign hash until issuance, avoiding a circular content
identity. The frozen campaign contains the complete immutable roster; issued tasks
must exactly match it and bind the final campaign hash. Learning compatibility
excludes task paths, elapsed entitlements and seed choice. Recipe model scope is
the canonical hash of resolved model fields excluding seed. Original qualification
must explicitly cover this model scope and the selected language; old unrelated
receipts do not automatically qualify discovery work.

`init` can consume a sealed `discovery-retained-inputs/v1` retained-inputs.json
with explicit named references at the original campaign root. It verifies supplied
bytes and reports all unresolved original inputs together. No search outside that
root or supplied references substitutes guessed workstation paths. The default
blueprint has capability+diversity selected and zero executable allocation until
measured preparation resolves and freezes actual caps.

The explicit `progressive_cli.CPU_TESTS` inventory is included in the existing
bigint-cpu workflow. It excludes the unrelated historical rl/progressive test
names; no source/test bytes are excluded from qualification closure. The later
T085 task still owns complete feature-wide CI/hardware inventories.

## Validation results and limits

Local interpreter: Python 3.12.14; actual model test uses torch 2.9.1+cpu.
The container pins and native compiler/runtime closure remain unchanged.

- Failing-first authority check: `PYTHONPATH=src .venv-progressive/bin/python -m
  pytest tests/contract/test_progressive_authority.py -q` initially fails collection
  with the missing contracts module; the implemented authority tests now pass.
- New increment tests use the explicit CPU_TESTS list, including real canonical
  encoder/decoder weights save/load and exact tensor comparisons, real subprocess
  hang cleanup, filesystem role projections, successful diagnostic freeze and
  Python `-S` isolation for both successful and blocked host paths.
- The existing `tests/contract/test_foundation_packaging.py` builds a real wheel,
  extracts it outside the source tree and validates the complete installed source
  closure including the copied 013 markdown contracts and JSON assets.
- Affected historical selection: generalization authority/roles/contracts/budgets/
  operational budgets/execution/qualification/exclusions/metrics/legacy/host,
  registration and foundation packaging: **137 passed** before final combined run.
- `uvx --from uv==0.8.22 uv lock --check`: PASS. Existing locked versions unchanged.
- `python3 -S scripts/foundation_run.py discovery init ...` and read-only `status`:
  mechanically complete; missing original campaign explicitly reported. All 12
  parser surfaces are tested; unimplemented endpoints do not contact an engine.
- `python3 -S scripts/foundation_run.py --engine docker validate --suite bigint-cpu
  --output run-state/discovery/013-t001-t023/cpu-native`: **BLOCKED**, exit 1,
  `docker CLI is unavailable; select an installed container engine`. Podman is
  also absent. T018 is not marked complete. The native witness integration test
  is skipped locally and becomes mandatory in the existing native-required image.
- T019 has **8 strict expected failures** solely because T024/T025 modules are
  absent. These are independent future behavior specifications, not claimed
  passing metrics/gates. Their strict markers must be removed when those owners
  are implemented; an unexpected pass fails the suite.
- Broad `test_progressive_*` globs also select old rl/progressive tests. Three
  failures reproduce identically on unmodified PR head 7829e13 in a detached
  worktree: two Wasmtime fallback TrapCode failures and the legacy tier-2 shaping
  rejection. They were not changed by this increment. Use the explicit new
  inventory to distinguish these from spec 013 tests.

No actual HIP, sustained run, retained D evaluation, training campaign, source
acquisition or scientific advancement was executed or inferred. T024 onward,
including the independent metric reducer, gate decisions, executable scoring and
readiness MVP publication, remain outside this requested increment. Original
historical evidence and qualification scope are unchanged.

Final combined increment + affected historical run: **206 passed, 1 skipped,
8 xfailed** (9.33 seconds). The skip is the actual packaged native-service witness
case; the eight expected failures are T019's future T024/T025 specifications.
The final run includes all 17 explicit CPU_TESTS files plus the historical selection
listed above, with foundation wheel extraction/source verification included once.
No container, native or HIP status is upgraded by these local results.

Final whitespace and new-source/test Ruff checks pass. No extension pre/post hooks
are registered. Research and later task boxes remain unchanged. This increment is
reviewable independently; a successful mandatory Docker/Podman CI receipt can complete T018; no workstation
handoff is required for this CPU/native task.


## CI follow-up — stale binding repair and T018 handoff

Reviewed the existing d013814 check logs without waiting for a new CI run. The
portable-engines Docker and Podman checks and native-qualification failed on the
same stale `current-dispositions.json` implementation hash. The separate
cpu-regressions check reports that same cause. Engine startup, private worker
mount/device isolation and reclamation had passed. This differs from the older
rl/progressive harness issues noted in the initial local report.

Refreshed only the current review-disposition package binding and its descriptive
revision. All 68 original review items, task states, historical producer records,
source/task hashes and `full_007_complete=false` remain unchanged. The actual
package implementation serializer is still used; no hashing or validation guard
was weakened. The original failing tests now pass locally, together with the
full disposition rejection/replay tests.

Added `scripts/qualify_progressive_foundation.py` to both portable-engine CI matrix
entries via the existing stdlib host and `container_stage` launcher. Each bounded
900-second CPU/native role uses the shared immutable qualification image, read-only
root, no network/socket/workspace mount, one owned evidence directory and actual
cleanup/inspection. `OEIS_REQUIRE_NATIVE=1` and `OEIS_REQUIRE_BIGINT_BUILD=1` are
mandatory. Reports and JUnit are uploaded as portable-engine artifacts.

The fixed acceptance roster covers all 18 progressive CPU test modules except
T019's explicitly deferred T024/T025 metric specifications, plus the affected
historical authority/roles/budget/qualification/metric/host/wheel regressions and
the two stale-binding regressions. Mandatory JUnit reconciliation requires every
requested module, the real native witness case and installed-wheel source closure
case. Empty, failed, errored, skipped (including xfail) or omitted mandatory cases
cannot produce PASS. The future metric specifications stay in the repository and
ordinary CPU inventory; they cannot be counted as current implementation acceptance.

T018 is now infrastructure-unblocked and **PENDING_CI**, rather than blocked on
this local environment. Its checkbox remains honest until the first complete
native CI receipt exists. No current-source PASS, HIP qualification or research
result is inferred from wiring a workflow. The requested follow-up is pushed
without waiting for CI.

Follow-up local validation: the complete progressive inventory plus affected
historical acceptance selection reports **216 passed, 1 native skip, 8 expected
future-metric failures**. The full related container/runtime/role, pipeline
contract, phase-10 and handoff modules separately report **100 passed, 1 actual
engine skip**. Both original stale-binding regressions pass. These two selections
overlap and their counts are not added together. The local Wasmtime installation
was aligned to the committed 48.0.0 lock for this follow-up; the repository lock
was not changed. Workflow YAML, embedded stdlib host syntax, Ruff and whitespace
checks pass. Actual Docker/Podman/native execution belongs to the new CI steps;
local skips do not satisfy them.

## T023 focused completion

The baseline-only follow-up adds a model-control loader for retained D, random,
compatible earlier weights and the shuffled/absent controls. Earlier weights
require an explicit resolved reference recipe with matching input/model/language
semantics; matching shapes alone are insufficient. All checkpoint paths still
use the existing verified weights loader. No new B/D campaign is required.

Conventional priors are an immutable, identity-bound policy: integer polynomial
coefficients, degree <=3, homogeneous integer recurrence order <=2 at indices
0..19, coefficient bits <=4096 and the existing 1,024-token codec ceiling.
Narrower bounds must be supplied when the shared portfolio is constructed and
cannot change between calls. Unsupported applicability and codec limits are
explicit. Zero recurrence, signed geometric recurrence and shifted polynomial
indices have independent exact-reference cases.

The drainable candidate iterator charges a single shared attempt/wall allowance,
records generation failures and late results, and starts each bounded fitting
window after generation. Per-method reports distinguish attempts, produced
candidates, generation/fitting costs and measured total wall time; fitting does
not count an existing candidate twice. Actual installed model generation, native
fitting and scorer publication remain T027.

Validation: the focused baseline/evaluation tests passed (25 cases). The complete
progressive CPU inventory plus historical acceptance roster passed 230 cases,
with one unavailable native-service skip and eight existing T024/T025 expected
failures. New tests first reproduced eight failures in the earlier baseline
implementation. Ruff and whitespace checks passed. The current disposition
implementation binding was refreshed using its original serializer; historical
dispositions and hardware/research qualification are unchanged.

## T024 paired censored metrics

Implemented `progressive/metrics.py` as an outcome-only reducer. It reports exact
counts, all-task yield/completion/censoring bounds and family macros per arm, seed
and stratum. Successful returned candidates can explicitly have incomplete
portfolios; cost completion is not accepted as scientific completion. An explicit
frozen task and seed roster fills absent results as unresolved. Diagnostic row-only
input must be a complete paired rectangle and is labeled `roster_supplied=false`;
installed evaluation must pass the independent frozen roster.

The primary endpoint is fixed at half composition plus half unseen-structure
family macro, with equal observed-seed weights. Parameter/OEIS metrics and
pessimistic familiar-panel retention losses are reported separately. All
fractions and bootstrap weighted sums are computed as exact rationals before
conversion to report numbers. The 10,000 replicates resample families jointly
across arms and seeds. Unlinked strata use independent multinomial draws. Linked
strata use multinomial draws of their union of dependency components, common
integer multiplicities and within-stratum normalization; empty-stratum draws have
a frozen 128-redraw cap and fail closed on exhaustion. Dependency components,
not aliases within a stratum, determine the 16-independent-group requirement.

The installed target policy and closed schema now bind these choices before
scoring, including nominal floor(2.5% B)/ceil(97.5% B) one-based indices and exact
integer arithmetic for the specified adjusted lower/upper order statistics.
Comparison counts above 500 are unsupported because alpha/K would be below one
empirical draw at B=10,000. Lower samples treat candidate unresolved as failure
and baseline unresolved as success; upper samples reverse those assumptions.
NO_MATERIAL_GAIN requires conservative adjusted upper <0.02 and confirmed primary
regression requires upper <0. The pre-existing verdict precedence remains
INVALID, INCOMPLETE, REGRESSION, successful registered claim, NO_MATERIAL_GAIN,
INCONCLUSIVE. Actual verdict/action records remain T025.

Missing primary strata, fewer than 16 independent groups or fewer than two paired
seeds yield incomplete support and no claim interval. Intervals are explicitly
conditional on observed seeds, with each seed effect and range reported. The
historical >=3-seed population uncertainty routine and its tests are unchanged.

Validation: 20 metric tests cover hand-computed denominators, unequal families,
fixed stratum/seed weights, both censoring directions, missing results/strata,
explicit incomplete successful portfolios, retention, sparse/dependent support,
malformed pairing, prospective quantiles and exact independently calculated
small linked-resampling fixtures. The initial run failed 16 missing-implementation
cases before implementation (the historical guard already passed). Final full
progressive plus historical acceptance inventory: **250 passed, one unavailable
native-service skip, five strict expected T025 failures**. Wheel/source-closure
and disposition-binding regressions passed; changed module lint and whitespace
checks passed. T024 metrics are mandatory in both-engine native CI acceptance;
only the moved T025 specifications remain deferred. No CI result, hardware
qualification or research outcome is claimed, and CI will not be awaited after
publishing.

## T025 decision boundary

Implemented closed discovery-decision/v1 records with outcome/roster/design/policy
bindings and independent recomputation. Frozen target evidence must register the
comparison design; threshold changes are rejected. Scientific verdict and action
are independently validated, including reservation checks. Efficiency requires
its registered route, measured total cost and noninferiority separately in both
primary strata with the registered joint dependency weights. Historical seed
interval rules remain unchanged.

Validation: initial complete progressive/historical roster 262 passed, one local
native-service skip and no expected failures. Additional production efficiency
coverage passes with the decision suite. All T025 cases are mandatory in both
engine CI; no CI or hardware qualification result is inferred locally.

## T026 synthetic sensitivity

Added a closed synthetic-only scenario runner for 0/2/5pp effects, unequal family
sizes, family-shared dependence, opposing seed shifts, 0/2/5/10% censoring and
frozen candidate/look counts including efficiency. It calls the actual paired
metric and decision reducers, reports false advancement, missed gain, incomplete
and inconclusive rates, and labels all sampling assumptions and conditional
uncertainty. Real confirmation outcomes and threshold overrides are rejected.
The 80% aspiration is explicitly scenario-specific. Four independent CPU tests
cover the default grid, paired task construction, reproducible report rates and
closed input rejection. No empirical power calibration or research is claimed.

## T027 installed generation and scoring

Added isolated installed generator/scorer handlers. Generation loads actual
checkpoint weights (or explicit random controls), uses the canonical synchronous
drain driver with eight-attempt/60s portfolios and bounded native fitting, and
seals candidate bytes before the scorer can access 100-term truth. Conventional
methods share their declared portfolio and use final-source native/reference
acceptance. Reports retain raw/rescue/selected outcomes, per-attempt token and
clock/queue details, total elapsed time, and scientific completion separately
from candidate success and cost completion.

Typed model projection now verifies and copies referenced recipe/weight bytes,
including nested inode checks against forbidden truth. Independent CPU tests
verify leakage rejection, actual payload copies and sealed-input rejection.
Native production integration includes real CPU sampling/checkpoint load plus
independent native/reference scoring, and conventional visible fitting followed
by hidden scoring; these tests are mandatory in the qualification image.
Local full progressive/historical roster: 270 passed, three native-service skips.
No native execution or actual HIP result is claimed on this host.

## T028 capability readiness publication

Added independent reconciliation of sealed cohort/exposure, witness coverage and
candidate/scoring artifact pairs, with every transfer stratum and baseline method,
fixed denominators, completion/censoring, historical exposure and measured costs.
Missing panels, controls or coverage remain named deficits; representation
classifications are separate from runtime/censoring outcomes. Installed generator
metadata can explicitly select a frozen random or compatible earlier control.
`publish_capability_inputs` joins the existing census/audit/evaluation outputs,
and the stdlib `discovery report` path publishes a bounded public readiness ZIP.
This capability-only export has no independent full-bundle verification authority
(T080) and cannot grant qualification or scientific advancement.

Three focused readiness tests and CLI/projection/packaging/traceability regressions
pass. The new tests use test_discovery_readiness.py because the pre-existing
test_progressive_readiness.py belongs to the legacy RL harness; that original
file and its explicit inventory exclusion are preserved. The full pre-rename
roster otherwise passed 272 cases with three local native-service skips.

## T029–T034 verified structural streams

Added failing-first stream/curriculum tests and implemented a closed typed AST
law with counter-keyed source generation, compositional arithmetic, active
branches and bounded loops. Structural fingerprints normalize all literal
magnitudes; finite 100-term behavior signatures disclose their limitations.
Frozen structure, behavior and finite prefix/shift exclusions remain custodian
inputs; rejected/unsafe/over-byte draws retain their ordinals and costs.

StreamStore reserves durably before work, debits replay work without changing the
ordinal, validates append-only hash chains and publishes immutable bounded shards.
An interrupted shard publication is recovered as the same cut. Shards are capped
at 4096 rows/65536 draws/256MiB and require native verification before ordinary
learner access. The installed prepare-stream role uses actual native/reference
100-term verification, then the existing control-taint/parameter-count and exact
grounding policy with whole-row literal fallback. Role purpose is inherited from
the validated task rather than relabeling acceptance data as research.

The fixed-bin controller reuses the existing Graves EXP3.S kernel with the
20% familiar/10% frontier/70% adaptive mixture, training-only probes and eight
complete updates per feedback block. Snapshots bind layout/probes, probabilities,
bounded reward history, RNG, charged/discarded tokens and stream cuts. Historical
PAC validators are unchanged. Native shard acceptance is mandatory in CI.

Validation: the complete roster initially passed 282 cases with four local native
service skips; additional independent finite-signature, prefix/byte and purpose
checks pass in the focused suite. No actual native/HIP qualification is claimed
on this host. T035/T036 own the subsequent matched-learning acceptance.

## T035–T036 matched support and real CPU learning acceptance

Added the prospective 25M-token/two-seed diversity blueprint with explicitly
unresolved live inputs, paired fresh initialization, shared initial shard/probes,
fixed allocation and separate nominal target/maximum full-update allowance.
PreparedSupport supplies static initial support or preallocated refreshed shards
at declared update boundaries, with a fixed initial familiar pool and training
probes excluded from gradient batches. No retained D warm-start is used.

The bounded DIAGNOSTIC_ONLY acceptance adapter calls the existing bigint worker
SFT step and GeneralizationCheckpointStore. It persists actual model/optimizer,
all RNGs, pending batches, controller, cursor and charged token ledger. Unproven
work after interruption stays charged and is classified as discarded; resuming
repeats the same targets under a fresh debit. Full updates can cross a nominal
target but cannot exceed the binding execution ceiling. Native-unverified rows
are accepted only by an explicitly diagnostic acceptance path, never the normal
load_shard learner gate. Broad research/24h dispatch remains with later task
entitlement/acceptance owners; this does not launch research.

CPU integration passed with actual finite parameter updates, matching initial
models/shards, declared support changes and byte-identical final weights after
interrupted preparation and consumption. Resumed target history/cursors match
uninterrupted runs while spent/discarded tokens remain larger. Budget exhaustion
cannot buy a resumed allowance. Reports preserve costs and baseline retention
without treating training loss as a capability gain. Preparation total-time
receipts and resumed whole-invocation timing are explicitly unavailable locally.
The native-shard-to-real-CPU-update case is mandatory in container CI.

Validation: full progressive/historical roster passed 286 tests with five local
native-service skips before the additional overshoot/ceiling case; that case also
passes. HIP remains in the shared later matrix.

### Efficiency receipt binding correction

The frozen comparison design now records the total-cost metric/scope, while
measured cost receipts are supplied separately after execution and bound into
the decision. This avoids requiring unknowable measured timings in a pre-outcome
policy. Missing measurements yield INCOMPLETE; costs cannot retrofit a capability
comparison into an efficiency claim. The synthetic sensitivity runner likewise
separates its cost-saving assumption from the measured-input slot.

## T037–T039 optimizer/trial contracts and research models

T037 supplies an independent per-program objective oracle, actual canonical full
batch update comparison, uneven-microbatch gold weights and real dropout RNG
restore checks. Two companion specifications require the future T040 accumulated
update to match an independent full-batch gradient, clip/step/LR/allocation once,
and leave parameters/clocks unchanged on interrupted microbatches. Actual spent
work recovery is exercised through T036's canonical CPU learner.

T038 uses actual pinned Optuna 4.5.0 for ordered-history ask/tell replay, failed
and pruned trial retention, and enqueued baseline/nonbaseline controls. Eight
companion specifications cover future T041–T044 durable proposal-before-dispatch,
conflicting tell, confirmation exclusion, grace/comparable-reference pruning,
unpruned controls and wall-limited PARTIAL behavior. These ten future-only tests
are strict expected failures only while their implementation entry points are
absent; they are explicitly separated from mandatory native runtime acceptance.
The independent Torch/gold/Optuna tests remain mandatory in that acceptance.
No accumulated optimizer or trial scheduler implementation is claimed here.

T039 adds a closed progressive-research-model/v1 profile resolved from the retained
recipe, with reference/smaller-width/shallower/larger fresh shapes, valid heads,
positions, codec, strict FP32, seeds and registered effective dropout. Existing
worker dispatch reaches the new constructor. Explicit research-profile adapters
in ChunkHybridEncoder/ChunkEncoder support the new widths while unprofiled and
hardware-only width/default guards retain their previous behavior. Existing
TriStreamEncoder and decoder parameters already support the required dimensions
and dropout; no gratuitous changes were made to those implementations.

Model tests verify actual parameter counts, all four finite CPU forwards, actual
dropout modules, fresh deterministic seeds, a finite dropout-enabled CPU update,
old hardware-profile separation and rejected CPU fallback. A focused comparison
confirms the new reference shape preserves the qualified initialization exactly.
T036 also now exercises a real nine-update training/probe cycle and the declared
eight-update refresh boundary. All model qualification fields remain PENDING.

Final complete progressive/historical roster: **304 passed, five local native
service skips, ten strict expected future-implementation failures**. The reference
initialization case added afterward passed separately. Additional legacy model/
mask/sizing selection: 21 passed; its unmarked native-fitting case could not run
because /opt/oeis-build-info/synthesis-worker is absent on this host. The actual
legacy CPU size/update/parity test passed. Container CI owns native execution;
no CI result or actual-HIP qualification is claimed. New-module lint, affected
legacy undefined-name checks, whitespace and current-disposition binding checks
pass. All 68 historical disposition item states remain unchanged; only current
implementation metadata is refreshed with the original serializer.

T025–T039 are marked complete at their stated software/specification scope.
At that increment, T040 onward, original-producer feature acceptance, actual HIP
and research launches remained separate later tasks. No pre/post implementation hooks are registered.


## T040 shared logical update (first increment)

The canonical `sft_trainer.logical_update` accumulates example-weighted per-program
means, releases each microbatch graph, then clips and steps once. The existing
`supervised_step` interface delegates one complete batch without new scheduler or
allocation behavior. Loss/gradient failures discard partial gradients and leave
clocks unchanged; optimizer failures require restoring the prior complete cut.

Seven independent specifications failed before implementation and now pass. The
optimizer/CI selection passes 18 tests. Token-loss, foundation metrics, bootstrap
learning and progressive stream regressions pass 46 tests with two local native
service skips. Optimizer specifications now join mandatory no-skip container CI;
only T041–T044 trial specifications remain deferred. The research policy adapter
and final T040 completion follow in a separate increment. No HIP claim is made.


## T040 complete: frozen optimizer policy and update-boundary state

`progressive_training.resolve_update_policy` seals the research model identity,
mean-per-program objective, FP32, effective batch 16/32/64, independently chosen
microbatch size, LR multiplier 0.3/1/3, explicit AdamW decay 0/0.01/0.1 and the
model's effective dropout. Unknown or altered policies fail before construction.
`PolicyUpdate` constructs the existing research models and delegates every update
to `sft_trainer.logical_update`; it does not add a second forward/backward engine.

Constant and warmup/cosine schedules use the **reserved cumulative charged-token
end coordinate** of the complete logical update. Default warmup is 5% of the
frozen horizon, rounded up to a token, and final multiplier is 0.1. Separate explicit
warmup and decay horizons support the contract's 1.25M warmup / 400M continuation
schedule without restarting at 25M. A failed attempt keeps its full reservation;
retry needs a new debit and includes spent exposure in its schedule coordinate.
Scheduler and optional fixed-bin allocation clocks step once after finite updated
parameters, never per microbatch. Full-update overshoot is not clipped.

The adapter requires the existing `Ledger`'s matching admitted debit before work.
Its optimizer/scheduler snapshots are available only at complete boundaries;
partial work poisons the adapter and requires a new instance restored from the
prior full checkpoint. It reuses `GeneralizationCheckpointStore` in CPU acceptance
for both an initial cut and a nonempty AdamW cut. Full model/RNG/data and durable
cost-journal ownership remains with the caller and the later T049/T050 installed
recovery implementation; this task grants no long-running research entitlement.

Validation commands (in the existing pinned CPU validation environment):

```sh
PYTHONPATH=src .venv-progressive/bin/python -m pytest -q \
  tests/unit/test_progressive_optimizer.py \
  tests/unit/test_progressive_optimizer_specifications.py tests/unit/test_progressive_ci.py
PYTHONPATH=src .venv-progressive/bin/python -m pytest -q \
  tests/unit/test_token_loss.py tests/unit/test_foundation_config_metrics.py \
  tests/unit/test_bootstrap_workflow.py tests/integration/test_progressive_stream.py
PYTHONPATH=src .venv-progressive/bin/python - <<'CHECK'
import runpy, pytest
from oeis_learn.pipeline_validation.progressive_cli import CPU_TESTS
raise SystemExit(pytest.main(['-q', *CPU_TESTS,
    *runpy.run_path('scripts/qualify_progressive_foundation.py')['HISTORICAL_TESTS']]))
CHECK
```

The first two selections passed **18** and **46** tests respectively (two native
skips in the latter). Final complete progressive/historical roster: **331 passed,
five native-service skips, eight strict expected T041–T044 failures**, in 51.64s.
All 19 policy-adapter tests pass: actual CPU batches 16/32/64 with uneven partitions
and dropout, analytic schedule endpoints, unsupported settings, one allocation
step, retained interrupted charges, blocked partial snapshots, and exact next
update/RNG/weights after disk restoration of a nonempty AdamW checkpoint.
The shared primitive's independent objective/finite-gradient/interruption tests
are no longer expected failures and now join mandatory Docker/Podman acceptance.

New adapter/test lint, affected legacy undefined-name checks and whitespace checks
pass. Installed-wheel/source closure and current implementation binding tests are
included in the full roster. All 68 historical disposition item states are
unchanged; only current implementation traceability metadata is refreshed using
the original serializer. T040 is complete at software/CPU scope. T041 onward,
actual HIP qualification and live research remain pending. No extension hooks are
registered. Coherent increments are pushed without waiting for CI.


## T041–T049: trials, tuning, selection and complete-state cuts

- **T041**: `progressive/trials.py` writes bounded immutable, fsynced event files
  under a process lock, chained to a fixed study/cap binding. Proposal ordinals are
  ordered and immutable. Dispatch, reserve, interrupted invocation, remaining
  cumulative costs and all terminal states are distinct; duplicate conflicting
  tells, confirmation feedback, cost rollback and exhausted restart fail. A
  closed installed trial schema validates every persisted state on write/reopen.
- **T042–T044**: `progressive/tuning.py` reconstructs actual pinned Optuna 4.5.0
  from ordered sealed development history. Study+ordinal seeds and durable proposal
  reuse survive reopen. Two fixed enqueued controls start four startup proposals;
  subsequent proposals explicitly use TPE without adding startup trials after
  failures. Four sizes are enumerated after locked learning selection. The
  installed `propose-trial` CPU operation accepts one bounded closed request;
  no user callback, GPU, external database or sampler pickle is introduced.
  Frozen non-executable policy/blueprint files record the 12+4 roster, distributions,
  controls, seeds, FP32, screening/confirmation horizons, panels and resource caps.
  Pure preflight requires startup, all three evaluations, training and finalization
  to fit together. Screening validates two tasks/group ×16 groups in every stratum.
  Median pruning requires 1M/2M coordinates and five completed comparable development
  references; both controls and all sizing candidates remain unpruned.
- **T045**: `progressive/selection.py` excludes short/partial/failed trials from
  equal-token eligibility, sorts yield/cost/stable ID, and publishes the
  capability/cost frontier and descriptive ranking uncertainty. A feasible baseline
  plus at most three alternatives receive a sealed fresh initialization/optimizer
  confirmation plan, two seeds, distinct unused confirmation panels, 25M/12h caps,
  1.25M warmup and fixed 400M decay horizon. This metadata grants no execution
  entitlement; enclosing campaign freeze/qualification still applies.
- **T046**: real CPU proposal → canonical learner → saved actual checkpoint →
  canonical native generator/scorer integration is in the mandatory CI inventory.
  The tiny acceptance coordinate is explicitly diagnostic, ends PARTIAL and cannot
  win a 4M-token ranking. Separate durable journal tests cover every terminal
  outcome, reopened proposals, grace, controls and selection. Public tuning reports
  retain learning/size/control/grace flags, equal-token ordering, total elapsed
  cost frontier and explicit missing confirmation/HIP evidence.
- **T047–T048**: CPU tests exercise actual AdamW moments/RNG, interrupted charged
  work and exact retry outputs; failed weights/optimizer/scheduler publication,
  fsync, rename and latest-pointer publication preserve the prior complete cut.
  Existing T040 real model/dropout/optimizer disk restore and T036 stream/allocation
  continuation tests remain in the same mandatory roster. Negative action cases
  cover all verdicts, invariant waiver, unfunded advance, recursive diagnostics and
  a fifth pool task. The third-seed executable specification remains a strict
  expected failure until its deliberately later T053 implementation; its original
  verdict and all-seed reporting contract cannot be silently dropped.
- **T049**: `progressive/state.py` and the explicit `ProgressiveCheckpointStore`
  adapter reuse `GeneralizationCheckpointStore` tensor/RNG serialization and
  atomic generations. Discovery records separate learning compatibility from
  campaign/task entitlement, validate closed envelopes before tensor I/O and bind
  full state, update number, stream/allocation/archive/library/cost/selection/
  milestone/consumed cuts. The same cuts are bound inside the tensor generation.
  Weights-only records cannot resume; pending microbatches cannot publish complete
  discovery state. Changed compatibility/task/cap or mismatched dynamic cuts fail.
  T050 still owns installed rollback/cost-journal reconciliation and task transitions;
  no operational recovery entitlement or actual-device qualification is claimed here.

Validation: final full progressive/historical roster **362 passed, six local
native-service skips, one strict expected T053 failure**, in 54.23s. An earlier
run observed two stale source-fingerprint failures while the inventory was being
edited; refreshing the serializer binding resolved both. After the full roster,
explicit TPE startup handling and a full sixteen-trial/sizing enumeration case
were checked with the tuning/current-binding selection: **9 passed**. The new
actual native scorer integration is one of the six local skips and is mandatory
in Docker/Podman CI. Only the separate T053 specification file is deferred from
no-skip runtime qualification; existing negative action tests are mandatory.
New-module lint, affected legacy undefined-name checks and whitespace pass. All
68 historical disposition states remain unchanged. Requirements checklist remains
16/16; no pre/post extension hooks exist. Commits are pushed incrementally without
waiting for CI. No HIP, long-task runtime or research gain is asserted.


## T050: transactional learning recovery

`progressive/recovery.py::Recovery` owns an exclusive stable task lease and an
immutable recovery event chain under the existing campaign budget root. Changing
an invocation output path cannot create a fresh allowance. The last committed
recovery event identifies the complete checkpoint; the tensor store's mutable
latest pointer is never recovery authority. Newly published but uncommitted
generations remain unadopted. Full reference/tensor verification precedes commit.

The transaction is durable cost reservation → pending batch → canonical logical
update → verified full checkpoint → immutable commit. Later journal costs survive
restoration of old weights. Retried work receives a fresh debit; useful and
 discarded exposure are reported separately. Pending retries bind exact entries,
conditioning and target bytes. Draw/rejection/native/attempt costs supplied with
the update use the same irreversible budget vector. Current archive/library and
stream/allocation cuts come only from the committed full checkpoint, never a newer
external pointer. Explicit logical update ordinals replay the original committed
checkpoint and metrics after a lost acknowledgement, without extra optimization
or charge; conflicting duplicate input is rejected.

`progressive_training.RecoverableUpdate` reuses `PolicyUpdate` and the existing
checkpoint serializer, restoring encoder/decoder weights, AdamW/schedule,
Python/NumPy/framework/device RNG and complete caller-owned data/replay/selection
context. Failure poisons the in-memory learner until a new instance restores the
committed cut. The supervisor owns begin/close and termination proofs; it can
reserve the invocation clock before model construction so startup is included.
Finalization headroom is checked before updates. A clean close releases only
proven unused elapsed time; paused calendar downtime is excluded. A reboot with
an unclosed invocation consumes its remaining reserved time and makes the task
terminal PARTIAL. Orphan zero-time update-cost records can be settled after a
proven supervisor close without refunding work. Budget journal publication now
fsyncs its parent directory as well as the event file.

Terminal tasks cannot resume or buy another allowance. A reserved compatible next
task and its dependency/gate remain the explicit T051/T052 dispatch/continuation
boundary; this library adapter neither launches tasks nor treats a new path as
such a reservation. Existing campaign/runtime qualification and enclosing process
supervision remain required. CPU acceptance does not qualify actual HIP or a
sustained long-running task.

Validation: full progressive/historical roster **372 passed, six local native
service skips, one expected later T053 specification failure** (53.81s). Final
startup-accounting and recovery/current-binding selection: **12 passed** (11.15s).
Real dropout-enabled CPU model tests interrupt after debit, during accumulation,
after optimization, after tensor publication, and after commit/lost acknowledgement.
Recovered model parameters and RNG exactly match uninterrupted constant-schedule
updates, while discarded charges remain spent. Additional cases cover exclusive
leases, protected finalization, clean pauses across reboot, unknown termination,
startup failure, conflicting replays and attempted output-path budget reset.
These new unit and integration cases are in mandatory CPU/native CI. New-source
lint and whitespace checks pass; current disposition metadata uses the existing
implementation serializer and preserves all 68 historical item states. Checklist
remains 16/16; no implementation extension hooks are registered. Changes are
committed/pushed incrementally without waiting for CI.


## Historical T051–T059 increment: lifecycle gaps later closed below

At this earlier increment, **T056–T059 software** was complete and T051–T054
helpers were partial. **The following gap inventory is historical and superseded by the T051–T055 delivery below.** In particular,
`train` is still absent from the installed implemented-operation set. The finite
controller, host command wiring, milestone/contingency calculations and Progress
sampler do not establish the requested installed two-task continuation. T055's
actual CPU installed-role continuation/interrupt/export acceptance is still missing.
No completed lifecycle or long-task readiness is claimed from these helper tests.

Completed admission/preparation/archive work:

- T056 counterexamples reject nontraining role, short horizon, imported metadata,
  native/reference disagreement and prefix-only continuation mismatch. Interrupted
  admission retains reserved attempt/native-work charges; interrupted archive
  publication leaves the previous immutable cut readable and sampleable.
- T057 adds closed admission/archive schemas and a separate OEIS-derived archive.
  It retains at most 4,096 total targets, at most two distinct structures per task,
  shortest-first deterministic ties, stable task-ID eviction and task-first then
  alternative sampling. Source/structure/token bindings are checked before archive
  insertion. Duplicate targets are idempotent, and only explicit committed cuts
  are sampled. Samples retain the admission identity, visible conditioning and
  charged target-token count. Historical mechanical replay validators are unchanged.
- T058 consumes the existing `fresh.bulk.import_snapshot` producer's exact
  `indexed-source/v1` manifest, checking its original nested terms/metadata hashes
  and reusing `fresh.bulk.indexed_terms`. Signed authoritative offsets, finite/table
  exclusions and complete 100-term support are preserved. Exact selected membership
  is bounded at 512, outcome blind, with deficits before search. Existing overlap
  grouping and exposure source/behavior/window relations exclude detected exposure.
  Only first-20 terms enter generator envelopes; separate private admission envelopes
  carry the training-only full verification horizon. Names/formulas/raw source
  metadata are excluded from both target and generator channels.
- T059 adds final-source 100-index native plus independent-reference admission,
  sealed member/snapshot/source/limits/implementation binding and finite validity.
  The established parameter taint/grounding/codec checks produce paired targets
  where valid; shifted-index members retain literal targets. The installed
  `admit-training` operation constructs the actual SynthesisClient and checks
  candidate/member identity through its typed channels. No fixture/native mock
  is used for its positive native integration test.

Partial lifecycle/scaling helpers are intentionally recorded as incomplete:
RunStore rejects unallocated predecessor slots and retains invocation-specific
pause requests; host handlers require fresh original qualification and lifecycle
feature capability before dispatch. Scientific edges are currently withheld rather
than advanced from process completion. Milestone helpers preserve the 400M horizon;
third-seed diagnostic calculations retain the original failed verdict and all seeds.
The previous third-seed test now passes and joins mandatory CI. Progress exposes
separate allocator counters and missing generator timing reasons, with storage/
inode/finalization preflight; it still needs continuous installed learner wiring.

Validation: full progressive/historical roster **386 passed, seven local native
service skips, no expected failures** (54.97s). Additional publication/interruption
and admission/lifecycle/preparation checks pass **12 tests with one native skip**.
The seven full-roster skips include the new real native admission case, which is
mandatory under Docker/Podman CI. Existing installed-wheel/source closure, portable
host imports and implementation binding checks pass. New-source lint and whitespace
checks pass. All 68 historical disposition item states are preserved. Checklist
remains 16/16 and no extension hooks are registered. Changes are pushed incrementally;
CI is not awaited. No actual HIP, scientific gain or completed T051–T055 claim.


## T051–T055 delivery: installed learning lifecycle

The installed `train` operation now consumes a sealed, bounded training package,
constructs the real research-profile encoder/decoder and uses the existing
PolicyUpdate/logical-update path. Acceptance data is explicitly DIAGNOSTIC_ONLY;
research requires native-verified prepared support and actual qualified HIP.
The portable host remains stdlib-only. Original runtime producers and complete
campaign-bound feature acceptance remain prerequisites for a research dispatch;
this shortened CPU receipt cannot replace the later T090/T091 device/full-feature
acceptance. Report/bundle verification beyond this bounded lifecycle export remains
owned by T079/T080.

- **T051:** run/pause/resume/status use one canonical campaign control/accounting
  chain and a nonblocking campaign lease. Each resume owns a fresh output and pause
  namespace; no path resets a task cap. Only the installed owned worker consumes
  a pause request, at a complete checkpoint boundary. Until acknowledged, pause is
  pending/external-deadline. Missing reserved tasks/edges produce scientific
  INCOMPLETE. Failed producers retain their first error and cannot authorize
  advancement. Typed decision bundles are recomputed against the original frozen
  policy, roster/seeds/design and actual remaining reservations before certifying
  a DAG edge. Process completion alone cannot certify scientific gain.
- **T052:** `parent_slot` resolves one paired predecessor in the frozen DAG;
  `parent_task` supports an explicit earlier compatible continuation within an
  already admitted finite slot. Only committed terminal COMPLETE/PARTIAL state can
  initialize a successor. Model/optimizer/scheduler/RNG, stream/allocation state,
  token/update clocks and ancestor costs survive. Tasks retain separate remaining
  allowances. Before the next atomic write, the shared checkpoint retention owner
  retires only proven owned tensor generations, preserving the current full cut
  and frozen first-crossing milestone pins. Immutable update/event receipts survive
  retirement; committed-update replays read their metrics from the latest full cut.
  Uncommitted tensor tips are discarded without becoming recovery authority.
  Pin capacity is checked against planned milestone exposure before execution.
  The cumulative 25M/100M/400M milestones record full-update overshoot;
  research scaling fixes the 400M horizon and 1.25M warmup. Marginal gain per added
  active hour includes visible ancestor task costs. Weights-only initialization
  starts fresh optimizer/clocks and keeps external ancestor cost unavailable,
  including through later full-state continuations. Supervisor-final accounting
  includes startup, all invocations and proven cleanup; overlapping phase times
  are never added as extra wall time. Unattainable targets stop PARTIAL.
- **T053:** diagnostic policies name concrete design/panel/task slots/reservation
  and terminal rules. Frozen design/panel bytes are required; assessment/training
  truth, reused panels, training-loss-only triggers, recursive episodes and
  exhausted reserves are rejected. The extra-seed route independently recomputes
  all paired rows and conservative adjusted bounds. All three fixed seeds count;
  mean >=2 pp, at least two positive, no seed below -2 pp, per-arm/seed/stratum
  completion >=95%, retention <=2 pp and positive adjusted lower bound remain
  binding. Both original and third-seed looks use the same prospectively frozen
  comparison count including the extra look. Original two-seed rows, verdict and
  decision certificate remain unchanged. Scalar helper outputs grant no authority.
- **T054:** a background <=60s heartbeat continues during blocked updates/checkpoint
  work, reporting committed/charged/discarded exposure, actual active-clock
  remainder and explicit waiting/unavailable reasons. Final interruption samples
  reconcile discarded charges. CPU/resource, mapped sensors, free disk/inodes and
  overlapping allocator/shared-memory counters are reported separately. Generator
  queue/fit/tail clocks are collected from original installed generation receipts,
  including tails, partial clocks and explicit omissions; they remain unavailable
  when no producer exists in this training scope. Preflight protects peak-live checkpoint storage, inodes and
  finalization; research requires at least 30 minutes. The enclosing existing
  process supervisor bounds blocked workers and cleanup.
- **T055:** real subprocesses import a built installed wheel and execute the same
  train/decision dispatcher. Tests cover interrupted debit, resume, two-task
  continuation, weights-only initialization, partial allowance, original producer
  failure and an eligible reserved diagnostic followed by actual CPU learning.
  Paired diagnostic outcomes are explicit mechanical fixtures, not checkpoint
  evaluation or research evidence. Bounded public metadata ZIPs preserve original
  result/error references and exclude tensors, training inputs, private truth and
  raw tracebacks. The acceptance receipt records actual short monotonic intervals,
  keeps sustained/HIP evidence PENDING and records the later sustained protocol.
  The packaged shared `progressive-acceptance.json` inventory queues the actual-HIP
  logical-update/recovery and two-task/pause cases for T090/T091.

Validation: final full progressive/historical roster **408 passed, seven local native
service skips** (129.95s), including installed-wheel learning and bounded tensor retention. The mandatory Docker/Podman CI roster includes every new
CPU case; native skips remain failures in mandatory native-image mode. Final
multiplicity, ancestry, host-failure and retention changes pass the complete roster
before completion. Installed-wheel learning, original source closure and unchanged
historical disposition items are checked. No HIP, sustained-duration or scientific
advancement claim is made. Requirements checklist remains 16/16; no implementation
extension hooks are registered. Commits are published incrementally without waiting
for CI.

### Critical convergence repairs T112–T116

- **T112:** freeze validates every selected cohort against original exposure cuts,
  historical exposure and detectable cross-panel relations, including value-window
  overlap concealed by renamed declared fingerprints.
- **T113:** ordinary gate authorization binds original frozen panels, paired
  arms/seeds, checkpoint identities, candidate files and scorer files. Recomputed
  producer outcomes retain the original roster and missing-result denominator.
- **T114:** strict imported initializers require verified ancestry ending at random
  mechanical initialization, complete original state and prior exposure. Unknown
  and assisted origins are rejected at freeze and in the installed learner;
  weights-only imports still start a new optimizer and new clocks.
- **T115:** generation, stream preparation, scoring and admission reserve their
  maximum non-time work before dispatch. Installed roles verify that reservation
  and debit attempts, rejected/retried work and native requests before execution.
  Late results and interruptions cannot restore spent work.
- **T116:** freeze measures the checkpoint manifest's tensor closure and optimizer
  growth allowance. Research preparation supplies the original `checkpoint_root`;
  a weights-only blob or incomplete manifest cannot establish that measurement.
  The frozen storage graph includes disjoint task-store reservations for all live
  predecessors/pins, nested projections, journals, shards and package reserve.
  Initial and subsequent checkpoint writes, copied inputs, attempt logs and owned
  evidence enforce bytes/inodes before publication. Prior invocation outputs stay
  charged; replacing archive cuts cannot evade a fixed history ceiling or delete
  entries referenced by older cuts. Growth failure preserves the prior full cut.

Both Docker and Podman CI failures used the same obsolete `correct` lookup in the
native tuning test. It now reads the scorer's published `equal_budget` outcome and
checks its original denominator. Native execution remains mandatory in CI.

Validation for these repairs: the complete registered progressive/historical run
returned **444 passed, seven local native-service skips**, plus two stale historical
implementation-hash failures. The current disposition hash was regenerated using
`experiment_runner.implementation_identity()`; both failed checks then passed in a
**20-test** archive/budget/identity rerun. An additional **52 historical training and
evidence tests** passed. Installed lifecycle/recovery, strict-origin, producer-binding,
resource debit and storage regressions passed, including actual optimizer growth
and repeated archive replacement. Syntax/undefined-name checks and `git diff --check`
passed. No native-container, HIP, sustained-duration or research acceptance is inferred
from the local CPU results. T117–T125 remain separate convergence work.

### OEIS cycle and component foundations T060–T066

- **T060:** the bounded yield audit verifies original selected candidates and
  admission provenance, counts distinct groups/structures, and publishes missing
  admissions, rejection reasons, deficits and active search/admission costs.
  The production policy retains the 512-task, eight-attempt, 60-second portfolio,
  12-hour audit and 32-group/16-structure/rate gates. Lower diagnostic thresholds
  cannot establish autonomous discovery or research advancement.
- **T061:** replay binds an immutable archive cut and uses charged target tokens
  for deterministic 25% OEIS/75% synthetic mixing with whole-example overshoot.
  Original membership is checked before consumption. Complete checkpoints and
  the irreversible journal reconcile mixture charges after failed updates;
  continued-synthetic controls receive no discovery archive.
- **T062:** the fixed host coordinator runs one search/admit/update/evaluate
  cycle with four frozen paired-seed/arm tasks, matching initializers, token
  milestones and inference budgets. It separates OEIS, transfer and retention
  results, reports discovery cost, defers on insufficient yield and requires a new
  reserved study for another cycle. It reuses installed generation, learning,
  scoring, recovery and native admission. CPU acceptance performs actual updates
  and restore; forced candidates remain diagnostic fixtures. Native positive and
  low-yield end-to-end cases are registered for mandatory container acceptance.
- **T063/T065:** typed pure-expression components preserve operand order and
  alpha-normalize holes without capture. Closed library validation rejects
  recursion, dependency cycles, nontraining origins and insufficient support;
  runtime and expanded-program node/byte/work limits include emitted wrappers.
  Native/research provenance and finite verification scope remain explicit.
- **T064:** seven independent prospective tests cover source-bound polynomial
  identities, wrong coefficients, unsupported loops, finite machine limitations,
  spoofed review/novelty and transitive refutation. They conditionally expect
  missing modules while T069–T071 are unimplemented, and are excluded from the
  mandatory passing acceptance roster. The packaged inventory marks this owner
  pending. This task authors contracts; it does not implement or certify proofs.
- **T066:** deterministic connected-subtree mining requires three independently
  verified training structures across two groups and preserves expression order.
  Original admissions or verified training rows supply the only source closure;
  research mechanical rows must reproduce their bound frozen generator law and
  avoid forbidden structures, behaviors and visible prefixes. Durable work debits
  precede mining and native requests. An owned reserve/verify/commit store binds
  implementation/policy, exact source references, full expansions and independent
  parser/reference/native checks. Interrupted work stays charged without exposing
  an uncommitted library. Previous cuts remain readable, and repeated verification
  reuses a compatible committed cut. Installed extraction/verification roles use
  the same bounded dispatcher and owned persistent state. Projection tests cover
  shared original entry closures across multiple archive cuts.

Validation: the complete registered progressive/historical roster returned
**488 passed, 11 local native-service skips** (199.54s), including installed-wheel
learning and real CPU replay/control updates. After final source/projection
provenance tightening, **50 focused tests passed, two native-service cases skipped**;
both historical implementation-binding checks passed against the refreshed
`experiment_runner.implementation_identity()` hash. The seven prospective T064
tests remain expected failures until their later production owners exist.
Blocking syntax/undefined-name checks and `git diff --check` passed. Native skips
are failures in mandatory native-image mode. No actual native-container, HIP,
sustained-duration, mathematical-proof or scientific-gain claim follows from these
CPU results. Requirements checklist remains 16/16; no implementation extension
hooks are registered. Changes are published incrementally without waiting for CI.

### Bounded component-guided proposals T067

`component_search.py::Proposals` composes mined pure expressions with deterministic
ordinary typed operands/wrappers in operand order. Each of slots 4–7 considers at
most two expansions and ranks their existing-codec token sequences using the
unchanged decoder's teacher-forced log probability. The canonical portfolio driver
continues to own all eight attempts, the shared 60-second deadline, final-source
fitting, independent admission and selection. Slots 0–3 remain ordinary neural
draws; the no-library arm retains all eight ordinary draws. Empty, invalid or late
proposals cannot add attempts or bypass fitting. No library tokens or embeddings
are introduced.

Installed extraction exports an immutable search bundle binding the committed cut,
original expansion proofs and irreversible discovery work. The generator projects
that exact artifact under frozen model authority, checks model compatibility and
requires native research provenance for research. It never mounts source solutions
through the library channel. Expansion and scoring reserve conservative work/token
maxima before execution; rejected expansions and interrupted inference remain
charged. Reports retain expanded nodes/tokens, scoring tokens, expansion reservation,
original discovery work and native search/execution costs rather than treating a
short component name as free work. Package identity and nested projection storage
include the actual input bytes.

Tests cover exact codec/source identity, typed expansion, four/eight attempt counts,
expired and repeated slots, real CPU decoder scoring, committed proof/cost tampering,
original payload projection and generator cost reporting. The native acceptance
case mines a three-structure library and sends a component-generated `2n + 1` source
through the original native fitter. Its ordinary draws are deliberately incomplete
mechanical fixtures, so this is a routing/semantics check, not an improvement claim.
T068 still owns distillation, atomic model/library cuts and matched usefulness gates.

Validation: **357 registered tests passed, 12 native-service skips** (138.05s);
**66 historical diagnostics/implementation-binding tests passed, one native skip**
(56.12s). Focused generator/projection/search checks returned **21 passed, three
native skips**, including the new real CPU scoring integration. The latter check
was added after the registered run began and passed separately. Blocking
syntax/undefined-name checks and `git diff --check` passed. No local native/HIP or
research superiority is claimed. The mandatory native-image mode fails unavailable
native cases rather than silently accepting skips. No implementation hooks exist.

### Component distillation and full cuts T068

Component proposals enter a separate training-only admission channel. The installed
admission role expands the original typed expression, preserves codec/source identity,
and checks all 100 mechanical training terms with the independent reference and
native service. It rejects confirmation/assessment members, altered proposals and
incorrect grounding. Reference-only fixtures remain acceptance-only. Successful
admission emits an immutable expanded target, not a new decoder vocabulary.

The learner's component adapter reuses charged-token, whole-example replay and its
irreversible journal reconciliation. The no-library arm receives neither library
nor targets. Compatibility binds the update rule while complete checkpoints bind
the actual target/library cut and embedded definitions/proofs. Mid-task cut changes
are rejected; an explicitly declared, unspent successor can atomically adopt a new
cut while retaining optimizer, RNG and ancestor charges. Failed publication preserves
the preceding model/library generation. `components.json` declares the paired two-seed
library/no-library blueprint, matched synthetic/token/inference allowances, fixed
harder held-out panel and separate discovery/validation/downstream costs.

Acceptance mines a real three-structure component, produces a new expanded candidate,
checks its terms, performs actual CPU model updates and restores the complete cut.
The installed subprocess rehearsal covers both arms and two seeds, interruption with
retained discarded charges, and declared-successor adoption of a new library version.
A negative harder-panel/cost example retains the baseline; syntax compression alone
never establishes utility. The native distillation case is mandatory in native CI
and skips locally without the service. No HIP or scientific improvement is claimed.

T068 focused validation: **52 passed, one local native-service skip** (45.79s),
including both historical implementation-binding checks and checkpoint/recovery,
role, budget and installed-asset regressions. Final native-producer/type and
actual-component-call tightening passes the targeted admission regression.

### Scoped discovery dossiers T069

The new closed `progressive-dossier.schema.json` and `claims.py` bind the exact
original generator selection, source, source snapshot/offset, library and generation
budget before novelty analysis. Independent finite execution is stored as an original
immutable receipt with observed/expected values, native/reference producer identity,
all detected counterexamples, explicit assumptions, original baseline comparisons
and untested range. The 100-term scope and lack of unbounded machine-safety proof are
explicit. Research requires actual native execution; reference-only CPU fixtures
remain diagnostic. Conflicting references, altered seals/source/offset/range and
missing untested scope fail validation.

Claim category and novelty are independent. An authorized named reviewer must have
an original frozen registry entry and exact candidate-bound scoped prior-work evidence
for `independently_confirmed`; a model name or a digest cannot register that authority.
Finite counterexamples prevent finite/bounded agreement promotion. Proof promotion
replays only the registered source-bound polynomial checker; until T070 provides it,
proof labels (including legacy certificates) are rejected. Six prospective proof and
quarantine cases remain owned by T070–T071. T064's novelty-axis case now runs in the
passing dossier suite, without admitting expected failures into mandatory CI.

The installed `review-claim` operation verifies a read-only projection containing the
complete original candidate/execution/review closure, repeats actual native execution
and publishes a new dossier with its parent identity. It preserves original bytes and
never emits a training-eligible target. Projection storage and pre-dispatch work
reservations include the exact nested inputs. The full public CLI roster and local
Podman support are unchanged; duplicate Podman CI jobs remain disabled as requested.

Final recovery review also closes a T068 edge: a successor's first component cut is
bound to that task in the complete checkpoint. Even a pause before its first update
cannot reopen the adoption boundary or replace the committed library. The installed
rehearsal pauses there, rejects a second adoption without changing the checkpoint,
and resumes actual learning with the selected new library.

Validation: complete registered CPU/historical roster **515 passed, 15 local native
service skips** (179.32s). Final focused admission/dossier/packaging/identity/recovery
checks cover later provenance and one-time boundary tightening. Native execution,
HIP, mathematical proof and research utility are not inferred from CPU results.
Requirements checklist remains 16/16; no implementation extension hooks exist.
Final focused tightening: **19 passed, two local native skips** (19.04s), plus
**seven component learning/recovery tests passed, one native skip** (38.65s).
Both historical implementation-binding checks pass against the final source hash.
Blocking syntax/undefined-name and whitespace checks pass. Changes are committed
and pushed incrementally without waiting for CI.

### OEIS first-task preparation accounting CI repair

Native CI on `f9bba36` returned 530 passing cases and one OEIS-cycle failure. The
first OEIS learner task also owns bounded discovery/admission, so its non-token
allowances are already charged before initialization. The prior pristine-resource
assertion incorrectly rejected that required path. Recovery now permits only
completed zero-token `oeis-host`/`generate`/`admit-training` reservations in the
OEIS-derived stage under the current original supervisor training reservation.
All charges remain spent. Any token debit, unfinished preparation, previous
training/update attempt, foreign stage/track or existing checkpoint still blocks
fresh initialization. Baseline evaluation continues to use the original initializer
after the learner's initial update, without changing task budget attribution.

Regression validation: **28 passed, two local native-only skips** (26.15s), including
actual installed replay/control updates and full restore with precharged discovery
work, seven preparation/learning-debit boundary cases, and optimizer/crash recovery.
Both historical implementation-binding checks passed after refreshing the source
hash. Blocking syntax/undefined-name and whitespace checks passed. Native end-to-end
acceptance remains mandatory in Docker CI; local CPU evidence does not replace it.

### OEIS starting versus trained baseline assertion repair

CI run `42598584420` on `7d5e800` passed 538 of 539 registered cases. The remaining
failure was the native OEIS-cycle test observer: it treated every `baseline` label
as the frozen initializer. FR-030 requires two comparisons, and the continued
synthetic arm correctly evaluates its trained checkpoint under that same metric
label. Production cycle routing and the metric labels remain unchanged.

The observer now distinguishes the starting comparison from both trained arms,
checks each trained checkpoint against that task's committed recovery weights and
completed optimizer update, and requires all six evaluations across both seeds.
The low-yield case still requires no training or evaluation. The native fixture
honors the existing `OEIS_SYNTHESIS_WORKER` override used by the production client;
the default CI path and mandatory missing-worker failure remain unchanged.

The exact prior assertion failure was reproduced locally using the actual worker
and WASM guest from this CI run's `phase7-native-qualification` artifact, with
LLVM 19 runtime libraries extracted into scratch. After the repair, all 12 OEIS
tests passed, including both native end-to-end cases, with no skips (69.38s).
This local native execution is not a Docker isolation, HIP or research-gain claim.

The complete registered CPU/native/historical roster then passed: **539 passed,
zero failures, zero errors, zero skips** (277.84s), with both mandatory-native
environment flags enabled. The qualification script reconciled the complete JUnit
inventory, actual native witness and installed-wheel coverage successfully. The
local launch supplied the runtime scratch directory normally created by the
container launcher. Blocking syntax/undefined-name lint and whitespace checks
passed. No runtime package code or historical implementation hash changed. T018
still awaits a successful container CI receipt; CI is not awaited after pushing.

### T070 — bounded polynomial proof core

The independent reference parser supplies syntax/types to a new exact coefficient
normalizer; no production lowering, arithmetic evaluator, fitter, sampled values
or legacy proof label supplies proof authority. Supported expressions are one pure
return over index/integer/add/sub/mul. Limits cap source, nodes, nesting, degree,
coefficient bits, symbolic work and deadline; callers may only tighten them.
Optional inclusive finite-domain interval arithmetic checks every original
intermediate, including expressions that algebraically cancel, against the fixed
bigint numeric profile. It does not establish fuel/memory/deadline sufficiency.

Certificates bind exact source bytes. Dossier promotion publishes a new immutable
record with a replayed checker/certificate identity and the explicit theorem scope
`program_equals_polynomial_only`. The sequence law, novelty and unbounded machine
safety remain unproved/unreviewed. Refuted finite expectations and unsupported
assumptions cannot acquire a proved label. Review reserves up to four bounded
symbolic replay attempts before dispatch and retains failed-attempt charges.

First increment: 78 focused polynomial/dossier/schema cases passed; two existing
native-only cases skipped in that CPU invocation. The five original T064 proof
cases now pass; only the T071 quarantine specification remains expected-failing.
Installed native review and full-roster validation follow before T070 completion.


### T070 — installed proof review and completion

The installed-wheel integration now exercises both finite-to-proved promotion and
re-review of an already proved dossier through the actual native worker. Both
preserve original dossier bytes, retain unreviewed novelty and training
ineligibility, and replay the source-bound certificate from the projected closure.
Independent expected work counts are three symbolic/review attempts for first
promotion and five for re-review, with one native dispatch. The host conservatively
pre-reserves five attempts/rejections; invalid coefficients retain all reserved
charges without publishing a dossier. Underfunded attempts, rejections or native
work are rejected before dispatch. The acceptance inventory and quickstart now
record the implemented proof scope separately from pending T071 quarantine.

Validation: **91 focused tests passed with no skips**, including installed-wheel
native review, invalid-proof publication refusal and budget accounting. The full
registered CPU/native software acceptance runner then passed **608 tests, zero
failures/errors/skips**, in 277.71 seconds, including the historical authority and
package-identity checks. Mandatory native mode used the previously recovered CI
worker/driver/bigint/solver binaries with their LLVM 19 runtime libraries; no native
sources changed. The local launch supplied the runtime scratch directory normally
created by the container launcher. This is local software evidence, not a Docker
or HIP qualification receipt, sustained-duration evidence or a research result.
Blocking E9/F63/F7/F82 lint and `git diff --check` passed. T070 is complete; T071 and
later tasks remain pending. No post-implementation extension hooks are registered.


### T071 — append-only quarantine

Closed typed dependency DAGs reject missing parents, cycles, duplicate identities
and changed ancestry under node/edge/journal bounds. Immutable chained refutation
overlays enumerate every descendant; late consumers inherit earlier refutations.
The campaign accounting root owns the quarantine journal. Real checkpoint save
records archive/library/consumed-update/previous-checkpoint dependencies; restore
and component adoption refuse affected lineage. Training binds admitted targets,
archive cuts, library origins and expanded targets before work. Explicit dossier
registration links review records without rewriting original claims. Recovery
selection retains supplied spent costs and labels absence of a clean checkpoint
CONTAMINATED; actual retraining uses the existing separately charged successor path.

Validation: 30 quarantine/claim/continuation/recovery tests passed; the installed
lifecycle/component regression selection passed 44 cases (one native-only skip in
that non-native invocation). A real checkpoint test verifies restore rejection and
unchanged spent-token journal. Dossier-overlay and invalid-graph tests preserve
original input bytes. Broader native acceptance will run after the source additions.


### T072–T074 — finite external-source contracts and bounded permutation preparation

Source records bind revision/hash, access/use conditions, indexing, object and
relation semantics, exact transformation, verification and external-assisted track.
Unknown/approximate/nonintegral values, unresolved terms, missing offsets, arbitrary
object order, finite-polynomial zero padding and downloaded code are rejected.
Outcome-blind relation closure groups related statistics/maps and exact duplicate
finite series across sources. The pilot-support reducer rejects role overlap and
reports missing independent groups explicitly.

The local adapter permits eight reviewed permutation statistics (within the cap of
32), moment orders one/two and complete n=0..8 collections. It reserves 46,234
objects and conservative work before enumeration, checks factorial cardinalities,
empty-collection conventions, alternate insertion enumeration through n=5, and
independent inversion/fixed-point total formulas. Preparation publishes immutable
custodian output and never imports source-provided code. Short-profile contract
tests enforce nine authoritative terms, four visible positions, explicit masks
and unchanged historical non-20-term rejection. 22 source/short/enumeration tests
passed, including each budget and adversarial-source branch. Neural integration
is owned by T075–T078 and is not asserted by this increment.
