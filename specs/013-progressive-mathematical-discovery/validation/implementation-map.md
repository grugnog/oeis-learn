# Spec 013 implementation map

Scope: T001–T024 software, with T018 acceptance pending. Read spec.md, plan.md, tasks.md, research.md, data-model.md,
quickstart.md, hardware-validation.md, planning-review.md, all five contracts,
the requirements checklist and adopted constitution in full before implementation.
Requirements checklist: 16 checked, zero unchecked. No extension hooks registered.

## Reused owners and preserved guards

| Boundary | Existing owner | 013 adapter |
| --- | --- | --- |
| Portable host | scripts/foundation_run.py → foundation_local.py::parser/main | progressive_cli.py → progressive_commands.py; stdlib host |
| Installed role | generalization_roles.py::invoke / project_and_invoke → generalization_role.py | progressive_roles.py; reuse process supervision and engine isolation |
| Real learning | generalization_training.py::run_torch → torch_update → rl/sft_trainer.py::supervised_step | later progressive_training.py; no second learner in this increment |
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

Source/test implementation is complete for T001–T017 and T019–T024. T018 now has mandatory Docker/Podman CI acceptance and remains unchecked pending its first successful receipt. The initial local validation environment has no Docker or Podman binary
and no packaged native worker or HIP device. Local CPU/stdlib checks cannot
establish container/native/HIP qualification. No training/research is launched.


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
