# Validation and workstation handoff

**Workstation handoff implemented; physical validation pending.** T126–T143 software fixes are complete. Start with the [consolidated workstation rehearsal](workstation-handoff.md), which builds the pinned images on your machine, runs tiny real-HIP qualification/acceptance, and retains a failure-safe support ZIP. Original hardware, sustained and scientific evidence is still required; CPU checks do not supply it.

Implementation order and optional branches are in [tasks.md](tasks.md). Follow [hardware-validation.md](hardware-validation.md) to batch actual-HIP validation after CPU/native checks: one consolidated qualification matrix, then reuse exact compatible producers and execute only missing post-freeze device cases. This reduces repeated qualification, not scientific training seeds. Changed source closure or an uncovered execution scope still requires matching validation.

## Prerequisites

- Spec 013 implementation installed from the same checkout/source closure as its images; existing functional Docker/Podman and actual HIP workstation access.
- Retained original campaign, checkpoint/recipe/input producers and prior exposure manifests, including fresh-002. Start with the existing original campaign root; `init` inventories dependencies and reports all missing references together.
- Measured free disk/inodes and shared-memory capacity; a single GPU owner; exact selected settings and finite stage/campaign budgets.
- [CLI contract](contracts/cli.md), [experiment contract](contracts/experiment.md), [execution contract](contracts/execution.md), [learning/discovery contract](contracts/learning-discovery.md) and [target/contingency contract](contracts/contingencies.md).

## Currently runnable design checks

From repository root:

```bash
.specify/scripts/bash/check-prerequisites.sh --json --require-spec
git diff --check
```

These check document prerequisites/whitespace, not scientific correctness or runtime readiness.

## Implemented acceptance to require before research

The named progressive tests are registered in the mandatory foundation inventory:

```bash
uv run pytest tests/unit/test_progressive_*.py tests/contract/test_progressive_*.py
uv run pytest tests/integration/test_progressive_*.py -m 'not gpu and not docker'
```

Include meaningful cases for every row in plan.md's traceability table. Required real tests: logical-batch accumulation equality, immutable draw/reject replay, complete state after interrupted update, trial idempotency, cross-role truth rejection, prefix-overfit admission failure, component expansion/refutation and independently recomputed gates. Container/native and actual-HIP coverage comes through `discovery qualify` and `acceptance`; local CPU tests do not substitute for them.

## Consolidated workstation sequence (execution remains gated)

The examples use workspace-local original paths from the prior workflow. The operator's actual producer refs must be verified by `init`; no copied sample hashes or unrelated qualification receipts are accepted. Select a fresh output suffix for each new campaign.

```bash
python3 -S scripts/foundation_run.py discovery init \
  --reference-campaign run-state/generalization/pac-012-bcd-r1-002 \
  --output run-state/discovery/013-001/inputs
```

Expected: blueprint, input/exposure inventory and explicitly named deficits. Resolve source/retained-artifact references locally. Inspect the proposed stage subset, target classes, claim and diagnostic routes, trial spaces, task roster, all caps and estimated storage. The default subset is capability + diversity; select the full dependency graph only when its large finite reservation is intended. Freezing a blueprint is required before any research run, but defining it can use the already authorized study scope without asking permission at every internal gate.

```bash
python3 -S scripts/foundation_run.py --engine podman discovery qualify \
  --blueprint run-state/discovery/013-001/inputs/blueprint.json \
  --output run-state/discovery/013-001/qualification
python3 -S scripts/foundation_run.py --engine podman discovery prepare \
  --blueprint run-state/discovery/013-001/inputs/blueprint.json \
  --qualification run-state/discovery/013-001/qualification \
  --output run-state/discovery/013-001/prepared
python3 -S scripts/foundation_run.py discovery freeze \
  --blueprint run-state/discovery/013-001/inputs/blueprint.json \
  --prepared run-state/discovery/013-001/prepared \
  --qualification run-state/discovery/013-001/qualification \
  --output run-state/discovery/013-001/frozen
python3 -S scripts/foundation_run.py --engine podman discovery acceptance \
  --campaign run-state/discovery/013-001/frozen/campaign.json \
  --output run-state/discovery/013-001/acceptance
python3 -S scripts/foundation_run.py --engine podman discovery run \
  --campaign run-state/discovery/013-001/frozen/campaign.json \
  --acceptance run-state/discovery/013-001/acceptance \
  --output run-state/discovery/013-001/execution
```

Expected: actual CPU/native/HIP evidence, outcome-blind support census, immutable campaign, shortened all-stage acceptance labeled diagnostic, then only selected research stages. A sparse cohort, model infeasibility or incompatible receipt stops before dependent research. A negative research gate completes its experiment with its evidence verdict preserved. A separately frozen baseline/diagnostic action may proceed within its own allowance; otherwise the affected branch stops. Record `mechanical_status`, `scientific_status` and `next_action` independently.

The reference recipe remains fixed for the first static/refreshed comparison. If qualification or a language audit proposes a new primitive/model family, its receipt and compatible child freeze must exist before that branch runs. The user can review one diagnostic package instead of uploading individual failure files.

## Recovery and independent review

```bash
python3 -S scripts/foundation_run.py discovery status \
  --run run-state/discovery/013-001/execution
python3 -S scripts/foundation_run.py --engine podman discovery pause \
  --run run-state/discovery/013-001/execution
python3 -S scripts/foundation_run.py --engine podman discovery resume \
  --run run-state/discovery/013-001/execution \
  --output run-state/discovery/013-001/resume-001
python3 -S scripts/foundation_run.py discovery report \
  --run run-state/discovery/013-001/execution \
  --output run-state/discovery/013-001/diagnostics.zip
python3 -S scripts/foundation_run.py discovery verify-bundle \
  --archive run-state/discovery/013-001/diagnostics.zip \
  --output run-state/discovery/013-001/verified
```

Recovery keeps the same task/campaign accounting. A spent 24-hour task can only advance to an already reserved compatible continuation; this is not a fresh 24-hour entitlement for that task. Status/report follow the append-only campaign invocation chain, including later resume outputs. A paused or failed run must still produce original failure and accounting evidence. Share the one bounded ZIP; publish concise findings in the repository.

Final assessment is a distinct `discovery assess` invocation with the frozen selected checkpoint record and the same campaign/acceptance scope. It is not automatically triggered by a trial, milestone or component update. Subsequent use of those scores is labeled exposed.

## Acceptance evidence matrix

| Scenario | Required observable outcome |
| --- | --- |
| Unsupported language witness / capped search | Distinct coverage and runtime outcomes; no inexpressibility or UNSAT invention |
| Static/refreshed learning | Matched model/policy/exposure; new structures measured; no constant-only diversity claim |
| Trial interruption between ask and launch | Same trial/parameters on resume; no duplicate allowance or tell |
| Two task boundaries and crash | Complete state, no lost spent work, <=24h each; actual sustained duration separately reported |
| OEIS prefix-only solution | Rejected using admission terms; never reaches learner archive |
| Insufficient discovery yield | Precise deficit and stopped self-learning branch; other software acceptance still testable |
| Learned component and invalid expansion | Real proposal/verified update for valid component; invalid or cyclic expansion rejected |
| Short combinatorial task | Real four-input update, restore and generation; exact n=0..8 scoring; no numeric padding or mixing into main 20/100 gate |
| Refuted target/proof | Descendant quarantine and honest claim status |
| Soft target missed / hard invariant violated | A pre-reserved diagnostic may inspect the first; the second stops/quarantines affected work. No failed gate becomes PASS |
| Missing/truncated required report evidence | INCOMPLETE/invalid verification; original producer failure preserved |

### Bounded component search (T067)

The installed `extract-components`/`verify-components` role exports
`search-library.json` alongside its immutable library. This self-contained input
binds the committed library bytes, original expansion proofs and spent mining work;
it does not contain training source solutions. Freeze its original reference in a
`library` model-channel envelope with profile `progressive-component-search/v1`
and the same compatibility identity as the recipe/checkpoint. Research requires a
native-verified research library. The generator accepts exactly one such library.

Library generation runs one correct-conditioning portfolio: ordinary neural draws
occupy slots 0–3 and typed component proposals occupy slots 4–7. Each component slot
expands and scores at most two deterministic typed AST proposals with the unchanged
decoder, then fits only the selected full source. All eight slots share the existing
60-second deadline; late scoring is discarded. Expanded codec tokens, conservative
expansion work, model-scoring tokens and native requests consume the frozen role
allowance. Original discovery work and per-prompt downstream costs are reported
separately. Freeze the paired no-library generator with `oeis_generation` data
`{"methods":["correct"],"forced":null}` and no library to give it eight ordinary
attempts under the same portfolio limit. T068 supplies distillation, complete model/library cuts and matched usefulness diagnostics;
search capability alone is not a finding of improved capability or efficiency.

### Component distillation and scoped dossiers (T068–T069)

`components.json` is a non-executable paired two-seed blueprint. Resolve its original
initializer, library, training pool, harder/retention panels, qualification and task
slots before freezing. In the installed admission role, pair a `component_proposal`
candidate with a `component_training_request` continuation containing the original
committed library input and training member. Successful independent admission emits
`component-target.json`. Freeze at most 64 targets through
`component_distillation.freeze_distillation`, place that package in the learner's
`component_distillation` field, and bind the same `component_rules` before resolving
online targets. The control package uses `arm="no_library"`, `library=null` and no
targets. OEIS and component mixtures are separate profiles. Complete checkpoints
embed the component input and bind both library and target identities. Changing
those cuts during a task is rejected. A predeclared unspent successor may adopt a
new committed cut atomically; ancestor model state and charges survive.

`claims.seal_candidate` binds an original generator selection, exact source,
source snapshot/offset, library and generation budget before literature review.
`create_dossier` independently executes its 100 declared terms and publishes a
separate original execution record, counterexamples, assumptions, known comparisons
and explicit untested scope. Native execution is mandatory for research. Reference-
only fixtures are diagnostic. The source remains confined to the review channel.

The installed `review-claim` operation accepts one `sealed_dossier`, one
`review_evidence` request and optionally a frozen `reviewer_registry`. The registry
is an explicit trust input naming independent reviewers and their exact scoped,
candidate-bound prior-work artifacts. A supplied model name or evidence digest does
not register a reviewer. Projection copies and verifies the complete original
candidate/execution/review closure, and the installed role independently repeats
native execution. Review publishes a new dossier with the original as parent;
it never rewrites evidence or creates a training target. Proof and novelty have
separate statuses. A `proved` request must reference a source-bound
`progressive-polynomial-certificate/v1` artifact. The registered checker independently
normalizes a pure return using integer literals, `$index`, `add`, `sub` and `mul`.
Its coefficients are canonical decimal strings in ascending degree, with no trailing
zeros (the zero polynomial is `["0"]`). For example:

```python
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.polynomial_certificate import check

source = "(program (return (mul (add $index (int 1)) (add $index (int 1)))))"
certificate = {
    "version": "progressive-polynomial-certificate/v1",
    "source": digest(source),
    "coefficients": ["1", "2", "1"],
    "arithmetic": "mathematical_integers",
    "assumptions": [],
    "machine_domain": {"lower": "-100", "upper": "100"},
}
checked = check(source, certificate)
assert checked["scope"] == "program_equals_polynomial_only"
assert checked["machine_safety"] == "finite_numeric_range_verified"
assert checked["sequence_law"] == "unproved"
```

Use `machine_domain: null` for mathematical identity alone. A finite domain also
requires a conservative interval argument for every original intermediate and the
fixed input/output limits; cancellation does not erase unsafe intermediate work.
Neither case establishes fuel, memory or deadline sufficiency, unbounded machine
safety, an unknown sequence law or novelty. Recurrences, loops, asymptotic claims,
nonempty assumptions and legacy IR proof labels cannot promote a claim. Unsupported
claims can retain finite/conjectural status. Counterexamples block proof promotion.

The checker caps source size, nodes, nesting, degree, coefficient bits, arithmetic
work and wall time. Optional `Limits` may only tighten these caps. Installed review
pre-reserves five attempts/rejections and one native dispatch, charges each symbolic
replay before work and retains spent charges on failure. The new dossier records the
checker identity and exact proof scope, and validation replays the certificate.

## Refutations, short sources, public reports and final assessment

The campaign accounting root owns `quarantine/`. Register exact target/archive/
library/update/checkpoint/dossier dependencies before consumption. Refutations
append immutable overlays; checkpoint restore and component adoption reject
marked descendants, including consumers registered later. `Journal.recovery`
selects only a registered clean checkpoint and preserves the supplied spent-cost
record; a missing clean ancestor is `CONTAMINATED`. Actual retraining still uses a
separately reserved successor. Never erase exposure to recover eligibility.

The external-source adapter supports complete permutation collections n=0..8,
first/second moment totals and the separate `combinatorial-short/v1` profile.
Source definitions stay in custodian preparation. The generator sees four terms;
admission/scoring independently check nine. `short_update` uses the canonical
recoverable learner transaction. The source-pilot blueprint stays non-executable
until snapshot, permission, grouping, initializer, runtime and reservations are
resolved. Related/duplicate statistics cannot cross roles; insufficient independent
groups yield a deficit. See [source-suitability.md](validation/source-suitability.md).

`discovery report --run DIR --output FILE.zip` automatically collects original
owned campaign evidence. An explicit `public-report.json` remains supported;
`progressive_evidence.public_report_input` seals the closed table request.
Required tables are roster, outcomes, accounting, ancestry and qualification
producers; exact requirements belong to the frozen public scope. No recursive
workspace export or weight/private-continuation/raw-corpus table is available.
Without scientific tables, terminal export remains `REPORT_INCOMPLETE` and retains
the original terminal statuses, first producer error and collector error references.

Before freezing a campaign, publish a sealed `progressive-public-scope/v1` record
with `source`, `scope`, `seeds`, `design`, `roster` (canonical public-roster digest)
and `qualification_producers` (scope-to-original-producer identity). Include its
artifact reference in the campaign with verifier role. The report request carries
that original reference as `scope_record` and the complete sealed `campaign_record`.
Verification checks their byte/hash closure and refuses a resealed narrower roster,
seed set, decision design or required evidence scope. Missing frozen authority
prevents `VERIFIED`. The ZIP is bounded to 512 MiB uncompressed metadata with
canonical partitions; a required record that cannot fit is explicitly incomplete.

`discovery verify-bundle --archive FILE.zip --output NEW_DIR` runs without Torch,
Optuna, native workers or private truth. It independently reduces outcome cells,
censoring, family weights, conditional paired-seed bootstrap bounds, decisions,
ancestry and charged/useful/discarded accounting, and verifies exact qualification
producer inventory/bytes. A verified negative scientific result remains negative.

`discovery assess` requires a sealed `progressive-locked-selection/v1` containing
campaign identity, reserved packaging task, final checkpoint reference, optional
library reference, frozen assessment-panel channel references and sealed candidate
channel reference. Candidate checkpoint/library bindings must match. Compatible
feature acceptance and a currently inspected qualified runtime remain mandatory.
The explicit command durably locks selection and marks panel exposure before
assessment dispatch; interruption cannot make that panel untouched. Ordinary run,
resume and adaptive installed operations reject an exposed campaign. Assessment
truth has its own channel and cannot enter learner/proposer/review/scoring channels.

## Actual HIP qualification handoff

T082 software is installed, but actual qualification is pending in this workspace:
mandatory device mode fails because no HIP build/runtime is available. CPU/native
checks do not satisfy this gate. `configs/foundation/progressive/acceptance.json`
is a non-executable blueprint; resolve selected profiles, source/image identities,
engine, actual device facts and persistent output in the pinned image first.

Inside the selected Docker/Podman HIP image, construct the request with
`progressive_qualification.matrix(profiles, source=..., image=..., engine=...,
hardware=hardware_facts())`. `source` is the canonical digest of the installed
`source_snapshot()`, and `image` must match `OEIS_QUALIFICATION_IMAGE`. Set
`OEIS_QUALIFICATION_ENGINE` to the inspected engine (`docker` or `podman`), matching
the request; the portable container-stage launcher supplies both identities. The matrix
covers only exact selected profile/dropout/input-length/microbatch paths and runs
serially; there is no wildcard qualification of changed settings or hardware.

Set `OEIS_REQUIRE_PROGRESSIVE_HIP=1`, `OEIS_PROGRESSIVE_HIP_MATRIX` to the sealed
request file, and `OEIS_PROGRESSIVE_HIP_OUTPUT` to persistent workspace output, then
run `python -m pytest -q tests/integration/test_progressive_gpu.py` inside that image.
The mandatory test rejects absent HIP, absent resolved scope/output, CPU fallback,
nonfinite updates, unchanged parameters, failed full restore/deterministic
continuation, missing generation/native scoring or unreclaimed owned tensors.
Original per-case producers persist and can be reused only under identical current
source/image/engine/hardware/model/input/update scope. A completed compatible matrix
can be replayed without overwriting original receipts. No actual HIP receipt or
sustained-duration claim has been produced here.


## Installed integrated diagnostic acceptance (T083)

The mandatory CPU/native CI inventory includes
`tests/integration/test_progressive_acceptance.py`. It builds one wheel, verifies
the installed source closure and executes 26 fixed production-path scenarios,
including positive and negative diagnostic branches, without depending on research
yield. CI retains `progressive-acceptance.zip` beside the foundation receipt. Its
sealed public records distinguish software COMPLETE from `qualified=false`,
scientific UNMEASURED and hardware/sustained PENDING. Private logs, truth and model
state are excluded. The command is implemented; its device phase still requires actual HIP and current original qualification.

The selected HIP matrix test also verifies reuse of its original producer bytes
without repeating the device matrix. `progressive_acceptance.hardware_plan` exposes
exact reused and missing capability cases; `run_hardware_delta` executes only the
missing cases serially in a fresh owned directory, with an explicit allowance of
at most 900 seconds. Current source/image/engine/HIP facts must match. Reusing Q
capabilities does not cover the campaign-dependent lifecycle/target-consumption
cases or sustained observations required by T090/T091. No current HIP result is
claimed by these CPU/native diagnostic results.


## Diagnostic templates and current preflight disposition (T084–T087)

`configs/foundation/progressive/diagnostics.json` contains all 16 contingency rows,
including sparse-support redesign and integrity quarantine. All templates start
disabled with unresolved immutable data/policy/design references, task/panel slots
and caps. Resolve them prospectively using
`oeis_learn.generalization.progressive.diagnostic_blueprints.resolve`; READY is a
pre-freeze proposal, never launch authority. The existing campaign freeze and
original evidence/gate remain binding. Selection reserves task and ancillary costs
from the existing pool (at most four tasks, 12h each, 24h aggregate); exposed panels
and repeated failure classes cannot acquire another episode. Integrity failure
stops affected execution without a diagnostic-budget waiver.

Verified public stdlib commands are `python3 -S scripts/foundation_run.py --help`,
`discovery init --reference-campaign PATH --output NEW_DIR` and
`discovery status --run DIR`. Init with missing original inputs returns a
non-executable blueprint and explicit deficits; status is read-only. The CLI
contract suite exercises parsing and invalid/failure states. These checks grant
no qualification. `--engine docker image ensure` and
`--engine podman image ensure` both returned exit 1 with the exact missing-CLI
reason. Install/use the workstation's existing engine, then verify its current
image/source facts before using the future sequence above. Local Podman support
remains; duplicate Podman CI jobs remain disabled.

Retain these artifacts together at their original owned workspace locations:

| Evidence | Required original artifacts |
| --- | --- |
| Software preflight | Source closure, uv.lock/package/image digests, engine facts, mandatory foundation receipt, JUnit, `progressive-acceptance.zip` |
| Diagnostic choice | Template inventory, selected immutable data/policy/design refs, task/panel slots, task and ancillary reservations, original failed decision |
| Qualification Q | Resolved selected matrix, hardware facts, original per-case `producer.json`, `qualification.json`, inspected engine/image/native identities |
| Preparation/freeze | Original retained inputs/exposure, prospective cohorts, measured preparation/iteration/storage records, blueprint, prepared manifest, campaign/runtime/tasks |
| Acceptance A | Exact original reuse plan, copied original producers, bounded delta cases/receipts and omissions, integrated campaign device evidence |
| Execution/recovery | Append-only task/campaign budgets and clocks, consumed/exposure cuts, full optimizer/scheduler/RNG checkpoints and ancestry, first failure plus collector diagnostics |
| Public handoff | One campaign `diagnostics.zip`, complete closed inventory and independent verification result; retain weights/private truth locally |

The small software-only ZIP is a CI diagnostic artifact, not the campaign's
research launch receipt. The selected hardware matrix is still unresolved (at
most 48 exact shape/dropout/input-length/microbatch cases); no fabricated
workstation/image/profile selection substitutes for original facts. Hardware and
sustained observations stay BLOCKED/PENDING until their original producers exist.
Unchecked optional research tasks mean unselected/not-triggered/deferred work, not
missing mandatory software. T085–T086 are complete with current-source Docker/native CI evidence.
T087 documentation is complete; the former missing command/role handlers are implemented.


## Immutable command plans and verified software routes

The command arguments in the sequence above are active. CPU acceptance tests
exercise command parsing, source/engine/profile/ref matching, missing-plan exit 2,
contract rejection exit 3, preserved engine failure exit 1 and successful bounded
orchestration. A valid qualification plan also reaches the engine boundary under
`python -S` without Torch, NumPy, Optuna or jsonschema. These tests do not claim a
physical HIP execution. `source-pilot` is the custodian's feasibility operation;
`verify` projects a bounded original public archive into the independent verifier.

Plans live under owned `run-state`, outside mapped source/build inputs, and use
workspace-relative byte/hash/path artifact references. Seal them with
`oeis_learn.generalization.progressive.contracts.seal` before adding their references to a blueprint or
campaign. Do not edit them after freeze. The separate base qualification retains
its original producer-relative reference namespace.

| Named reference | Closed plan fields in addition to `identity` |
| --- | --- |
| `qualification_plan` in blueprint references | `version: progressive-qualification-plan/v1`, `matrix`, `base_qualification`, `seconds`; optional `original_cases` array |
| `iteration_calibration_plan` in blueprint references | `version: progressive-calibration-plan/v1`, `seconds`, `caps`, `checkpoint_bytes`, `maximum_inodes` |
| `acceptance_plan` in campaign references | `version: progressive-acceptance-plan/v1`, `qualification`, `seconds` |
| `preflight_plan` in retained-input references read by init | `version: progressive-preflight-plan/v1`, `cohorts`, `exposure`, `sensitivity` |

The matrix is the sealed selected request produced by
`progressive_qualification.matrix`: actual source/image/engine/device facts,
resolved profiles, input lengths and microbatches. Obtain the base v2 qualification
through the existing foundation producer first, using the same unchanged source
and immutable image pair. Discovery qualification freshly inspects that original
authority and adds actual selected device-case proofs; it neither rewrites the
base PASS nor reuses an unrelated/stale source receipt. The base qualification's
CPU/native stage includes the discovery test inventory and precedes its HIP work.

`seconds` is finite (at most 3,600 per command plan) and must fit the selected
prospective/campaign allowance. The qualification wrapper subtracts inspection
time from its device deadline. With `original_cases`, only exact matching producer
bytes are reused; the remaining bounded delta retains original execution costs
separately from new elapsed time. Original case files survive failures for a
compatible new output/plan; do not overwrite earlier outputs. No additional pool
is created.

For a training campaign, add `iteration_calibration_plan` before running the
existing `discovery prepare` command. Preparation first resolves the immutable
learner channels and every finite trial choice, then runs the exact-input
calibration serially inside the qualified GPU image. It measures two complete
updates (including initial Adam allocation), two boundary probes and full-state
publication/restore at the maximal prepared input/target sizes. Identical
learning compatibility across continuation tasks shares one original measurement;
generic Q cases do not substitute for this calibration.

The plan's `caps` is the closed resource vector used by campaign budgets. Its
`elapsed_us` must equal `seconds * 1000000`; exact calibration may reserve up to 86,400 seconds including inspection and cleanup, with at most 3,600 seconds per serial input. Other handoff plans retain their 3,600-second limit. Calibration caps **plus** the prepared campaign caps must
fit the blueprint's proposed caps. Reserve at least the two-update token charge
for every distinct compatible input, two checkpoint generations per input, copied
input packages and 16 MiB metadata headroom. `checkpoint_bytes` bounds one complete
generation (including conservative optimizer growth); `maximum_inodes` bounds the
whole calibration tree and must cover at least 64 per input plus 256 metadata
entries. Exact calibration covers the 648 registered shape/policy combinations at both screening and confirmation trajectories, plus distinct source/library inputs. Its inventory ceiling is derived from the existing one-million-inode bound (64 per input plus 256 metadata entries), rather than the separate 48-case generic Q matrix. Every case must still fit the original token/time/storage reservation. Insufficient reservations
return a deficit before device dispatch; they are never enlarged automatically.

Retain `prepared/calibration-input.json`, `prepared/calibration/request.json`,
the per-input `cases/*/producer.json`, the aggregate calibration producer and the
original debit under `run-state/discovery-calibration`. The final `prepared.json`
binds these originals and records twice each measured phase in each task/choice's
iteration budget. Freeze independently replays the setup closure and includes the
measured checkpoint size in its storage graph. CPU measurements remain diagnostic;
research preparation requires actual HIP on the matching source/image/device.

The calibration reservation is consumed before dispatch and is tied to the
blueprint, not the output directory. On timeout or failure, retain the original
error and debit; a sibling output cannot retry with a fresh allowance. Reuse a
successful prepared output, or prospectively revise the blueprint and its setup
reservation after examining a failed attempt. Research preparation without the
required plan reports `iteration_calibration_plan` as a deficit.

Acceptance builds/probes a wheel and runs all 26 CPU/native cases first. It then
reinspects the frozen runtime and executes the seven registered device scenarios
in one owned GPU stage using the selected Q profile: two linked tasks/recovery,
OEIS admission/replay, mined-component use, four-term source updates/generation,
and native-verified refreshed structure consumption. Q requires matching 4/20
input and accumulation scopes. The inspected image/device/numerical policy must
still match; CPU fallback, skipped/missing cases and stale producers fail closed.
The installed HIP numerical startup flags are set before Python starts.

Acceptance measures the selected profile's actual parameter and persistent-buffer
sizes on CPU before learner dispatch. Its finite diagnostic checkpoint envelope
includes weights, both Adam moments, serialization/context headroom and a second
allowance; it is rounded upward rather than retaining the small fixture's 64 MiB
limit. Task disk partitions cover simultaneous generations, imported/projected
predecessors and retained outputs. The campaign graph additionally reserves all
task partitions, projections, journals and packaging, checking real free bytes
and inodes before publishing the fixture. OEIS and four-term source fixtures use
the same bounds. `acceptance-limits.json` records the profile and measurements.
These are diagnostic resource bounds, not substitutes for exact research timing
or actual HIP qualification. No task time ceiling is enlarged by this sizing.

Retain the original CPU suite receipt/cases/JUnit, device request/producer/JUnit,
Q reuse plan, frozen campaign/runtime, and qualification closure. Acceptance
replays the original case results and writes one bounded `diagnostics.zip` with
metadata only. `acceptance.json` becomes visible only after packaging succeeds.
`run`, `resume` and `assess` replay these original feature producers and freshly
inspect runtime facts; a supplied qualified bit alone grants no research launch.
The later study `report`/`verify-bundle` route remains the independent statistical
reconciliation interface for the final campaign report.

The init preflight can inventory original cohorts/exposure and validate a
prospective sensitivity design without model scores or qualification. It reports
`VALIDATED_UNEXECUTED` for the design: running the simulations still needs its
separately frozen preparation allowance. No final research/diagnostic freeze is
claimed before Q. The source-support and report APIs retain explicit deficits.

For T091 software preparation,
`oeis_learn.generalization.progressive.preflight.sustained(plan, allowance_us=...)` requires
two named linked tasks with finite active durations <=24h each, an observation
interval <=60s, >=30min finalization per task, a declared useful-work interruption
and explicit memory/disk/thermal/throughput availability or missing reasons. It
checks the existing total allowance and returns `PENDING_ACTUAL_DURATION` with
`actual_elapsed_us=null`; it launches no soak or research.

### Closed host cycles and newly produced component inputs

Use the existing `prepare`, `freeze`, `run` and `resume` commands. A frozen task
request can select `oeis-cycle` or `source-cycle`: `artifacts.cycle` contains one
original declared plan path, `image` is immutable, `seconds` is finite, and
`reservation` is the zero resource vector. The host orchestrator spends only its
named child tasks' original allowances; it is not an installed arbitrary role.
Training, model and plan channels must be declared before freeze. Exact-input
calibration resolves the child templates before future admitted cuts exist.
Research targets outside the calibrated token/numeric envelope are rejected.

OEIS cycles retain the original output owner, phase receipts, search/admission
costs, archive cut and learner packages. Resume reuses committed phases and full
learning state; a sibling output cannot reset costs or buy a second cycle.
Uncommitted learning needs supervisor recovery proof. An uncommitted search
without replay authority fails explicitly. Yield deficits retain spent search
costs and prevent an unvisited paired slot from restarting the cycle. Completed
child training results are registered as original predecessor outputs; missing or
foreign cycle receipts cannot supply later checkpoint slots.

An assisted `progressive-source-cycle/v1` plan binds `roster` (original `sources`
and corresponding `roles`), four `training` entries (`task`, `channel`, `model`),
per-role `seconds`, and `generation`. Select two matched with/without-source seed
pairs with fresh initialization. Each model has one recipe/checkpoint pair.
Independent source support is mandatory; this bounded pilot permits at most 128
training and 128 evaluation members. Generation and learning see four features;
admission and held-out scoring independently check nine terms through separate
truth channels. Full restore preserves the short profile and mixture. These
results remain ineligible for primary scaling; forced candidates are diagnostic.

Component requests can consume original committed predecessor outputs, for example:

```json
{"task":"mine","operation":"extract-components","output":"library","kind":"library"}
```

Generation publishes `component_proposals` with a frozen `member` selector;
admission publishes `target` with kind `admitted_target`. Training combines those
slots with its original training request at a complete recoverable task boundary.
Do not prebuild the future library or copy unowned output paths into a request.
Campaign/task, arm, seed, dependency, profile and compatibility checks apply. The
no-library control resolves its empty cut without receiving discoveries. The
component-enabled portfolio bounds neural decoding to the first half so its
four component slots remain reachable within the same 60-second ceiling;
historical eight-neural portfolios retain their original policy.

Installed review retains independently checked counterexamples and original
dossiers before reporting a denied promotion. Host reconciliation appends
transitive target/library/update/checkpoint/dossier quarantine without rewriting
history or refunding cost. Resume, generation, library adoption and positive
review reject affected lineage. Recovery needs a known clean ancestor under its
original authority; otherwise retain the explicit contamination result.

### Automatic terminal evidence collection

`discovery report --run DIR --output NEW.zip` automatically collects the original
campaign, frozen public scope/cohort metadata, committed role and host-cycle
results, qualification-case producers, checkpoint identities, ancestry, outcomes,
resource vectors, decisions and available observations. Explicit
`public-report.json` requests remain supported. Collection follows the owned
invocation/output inventory rather than recursively exporting workspace files.
Private continuations, weights and raw corpora are excluded; bounded original
producer tracebacks and subsequent collector errors remain attributable.

Freeze one `progressive-public-scope/v1` authority with the full comparison
roster, paired seeds/design, required tables, qualification producers and accounted
tasks. `verify-bundle` independently reconciles those originals. A complete
negative result can yield `NO_MATERIAL_GAIN`; partial/failed work retains spent
costs and unresolved denominators. Missing scope, qualification, required evidence
or collector errors prevent verified status. Altered caps or unsupported gates
are invalid. Required evidence that cannot fit the frozen package allowance and
512 MiB metadata ceiling remains explicitly incomplete with local originals kept.


## Convergence acceptance and comparison handoff

The installed inventory now includes canonical extraction/admission/learning,
short-source prepare/freeze/run/resume, post-learning refutation, automatic
report/verify and one selected finite trial choice with interrupted restore and
full-state continuation. The seven device nodes use the selected architecture
and dropout; the finite choice case preserves that exact scope. CPU/native
counterparts exercise orchestration, with original device facts required for
actual-HIP receipts. T082/T090 and sustained/research executions still require
their original physical producers.

Freeze one `progressive-public-scope/v1` record named `public-scope` for the
primary summary. Additional original scopes remain in the same campaign package.
When the same panel/design is scored at multiple checkpoints, set its optional
`comparison` field to the original `progressive-gate-registration/v1` identity.
For an OEIS cycle, use `digest([cycle_plan_identity, baseline])`, with baseline
`starting` or `continued_synthetic`. Both comparisons are always retained and
independently replayed; ambiguous primary selection is incomplete evidence.

Comparison export includes complete original panel membership, paired producer
bytes, checkpoint/owned-output bindings and declared cost-task closure. Explicit
exploratory actions carry their original decision, prior transition cut and
budget prefix; the verifier replays their trigger and remaining reservation.
A missing original binding cannot receive VERIFIED. Original artifacts above
one table-row limit are partitioned without changing bytes; evidence exceeding
the frozen package reserve is explicitly omitted and prevents verification.
Repeated collector errors retain occurrence counts without breaking the closed
manifest. Weights, private evaluation values and training corpora stay outside
this public package.

Source-cycle research training has its own finite allowance, which must fit
startup, measured complete-iteration work, the thirty-minute finalization reserve
and all enclosing search/admission/evaluation work. Search/admission/scoring role
limits remain separate. Acceptance fixtures reserve retained phase inodes before
freeze; their shorter diagnostic reserves do not establish research feasibility.

### Post-012 hourly persistence in new research packages

New research construction binds `checkpoint_policy: hourly/v1`. Full model,
optimizer, controller and RNG snapshots publish hourly and at clean pause,
milestone and task boundaries. Per-update token charges remain durable. A crash
can discard the latest window of model progress, never its costs. Live progress
shows both completed and recoverable exposure. Historical inputs without this
field keep their original per-update policy; original rehearsal/data outputs are
not rewritten or rerun by this change. Current-source HIP checks are still required.

The prior physical reference-model measurement gives approximate hourly-policy
component projections of 3.2/6.2/12.2/24.2 minutes for 500k/1M/2M/4M tokens and
150 minutes for 25M tokens. These use measured update, probe and snapshot costs;
whole-learner journal/retention overhead and allocation-dependent throughput are
not yet measured on HIP. Evaluation, image startup, qualification, preparation
and recovery are separate. These figures are not campaign ETAs or scientific
support for shorter training.

Do not rerun the old `measure-training` command in a sibling session to obtain a
new allowance. Its completed evidence remains immutable. A current-source HIP
validation/learning pilot must have its own prospectively bound scope. Before
freezing a shorter pilot, derive support with `prepared_support.recadence` using
the complete trajectory horizon and the first planned look, retain both returned
publications, and require `COMPLETE`. Construct the learner input from that derived
support via `training_input(..., checkpoint_policy='hourly/v1')`. Never change
cadence independently between cumulative screening looks or silently recycle
shards. No scientific horizon or existing study budget has been changed here.

### Full installed-learner runtime measurement

Run this once under a separate finite diagnostic owner. It reuses the completed
service and original component-training sessions, never the raw corpus or a new
census. It builds/reuses the current HIP image, runs seven 128k-token paths with
hourly persistence and full-state successor restore, then four independent
maximal-input calibration samples. Scientific horizons and budgets remain unchanged.

```bash
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman \
  --session run-state/discovery/013-runtime-timing-001 \
  measure-runtime \
  --service-session run-state/discovery/013-services-001 \
  --training-session run-state/discovery/013-training-timing-001
```

The provisional expectation is **20–40 minutes plus any image build**, based on
prior component measurements, not a measured end-to-end ETA. The finite guard is
much larger: 45 minutes per learner case, 15 minutes per calibration case and
6 hours 35 minutes overall. Each learner case contains two tasks with 20-minute
allowances and 60-second finalization reserves. Those are failure guards, not
scheduled durations. A much slower case retains partial work and diagnostics;
unrelated cases continue after proven cleanup and a device health check.

Share `run-state/discovery/013-runtime-timing-001/share/measure-runtime.zip`, even
if the command fails. It contains exact source/image/command bindings, environment
telemetry, per-case process/health logs, producer failures, bounded update windows,
checkpoint/probe/accounting timings and `runtime-review.json`. Training rows and
weights remain local. A repeat invocation reuses completed steps and retains
failed originals; it never creates another free training attempt. Collecting the
existing session does not rerun training.

The report estimates the currently selected capability/diversity/tuning round,
including incremental continuation tokens and historical evaluation costs. Its
low/high values are sampled planning scenarios, not confidence intervals. The
historical generation estimate uses slow random-model measurements; training can
change generation cost substantially. Four batch-16 calibration samples project
preparation overhead but do not bound every batch/dropout/input combination.
Unmeasured setup, remaining qualification, later conditional branches and final
assessment remain explicit. Timing alone does not select shorter trial horizons
or establish scientific progress, sustained execution or HIP qualification.

### Measured runtime reservation and completed-timing cleanup

Use the new operational reservation only in a fresh research preparation. The
measured initial capability/diversity/tuning workload remains approximately
367–401 hours plus preparation, not the much larger hard ceiling. Exact calibration
samples project about 13.15 hours before overhead; the 72-hour calibration cap
allows substantial variation. These estimates exclude later conditional stages.

The following command reuses the original completed native data and sensitivity
from round 002, retains rehearsal 007 as historical evidence, obtains fresh
current-source qualification/evaluation timing/exact calibration, freezes and
accepts the selected campaign, then runs it only if every gate passes. It does
not rerun the census or rehearsal and does not expose assessment:

```bash
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman --session run-state/discovery/013-research-round-003 \
  prepare-research \
  --registration-session run-state/discovery/013-research-inputs-003 \
  --rehearsal-session run-state/discovery/013-hip-007 \
  --reuse-preparation-data run-state/discovery/013-research-round-002 \
  --expanded-runtime-budgets --run-after-prepare
```

Before launching that work, the completed runtime/training timing models may be
retired. Do not delete either session wholesale: their original observations and
inputs remain useful. Planning is read-only apart from the maintenance report and
lock files, and reports the precise candidate bytes plus a broader disk inventory:

```bash
python3 -S scripts/discovery_cleanup.py plan \
  --session run-state/discovery/013-runtime-timing-001 \
  --session run-state/discovery/013-training-timing-001 \
  --output run-state/maintenance/timing-cleanup-001
```

Review `run-state/maintenance/timing-cleanup-001/plan.json`, then apply:

```bash
python3 -S scripts/discovery_cleanup.py apply \
  --output run-state/maintenance/timing-cleanup-001
```

Apply hashes all selected tensors before deleting any, preserves a durable
inventory, and safely resumes with the same command after interruption. It keeps
all original metadata, logs, budgets, input references, state manifests and ZIPs;
the disposable timing models become non-resumable. It refuses active/unreconciled
GPU owners and changed/aliased files. Share
`run-state/maintenance/timing-cleanup-001/cleanup-report.zip` for review of the
remaining storage. No generalization, rehearsal or research run is automatically
removed. In particular retain `pac-012-bcd-r1-002`, all research-input sessions,
round 002's original native materials and service measurements. Unknown older
runs and `tmp-validation` require a dependency review, not a name-based `rm -rf`.
