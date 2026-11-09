# Validation and workstation handoff

**Planning artifact: the `discovery` commands below are proposed and unavailable until implementation. Do not paste the research commands into the current checkout expecting them to run.** This guide specifies the future end-to-end acceptance interface; it does not assert completed gates.

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

The implementation task list must create the named progressive tests before these commands are used:

```bash
uv run pytest tests/unit/test_progressive_*.py tests/contract/test_progressive_*.py
uv run pytest tests/integration/test_progressive_*.py -m 'not gpu and not docker'
```

Include meaningful cases for every row in plan.md's traceability table. Required real tests: logical-batch accumulation equality, immutable draw/reject replay, complete state after interrupted update, trial idempotency, cross-role truth rejection, prefix-overfit admission failure, component expansion/refutation and independently recomputed gates. Container/native and actual-HIP coverage comes through `discovery qualify` and `acceptance`; local CPU tests do not substitute for them.

## Proposed consolidated workstation sequence

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

`discovery report --run DIR --output FILE.zip` packages explicit public tables from
`public-report.json`. `progressive_evidence.public_report_input` seals this request.
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
the installed source closure and executes 22 fixed production-path scenarios,
including positive and negative diagnostic branches, without depending on research
yield. CI retains `progressive-acceptance.zip` beside the foundation receipt. Its
sealed public records distinguish software COMPLETE from `qualified=false`,
scientific UNMEASURED and hardware/sustained PENDING. Private logs, truth and model
state are excluded. The `discovery acceptance` command warning above still applies.

The selected HIP matrix test also verifies reuse of its original producer bytes
without repeating the device matrix. `progressive_acceptance.hardware_plan` exposes
exact reused and missing capability cases; `run_hardware_delta` executes only the
missing cases serially in a fresh owned directory, with an explicit allowance of
at most 900 seconds. Current source/image/engine/HIP facts must match. Reusing Q
capabilities does not cover the campaign-dependent lifecycle/target-consumption
cases or sustained observations required by T090/T091. No current HIP result is
claimed by these CPU/native diagnostic results.
