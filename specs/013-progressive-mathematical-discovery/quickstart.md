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
