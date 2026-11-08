# Validation and workstation handoff

**Planning artifact: the `discovery` commands below are proposed and unavailable until implementation. Do not paste the research commands into the current checkout expecting them to run.** This guide specifies the future end-to-end acceptance interface; it does not assert completed gates.

## Prerequisites

- Spec 013 implementation installed from the same checkout/source closure as its images; existing functional Docker/Podman and actual HIP workstation access.
- Retained original campaign, checkpoint/recipe/input producers and prior exposure manifests, including fresh-002. Start with the existing original campaign root; `init` inventories dependencies and reports all missing references together.
- Measured free disk/inodes and shared-memory capacity; a single GPU owner; exact selected settings and finite stage/campaign budgets.
- [CLI contract](contracts/cli.md), [experiment contract](contracts/experiment.md), [execution contract](contracts/execution.md) and [learning/discovery contract](contracts/learning-discovery.md).

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

Expected: blueprint, input/exposure inventory and explicitly named deficits. Resolve source/retained-artifact references locally. Inspect the proposed stage subset, trial spaces, task roster, all caps and estimated storage. The default subset is capability + diversity; select the full dependency graph only when its large finite reservation is intended. Freezing a blueprint is required before any research run, but defining it can use the already authorized study scope without asking permission at every internal gate.

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

Expected: actual CPU/native/HIP evidence, outcome-blind support census, immutable campaign, shortened all-stage acceptance labeled diagnostic, then only selected research stages. A sparse cohort, model infeasibility or incompatible receipt stops before dependent research. A negative research gate completes its experiment with a stop/inconclusive report, not a hidden retry. Record `mechanical_status` and `scientific_status` independently.

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
| Missing/truncated required report evidence | INCOMPLETE/invalid verification; original producer failure preserved |
