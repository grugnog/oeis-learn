# Phase C attempt and non-HIP local checks

Phase C is **incomplete**, with a failed Phase B prerequisite. No HIP command,
container command, training, research campaign or acceptance session ran.

The Spec Kit preflight located this feature and both reviewer checklists passed
(requirements 16/16, design 15/15). No extension hooks exist. Existing Python and
Docker ignore files cover the generated local artifacts.

Draft evaluation adapters verify actual checkpoint file identities, mark absent
private weights unavailable, project visible prompts to callbacks and call Phase B
portfolio/diagnostic policies. Draft recorded reporting preserves fixed denominators
and missing checkpoint identities. These adapters are not registered public commands
and have not been connected to the actual native/model container roles.

Executed:

```bash
PYTHONPATH=src /workspace/scratch/5b234f116159/phase-a-venv/bin/python -m pytest -q \
  tests/integration/test_generalization_evidence.py \
  tests/unit/test_generalization_*.py tests/contract/test_generalization_*.py
```

Before the new counterexample: **63 passed**. This covers the two new adapter tests
and existing feature policies/contracts; it does not establish Phase C completion.

Then expanded the offline evidence fixture to replace the family artifact with an
invented family having 999 successes, recalculating the file reference and outer seal.
The verifier should reject it even though every hash is valid. Executed:

```bash
PYTHONPATH=src /workspace/scratch/5b234f116159/phase-a-venv/bin/python -m pytest -q \
  tests/unit/test_generalization_recovery.py
```

Result: **1 failed, 4 passed**. Failure:
`test_offline_complete_bundle_and_stale_summary[True]` — expected `ValueError`,
none raised. The regression is deliberately retained without skip or expected-failure
masking. T037/T044 are reopened; integration stops at this failed prerequisite per
`speckit-implement`. The single algorithm-phase rule requires repairing/reviewing this
inside reopened Phase B before continuing routine Phase C wiring.

T063 is only partially attempted. T064–T070 remain uncompleted; no fabricated image,
preflight, archive or HIP evidence is recorded. The requested no-HIP scope excludes
T066, while the other gates require completed integration or a usable engine.

## Subsequent repair

The failing family regression above now passes. Strict reconciliation also rejects
false seed/cost/bound/coverage/aggregate projections. The combined feature, draft
adapter and historical regression selection passes 131 tests and 4 subtests. See the
latest algorithm-review entry. C/D remain incomplete; no HIP command ran.

## Preparation/recipe integration increment

Executed the following local non-HIP selection after the preparation, role,
recipe and CPU/source-registration changes:

```bash
PYTHONPATH=src /workspace/scratch/5b234f116159/phase-a-venv/bin/python -m pytest -q \
  tests/unit/test_generalization_*.py tests/contract/test_generalization_*.py \
  tests/integration/test_generalization_*.py tests/unit/test_bigint_reference.py \
  tests/contract/test_bigint_contracts.py tests/unit/test_pilot_followup.py \
  tests/unit/test_pipeline_reuse.py tests/unit/test_pipeline_bundle.py
```

Result: **136 passed, 4 subtests passed**. `git diff --check` passes. Reviewer
checklists remain read-only and passing (requirements16/16, design15/15); no hooks.

New coverage exercises per-cell exact materialization, production/reference
agreement through an injected native interface, failed quota publication, immutable
output roots, role-channel mount resolution, sealed recipe metadata, and changed or
unmapped authority invalidating source reuse. The native interface is synthetic in
these local tests. Actual container-role launching is not established by them.

Tool inspection: `docker`, `podman`, and `cargo` are unavailable. T064's image
build/inspection cannot run in this environment. T063's local portion ran; its
qualification-container portion remains pending. No HIP command, model training,
acceptance invocation or research campaign ran. T065/T067 depend on the still
unimplemented canonical helper/archive integration; no dry-run/archive success is
claimed. T068 requires actual acceptance evidence and remains pending.

## Full-state/update/replay adapter increment

Added a prospective `GeneralizationCheckpointStore` with immutable private state
generations and atomic latest publication. All blob references are verified before
state deserialization. Added an update adapter that persists the pending batch and
debit before execution and commits after a complete update, plus a callable adapter
to the existing `_batch_loss` implementation. Local tests inject JSON serialization
and a deterministic update callback; actual Torch model equivalence is not claimed.

Replay adapters call the bounded native-client interface, independently audit all
training terms, preserve empty yield, and rerun admission against the training roster
when restoring a pool. Tests use a synthetic native client and verify visible-only
proposal inputs, eight charged attempts, task deduplication and exact snapshot restore.

The same non-HIP selection above now passes **140 tests and 4 subtests**.
`git diff --check` passes. No HIP command, actual learner training, container build,
or research campaign ran. T051–T053 and T061–T062 remain partial until canonical
worker registration and full production integration are implemented and verified.
This increment does not complete all requested C/D tasks.

## CPU learner and qualification integration continuation

The latest combined local selection passed **178 tests, 9 subtests**, with **one GPU
test deselected**. The selection was:

```bash
PYTHONPATH=src python -m pytest -q -m 'not gpu' \
  tests/unit/test_generalization_*.py tests/contract/test_generalization_*.py \
  tests/integration/test_generalization_*.py tests/unit/test_bigint_reference.py \
  tests/contract/test_bigint_contracts.py tests/unit/test_pilot_followup.py \
  tests/unit/test_pilot_followup_host.py tests/unit/test_pipeline_reuse.py \
  tests/unit/test_pipeline_bundle.py
```

Unlike the earlier callback-only increment, this selection executes actual CPU
Torch updates and checks uninterrupted/resumed state equivalence for static and
refreshed data with literal and parameterized targets. T045 and T051 are checked;
remaining integration tasks retain their unchecked status.

Role projections now accept repeated identical references while rejecting conflicts.
A regression reproduced the duplicate-reference failure before the fix. Qualification
producer allocation now leaves 600 seconds outside the canonical runner's 6600-second
cap for post-run inspection and verification (480 seconds of explicit reservations).
The producer's failure path is exercised with a fake runner, without HIP execution.

An actual HIP checkpoint-load/finite-update test is registered but has not run.
Neither Docker nor Podman is available in this environment. Container qualification,
image inspection, hardware acceptance and scientific campaign results are not claimed.
Canonical CLI, scoring, replay and complete bundle integration remain unfinished.
