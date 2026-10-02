# Phase 5 review and remediation — 2026-10-02

Scope: T030–T037 (US3), based on merged phase-4 PR #5, main commit
`82c3826b1421fe932871863e3195eeefabf7f251`. The historical task audit marked
several files partial, but the current branch contained none of the sampler,
admission or pool modules. This implements them in the existing data package,
reusing the existing complete codec, supervised execution workers, independent
interpreter, artifact writer, cohort membership and CLI. No parallel model or
training framework is introduced.

## Findings and remediation

| Priority | Finding | Remediation / evidence |
| --- | --- | --- |
| Critical | No strict generic generator existed; the legacy SFT generator uses named families. | Add typed target-independent expression/statement sampling with the exact plan priors and explicit additional allocation/leaf rules. Source/config/counter provenance is reproducible. Label the existing named generator legacy/assisted and keep it out of the new path. |
| Critical | An origin label alone could not demonstrate generic provenance. | Admission regenerates the exact sample and checks complete source/token round trips. Imported OEIS/LODA, teachers, old replay/pretrained origins, changed counters/revisions and corrupt tokens are rejected. |
| Critical | No admission path combined exact100 production/reference verification and evaluator-owned heldout membership. | Reuse phase-3 supervised execution and finite verification. Enforce signed values, exact independent evidence, resource limits and the first20 membership set for every heldout group member. Membership works without private truth files; no rejection exposes a target ID or continuation. |
| High | Program/evidence/admission identities had no implemented acyclic storage contract. | Hash only the nine immutable program-core fields. Store outputs, production/reference evidence and decisions separately. Strict loading rejects stale IDs, missing conditioning, wrong tokens, self/cyclic references and mismatched outputs even when the outer file is rehashed. |
| High | Completion timing, duplicate outputs and crashes could change pool construction or lose state. | Fixed eight-sample chunks, ascending publication, complete source/output dedup and shortest-body/hash representatives. Persist counters, RNG identity, attempt journals, issued batches and dedup checkpoints. Test reverse future completion and interrupted versus uninterrupted example order. |
| High | An expired crash after a saved attempt but before its chunk checkpoint could omit issued work from summaries. | Journal the batch before execution. Recover durable decisions without starting workers after expiry; expose unknown issued outcomes as pending. Tests cover both attempt and chunk crash points, clock/boot changes and backward time. |
| High | Publication/report interruption could reopen a completed build or convert it into a shortfall. | Treat the immutable manifest as the commit point and reconstruct the same report from it. Inject failures after attempt, after chunk, before publication and after publication. Completed retries validate instead of resampling. |
| High | A naive independent-output comparison treated differing lengths after fuel exhaustion as numerical disagreement. | Shared-prefix value disagreements remain fatal; resource-limited incomplete execution is a recorded rejection. Regression reproduces 23 production outputs versus 24 reference outputs under a fuel limit. No shared runtime semantics were changed. |
| High | No verified narrow learner view existed; legacy missing-data behavior can generate teachers. | Controller verifies the full admission archive and exports only program/codec IDs, twenty exact conditioning values and complete body tokens. Learner loading requires the controller-approved view identity and has no truth, evidence or membership path. Missing/unverified pools fail. |
| Medium | `build-pool` exposed a legacy artifact-directory placeholder rather than the contracted arguments. | Implement `--config --cohort --output`, structured reports and correct success/input/gate/resource codes, including malformed YAML and bounded yield shortfall. |

The concrete [generic pool interface](../contracts/generic-pool-bundle.md)
documents the sampling projection, extra disclosed priors, artifact graph,
chunk stopping rule, archive/trainer role split and reproduction commands.
The final chunk may have surplus output classes; all its attempts are counted,
and only the first requested classes' chosen representatives enter training.
Source/RNG sampling does not depend on worker count, time limits, yield targets
or private cohort identities. Completion order cannot select easier examples.

## Validation and evidence

The final CPU suite passed **433 tests**, with **two pre-existing AMD/HIP
skips** and no failures. All **36 phase-5 tests passed** without skips.
The [software validation report](phase-5-software-report.json) records the final
test counts and environment. The [G3 pool report](phase-5-pool-report.json)
records the actual bounded CLI run, including rejections, output duplicates,
constant fraction, statement/leaf/constant counts and opcode coverage. A
[complete admitted evidence sample](phase-5-evidence-sample.json) preserves one
source, full tokens, outputs, both execution records and its decision. The
complete locally generated archive was checked by `load_pool`, including all
admitted candidates, dedup choices and the exported 64-example trainer view.

The first diagnostic run reached 64 published output classes from 624 attempts
in 89.24 seconds. It exposed no independent arithmetic disagreement. After the
partial-chunk accounting fix, the same frozen fixture was rerun; the linked
machine report is the **final** implementation's evidence: **624 attempts, 237
admitted programs, 66 unique output classes, 64 published examples in 86.72
seconds**. Of the published examples, **23 (35.94%) are constant**. There are
387 recorded rejections and 171 duplicate admitted output arrays. The generator priors,
seed, limits and target were not tuned between runs. The fixture's source-only
holdout is committed under `tests/fixtures/foundation/bootstrap/`.

CPU regression command, from the repository root:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src:. python -m pytest -q \
  tests/unit/test_foundation_*.py tests/contract/test_foundation_*.py \
  tests/integration/test_foundation_*.py tests/unit/test_readiness_policy.py \
  tests/integration/test_canary_qualification.py \
  tests/contract/test_canary_benchmark_contract.py \
  tests/contract/test_readiness_report_contract.py \
  tests/unit/test_checkpoint*.py tests/contract/test_cli_contract.py -ra
```

Validation includes the 36 phase-5 tests, the impacted foundation/legacy suite,
Spec Kit prerequisites, the project artifact/coverage/dependency validator,
offline lock consistency, wheel build/install/import, Ruff `E4,E7,E9,F` on changed
modules/tests and `git diff --check`. The requirement/design checklists remain
read-only and fully checked (19/19 and 22/22). No extension hooks are configured.
This is not a claim that every historical training/analysis test was run.

Environment: Python 3.12.14, Wasmtime 48.0.0, PyTorch 2.9.1+cpu, pytest 9.1.1,
jsonschema 4.26.0. Existing dependencies were reused from an isolated venv; no
system package, driver, Dockerfile or dependency lock changed. Local spawned
worker/socket IPC required execution outside the tool's restricted sandbox.

## Task status and handoff

T030–T037 are complete with CPU evidence. This phase requires no GPU computation.
The pre-existing phase-2 and phase-4 AMD/HIP tests remain skipped and their tasks
remain partial; this PR does not turn them into hardware evidence. All later
phase statuses remain unchanged.

The pool is a generic software fixture, not a trained model or a useful-OEIS-
coverage result. In particular, constants and duplicate output classes are
reported rather than hidden or removed by a named-family curriculum. Qualifying
training, full continuation/arm accounting, device residency, Docker mounts/
network/host quotas and run qualification remain T038–T049. The learner-side
file/message boundary is implemented; full container isolation is not claimed.
The accepted root Dockerfile and `lab-gpu` workflow are unchanged.
