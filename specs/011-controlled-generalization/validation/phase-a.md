# Phase A implementation evidence

**Date**: 2026-10-30. **Scope**: T001–T008 only. Phase B algorithms and C/D integration/hardware gates remain pending.

## Completed work

| Task | Delivered evidence |
| --- | --- |
| T001 | Import-safe generalization package and protocol/interface skeletons. |
| T002 | Thirteen packaged Draft 2020-12 resources: eight authority formats, shared definitions and separate visible/training/scorer/weights-only shapes. Unknown record fields and incompatible versions are rejected. |
| T003 | Standard-library mechanical fixture generator, six reproducible JSON files, inventory hashes, independently computed terms/control-flow expectations and golden policy counterexamples. |
| T004 | [Algorithm handoff](../contracts/algorithm-handoff.md) with ownership, invariant matrix, reference cases, scope limits and B exit gate. |
| T005 | Immutable, separate input/output mount declarations for every role; inputs read-only and output destinations distinct. |
| T006 | Frozen/slotted request/result and artifact types, separate weights-only/full-state checkpoints, original producer references and immutable status/exit mapping. |
| T007 | Authority and role contract tests, including unknown/nested hidden fields, canonical integers, malformed paths/hashes, duplicate rosters, fake qualification flags and source-size bounds. |
| T008 | All nine proposed command routes return BLOCKED/UNMEASURED with no I/O; unknown operations raise. Public host command registration remains in C. |

## Checks actually performed

- **50 targeted tests passed**: 27 new generalization contract tests plus 23 existing bigint/follow-up regression tests. Exact selection:

```bash
PYTHONPATH=src /workspace/scratch/5b234f116159/phase-a-venv/bin/python -m pytest -q tests/contract/test_generalization_authority.py tests/contract/test_generalization_roles.py tests/contract/test_bigint_contracts.py tests/unit/test_pilot_followup.py
```

- Local isolated virtual environment: Python 3.12.14; pytest 9.1.1 and jsonschema 4.26.0 match the committed lock's versions. No runtime dependency was added. Initial tests failed on absent fixture resources; reference execution then exposed invalid local names in the mechanical fixtures, which were corrected before the final passing run.
- Fixture regeneration reproduced all six JSON files byte for byte. Inventory verifies the five payload files and excludes itself to avoid a recursive hash.
- A built wheel imported independently of the source checkout, contained all thirteen schemas and resolved local schema references offline for all eight authority fixtures. Build used ephemeral setuptools 84.2.1 / wheel 0.48.0; dependency/toolchain locks were not changed.
- Existing portable host CLI help succeeds; no prospective command is registered with that dispatcher in this phase.
- `git diff --check` passes. Existing specification/design checklists were inspected read-only: 16/16 and 15/15 complete. No extension hook file exists.

## Limits and next gate

Docker and Podman executables are unavailable in this environment. The checks above are local Python contract/regression and packaging checks, not container/native or HIP qualification. No training or accelerator work ran. Container test selection/source-closure registration is T060; consolidated hardware acceptance remains T066.

Shape validation is intentionally insufficient for acceptance: synthetic fixture identities and PASS/hip_available flags are not evidence. Phase B must verify actual bytes, ancestry, state, isolation, original producer receipts and execution. Role declarations do not enforce mounts by themselves; C must wire the declared boundary to the container runner.
