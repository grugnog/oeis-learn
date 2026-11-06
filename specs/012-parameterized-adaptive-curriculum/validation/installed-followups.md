# Installed PAC follow-up integration

This is a separate stack on PR #63, based on `bbeacb4`, on branch
`012-pac-followup-integration`. The operator's R1 retry started on #63 before this
integration. Its checkout, plan, images, run state and qualification scope remain
unchanged. This branch is a future installed-source scope, not a replacement for
that active campaign.

## Ownership

| Follow-up | Installed owner | Integration |
| --- | --- | --- |
| R0 output claim | `generalization_pac_rehearsal.claim_output` | Already installed on #63; preserved |
| Measured free-space stop | `generalization_commands._execute_pac_plan`, `generalization.resources.pac_free_storage` | Already installed on #63; preserved |
| Qualification/test causes | `generalization_failures`, `generalization_qualification_failures` | Already installed on #63; preserved |
| Retained baseline projection | `generalization_pac_evaluation.project_retained_checkpoint` | Moved from the temporary helper; exclusive owned root, original weight verification, unchanged descriptor and A/A-reference labels, canonical consumer checks |
| Helper exception capture/validation | `generalization_helper_failures` | Moved from the helper; sealed public kind/location and bounded private original; strict inventory, schema, seal, status, vocabulary and source-function checks |
| Helper failure publication/recovery | `generalization_pac_export` | Shared canonical rich/minimal projection; temporary archive supplementation removed |

The temporary runner retains serial composition only. It delegates projection and
capture to installed APIs, and its recovery function calls the existing canonical
exporter directly. Installed collection now retains `failures/helper-<stage>.json`
without a second archive unpack/repack step. Existing stage causes keep their
ordering and failed/interrupted outcomes remain negative. Unsafe or unbound helper
records become explicit omissions. Full exception strings, tensors, protected
truth and original host paths remain private. Discarded old exception text cannot
be reconstructed by this integration.

The four operator follow-up regression cases moved from validation-local tests
into `tests/integration/test_generalization_followups.py`. New normal unit tests
exercise the installed projection and failure APIs directly. They verify stale
output, changed weights, wrong control identity, path escape, interruption,
private-detail truncation and tampered/resealed unsafe diagnostics. Both direct
canonical export paths retain bound diagnostics; read-only recovery preserves
the original workflow bytes.

## Validation and qualification scope

Counterexamples were committed first as `ee2d348`: four failures and two passes
show the absent installed projection API and missing direct rich/minimal export
diagnostics. Implementation is `ba64a33`.

The final affected evaluation/archive/governance selection passes **175 tests and
five subtests**, with **three actual-native qualification checks skipped**. The
earlier contract/privacy selection passes 161 tests and five subtests. These are
local CPU/host construction and report checks, not actual HIP or completed
structural inference evidence. No CI wait or hardware run was performed.

```sh
PYTHONPATH=src python3 -m pytest -q \
  tests/unit/test_generalization_helper_failures.py \
  tests/unit/test_generalization_retained_projection.py \
  tests/integration/test_generalization_followups.py \
  tests/integration/test_generalization_pac_evaluation.py \
  tests/integration/test_generalization_research_closure.py \
  tests/integration/test_generalization_archive.py \
  tests/integration/test_generalization_evidence.py \
  tests/unit/test_generalization_failures.py \
  tests/unit/test_generalization_qualification_failures.py \
  tests/unit/test_pipeline_contracts.py \
  tests/unit/test_foundation_handoff.py \
  tests/unit/test_foundation_task_inventory.py \
  tests/contract/test_foundation_qualified_cli.py \
  specs/012-parameterized-adaptive-curriculum/validation/test_run.py \
  specs/012-parameterized-adaptive-curriculum/validation/test_rehearsal.py \
  -m 'not gpu and not docker'
```

Dependency-free helper, foundation CLI and input-assembly help pass under
`python3 -S`. Prerequisites, shell syntax, whitespace and the read-only 16/16
requirements checklist pass; no implementation hooks are registered. The actual
package implementation serializer refreshes the 007 identity without changing
its 68 dispositions or completion flag.

Installed snapshot for this integration:
`sha256:54de7bd6984b61b809f96865e8393c5299f228f0dc606dd3ab4359d612c8b3b6`.
The active R1 scope on #63 remains
`sha256:b0951861c976252013a86718762666c6f9d0ae6eed017383650edf57f9715e41`.
Historical source/image/device proofs retain their original identities. Future
campaigns using the installed integration must build and qualify this new source;
the old R0 gate receipts do not qualify it. Do not switch the active campaign to
this branch or regenerate its plan as part of this code integration. T083 retains
its historical verified completion; T073 still awaits actual research results.
