# Validation and first-pilot guide

All nine generalization commands are registered. The implementation has local
CPU/reference and fixture integration evidence. Actual pinned-container/native
qualification and HIP acceptance have **not** run in this environment. No research
campaign is started by these instructions or by the acceptance helper.

## Assemble real inputs first

Reconstruction starts from verified Git commit `6e63421`; the earlier conversational
`dae1469` reference was unsupported and must not be used. The commands below are now
implemented. They perform no HIP work. Place the two supplied ZIPs in the checkout
root, retain the original private weight files, and import the verified uniform arm:

```bash
python3 scripts/generalization_inputs.py import-uniform \
  --bundle study-20261016.zip --metadata uniform-import-metadata.zip \
  --weights tmp-validation/010-followup-004/study-20261016/stages/followup.uniform/train \
  --output run-state/generalization/uniform-import
```

This checks the actual files before importing. It produces `best-checkpoint.json`
and `latest-checkpoint.json` with original provenance. Best is at 17,885,681 tokens /
18,433 updates; latest is at 24,999,970 tokens / 25,764 updates for the uploaded run.
Neither is a full-state resume or a P4 selection-designated initializer.

After final source changes and image builds, record the exact source snapshot:

```bash
python3 scripts/generalization_inputs.py source-closure \
  --output run-state/inputs/source-closure.json
```

Copy and fill the four `inputs-*.example.json` blueprints under
`configs/foundation/pipeline_studies/`. Supply actual recipe, descriptors, mechanical
candidates/exclusions, inspected image IDs and (for P4) verified initializer proof.
These are explicit unresolved examples; they do not create weights or claim native
admission. Assemble each with, for example:

```bash
python3 scripts/generalization_inputs.py study \
  --blueprint run-state/inputs/refresh-blueprint.json \
  --output run-state/generalization/acceptance-inputs/refresh
```

Create a cases JSON mapping the four case names to the four emitted `study.json`
paths, then run:

```bash
python3 scripts/generalization_inputs.py session \
  --cases run-state/inputs/acceptance-cases.json \
  --output run-state/generalization/acceptance-inputs/generalization-acceptance.json
```

The session file must be above all four child studies. Its dry run remains the
serial helper below, with this generated manifest path. Missing native inputs or
initializer proof is a blocker, not permission to fabricate receipts. The uploaded
uniform metadata alone is insufficient to assemble all four cases. In particular,
P4 needs either existing original selection/initializer evidence or a separately
implemented bootstrap inside the same serial workflow before a one-invocation
acceptance can be promised. Such a bootstrap is not implemented by input assembly.

## Freeze inputs before execution

Use Python's standard library on the host and the pinned Dockerfile/locks for
installed roles. Prebuild `qualification` and `gpu` targets on the operator's engine,
inspect their immutable `sha256:...` IDs, then freeze code and contracts. Do not
change host drivers. `validate --suite bigint-cpu` is CPU/native only;
`pipeline all-ready` may select HIP and is not a CPU-only precheck.

Each executable study has a sealed v3 `study.json` and v2 `runtime.json` beside it. The runtime binds
the resolved recipe, CPU/GPU image pair, role views, storage reservations and original
feature acceptance. References are relative to the study directory; use a dedicated
study input/output tree. Explicit `--runtime FILE` selects a sealed sidecar without
overwriting an earlier one. See [artifact formats](contracts/artifacts.md).
The templates in `configs/foundation/pipeline_studies/` are unresolved recipe
commitments, not ready-to-run study manifests.

Preparation views name frozen cells. `prepare` publishes a new study/runtime with
exact role files; its `result.json` reports `result.manifest` and `result.runtime`.
Use those emitted workspace-relative paths for subsequent commands. Never assume
the resolved study is directly under the command output root.

E0 evaluation must retain actual private weights and exact descriptor identities.
P1 freezes paired static/refresh views with the same family support and fresh variants.
P3 requires independently checked target-pair grounding artifacts. P4 requires a
version 3 study with a verified common retained literal initializer, original
selection/producer/ancestry proofs, paired continued-SFT/replay arms and selection
evaluation. Weights-only artifacts cannot serve as full-state resume checkpoints.

## Four frozen inputs, five acceptance cases

A sealed `generalization-acceptance-session/v1` manifest binds four case entries:
`retained-eval`, `refresh`, `parameter-targets` and `replay-resume`. Each entry has
exact `manifest` and `runtime` ArtifactRefs relative to the session document.
All children use acceptance purpose, the same source closure and exact image pair.
The failure-packaging case uses the refresh input. The session cannot override
child sidecars with a runtime CLI flag.

Use tiny actual model/data fixtures. Retained evaluation loads an identified
checkpoint. Refresh must consume variants outside the static subset. Parameter
targets must exercise both forms with verified grounding. Replay/resume must
produce at least one independently admitted training alternative and compare
complete resumed state with uninterrupted state. Empty replay yield is valid in
research but cannot establish acceptance coverage of an admitted replay update.

Acceptance inputs have at most one seed and 4,096 target tokens. The loader also
bounds each case's work to fit inspection, validation and work inside its 900-second
reservation. Preflight validates all four inputs and storage before expensive work.

## One consolidated HIP session

First inspect the plan without launching an engine:

```bash
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/generalization/acceptance-inputs/generalization-acceptance.json \
  --output run-state/evidence/generalization-acceptance-001 --dry-run
```

After the operator completes real image/input/preflight gates, the single normal
invocation is the same command without `--dry-run`. It runs preflight, one original
qualification, preparation, five serial cases, report and independent verify-bundle.
The helper forwards the frozen CPU image. Reservations total 13,800 seconds plus
600 seconds of recovery margin within four hours; they do not authorize research.

The helper preserves original receipts, bounded logs and failure status. A verified
complete or partial report is copied to the session sibling ZIP. Report/verifier
failure produces a minimal recovery archive that does not claim qualification.
A successful complete session publishes `report/qualified/`, containing the original
qualification and feature proof closure for subsequent research consumption.
A partial session cannot qualify research. Generic individual-command ZIPs carry
receipts only; they are not feature qualification archives.

Restart into a fresh output root with `--reuse-run OLD_SESSION_ROOT`. Original
case producers and exact hashes must still verify against unchanged inputs.
Preflight and preparation are revalidated with an explicit repeat reason.
Changed source/images/authority invalidate affected evidence. Do not use unchecked
skip/start-step flags. Learner `train --resume FULL_CHECKPOINT` is a separate route:
it verifies the full state, optimizer/RNG/data/replay compatibility and retained pins.
A weights-only checkpoint is rejected. Selection persistence emits a fresh private
store whose verified full-state pointer can be used for later resume.

## E0 → P0 → P1 operator sequence

Research is separately authorized after acceptance. Freeze research-purpose runtime
inputs with the accepted feature receipt and current source/image identities.
The following paths and descriptor IDs are operator-provided inputs, not artifacts
created by implementation. Set `QUALIFICATION_IMAGE` to the inspected immutable
CPU image and `QUALIFIED_ROOT` to the complete session's `report/qualified/`.

E0 extends evaluation without retraining. Select exact descriptor identities from
the frozen runtime checkpoint roster; display names are not IDs:

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization preflight \
  --manifest run-state/inputs/E0/study.json --output run-state/evidence/E0-preflight
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization evaluate \
  --manifest run-state/inputs/E0/study.json --checkpoints "$EARLY_ID" "$SELECTED_ID" "$UNIFORM_BEST_ID" "$UNIFORM_LATEST_ID" \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/E0-evaluate
```

Retain unavailable checkpoint status rather than substituting weights. Compare
fixed denominators, raw/fitted budgets, shared-bank costs and completion. Selection
uses the declared candidate set. Assessment consumes a frozen authorization and
durable exposure debit; it does not rank new checkpoints.

P0 adds only uniform seeds through the opt-in historical route, preserving its
recipe and selection semantics:

```bash
python3 scripts/foundation_run.py --engine docker pilot-followup \
  --run run-state/runs/historical --admission run-state/inputs/admission.json \
  --inputs run-state/inputs/historical --arms uniform --tokens 25000000 --seed 20261017 \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/P0-seed17
```

Repeat serially for seed20261018 in a fresh root. Include seed16 only with compatible
identified evidence; missing evidence remains missing or requires separately
registered regeneration. Do not silently add historical balancing/decay arms.

P1 is the first paired learning intervention:

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization prepare \
  --manifest run-state/inputs/P1/study.json --qualification "$QUALIFIED_ROOT" \
  --output run-state/evidence/P1-prepared
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization train \
  --manifest "$PREPARED_STUDY" --runtime "$PREPARED_RUNTIME" --arms static refreshed --seeds 20261016 \
  --qualification "$QUALIFIED_ROOT" --output run-state/evidence/P1-train
```

Set `PREPARED_STUDY` and `PREPARED_RUNTIME` from the prepare result. Compare fixed
exposures and selection-supervision results; freeze checkpoint/portfolio choices
before isolated assessment. Advance only under the registered completion rule.
If screen assessment influenced advancement or any recipe/incumbent choice,
confirmation uses fresh assessment. P2/P3/P4 have separately frozen studies;
they are not automatic follow-on jobs or acceptance-helper actions.

## Report and independently verify

Research `report` consumes **one explicit aggregate public evidence directory**
containing `evidence.json` and its exact inventory. Assemble observations and missing
statuses according to the evidence contract. Do not pass multiple raw stage roots
or infer scientific measurements from private checkpoint files. Acceptance-session
reporting assembles its original case proof closures automatically.

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization report \
  --manifest "$PREPARED_STUDY" --inputs run-state/evidence/P1-public \
  --output run-state/evidence/P1-report
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization verify-bundle \
  --archive run-state/evidence/P1-report.zip --output run-state/evidence/P1-verified
```

Every output and archive path must be new. Preserve actual costs, every registered
seed/cell or explicit missing status, and independently reconcilable bounds.
A successful software receipt does not establish improvement. Private learner
states, sealed truth and checkpoint bytes remain outside sharing archives.

See [local evidence](validation/local-checks.md), [algorithm review](validation/algorithm-review.md),
[hardware status](validation/hip-acceptance.md), [coverage audit](checklists/implementation.md)
and [CLI contract](contracts/cli.md) for implementation and validation boundaries.
