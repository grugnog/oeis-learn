# Validation and first-pilot guide

This guide distinguishes implemented evidence commands from the planned research workflow. Public `generalization report` and `verify-bundle`, and opt-in `pilot-followup --arms uniform --qualification`, are implemented. The remaining generalization endpoints and complete serial acceptance workflow are not ready. No HIP readiness or research result is claimed.

## Prerequisites

Use the repository's pinned Dockerfile/locks and existing foundation container workflow. Host requires Python standard library and a usable Docker/Podman connection; no host toolchain/driver changes. Retain private original weights and source/cohort manifests for historical evaluation; public archives cannot reconstruct them. Resolve study/acceptance manifests and quotas using [contracts/experiments.md](contracts/experiments.md), and keep assessment files in scorer-only mounts.

## Before hardware

Build the existing qualification container and run its CPU/native suite, with the implementation registering new test nodes in `container_validation.py`:

```bash
docker build --target qualification -t oeis-learn:qualification .
python3 scripts/foundation_run.py --engine docker --image oeis-learn:qualification validate --suite bigint-cpu --output run-state/evidence/generalization-precheck-001
```

Include isolation, parameter, replay, resume, metric, stale-receipt and partial-package cases. This is CPU/native evidence only. Do not use pipeline `all-ready` as a CPU-only precheck; its dependency registry can select HIP stages. Prebuild the GPU target with `docker build --target gpu -t oeis-learn:gpu .` as well, then freeze code, both inspected images, contracts and helper after integration passes. The hardware session validates compatibility and executes its required qualification scopes once; this local precheck is not an extra HIP run.

## One consolidated real-HIP acceptance invocation

```bash
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/inputs/generalization-acceptance.json \
  --output run-state/evidence/generalization-acceptance-001
```

Expected: one original CPU/native and HIP qualification producer, serial representative preparation/evaluation/refresh/target/replay/resume cases, consumer compatibility checks, verified archive and readable report. Tiny real model updates prove loaded checkpoint and finite forward/backward execution on HIP. Full research experiments do not run here. See [CLI contract](contracts/cli.md) for restart and failure behavior.

On failure, inspect the ledger and bounded log tail in the partial bundle. Fix algorithm defects in implementation phase B, then invalidate affected source/qualification evidence. Restart into a fresh output root; reuse only intact compatible original producers. No repeated stage should lack an explicit reason.

## E0: evaluation extension without training

```bash
python3 scripts/foundation_run.py --engine docker generalization preflight \
  --manifest run-state/inputs/E0.json --output run-state/evidence/E0-preflight
python3 scripts/foundation_run.py --engine docker generalization evaluate \
  --manifest run-state/inputs/E0.json --checkpoints early selected uniform-best uniform-latest \
  --qualification run-state/evidence/generalization-acceptance-001/qualification \
  --output run-state/evidence/E0-evaluate
```

Those checkpoint IDs must resolve to the preflight-frozen private hashes. Missing IDs are unavailable; revise the explicit roster before execution instead of substituting weights. Check full denominators, raw versus fitted budgets, source diversity, shared-bank costs and completion. Historical results remain exploratory.

## P0: two additional uniform seeds

The implementation adds opt-in arm selection and qualification reuse to the historical endpoint:

```bash
python3 scripts/foundation_run.py --engine docker pilot-followup \
  --run run-state/runs/historical --admission run-state/inputs/admission.json \
  --inputs run-state/inputs/historical --arms uniform --tokens 25000000 --seed 20261017 \
  --qualification run-state/evidence/generalization-acceptance-001/qualification \
  --output run-state/evidence/P0-seed17
```

Repeat serially for seed20261018 in a new output root. Use the frozen historical recipe and its selection semantics; do not also run balancing/decay. Include existing seed16 only with compatible identified artifacts/evidence. A missing seed16 becomes an explicit missing reference or separately registered regeneration, never an invented third seed.

## P1: first learning intervention

```bash
python3 scripts/foundation_run.py --engine docker generalization prepare \
  --manifest run-state/inputs/P1-preparation.json \
  --qualification run-state/evidence/generalization-acceptance-001/qualification \
  --output run-state/evidence/P1-prepared
python3 scripts/foundation_run.py --engine docker generalization train \
  --manifest run-state/evidence/P1-prepared/study.json --arms static refreshed --seeds 20261016 \
  --qualification run-state/evidence/generalization-acceptance-001/qualification \
  --output run-state/evidence/P1-train
```

The frozen preparation contract fixes generation, coverage and isolation policy. Preparation emits a fully resolved frozen `study.json` bound to prepared artifacts; it never mutates existing authority. Compare fixed exposures and selection-supervision results; freeze checkpoint/portfolio choices before isolated assessment. Only valid, adequately completed studies advance under their registered rule to the other paired seeds. If screen assessment influenced any advancement, checkpoint/arm/incumbent choice or recipe change, confirmation uses fresh assessment.

## Package and inspect

```bash
python3 scripts/foundation_run.py --engine docker generalization report \
  --manifest run-state/evidence/P1-prepared/study.json \
  --inputs run-state/evidence/P1-prepared run-state/evidence/P1-train \
  --output run-state/evidence/P1-report
python3 scripts/foundation_run.py --engine docker generalization verify-bundle \
  --archive run-state/evidence/P1-report.zip --output run-state/evidence/P1-verified
```

Expect a scoped positive/negative/inconclusive scientific result, actual costs, every registered seed/cell or explicit missing status, and independently reconcilable bounds. A successful software receipt does not establish improvement. Subsequent P2/P3/P4 recipes are defined in the experiment contract and run separately, without changing the single-algorithm-phase implementation requirement.

## Current integration status

The research examples above remain a target workflow. Only `report` and
`verify-bundle` are currently public generalization endpoints. Both require the
global `--image sha256:...` argument naming a prebuilt immutable qualification
image. Report currently accepts **one** explicit evidence directory containing
`evidence.json` and its exact public inventory; multi-stage assembly is pending.
For example, after preparing that inventory:

```bash
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization report \
  --manifest run-state/evidence/study.json --inputs run-state/evidence/public \
  --output run-state/evidence/report-001
python3 scripts/foundation_run.py --engine docker --image "$QUALIFICATION_IMAGE" generalization verify-bundle \
  --archive run-state/evidence/report-001.zip --output run-state/evidence/verified-001
```

Every output must be new. These commands verify hashes and original evidence;
they do not qualify an unexecuted study. Full-state learner resume is an internal
adapter, not yet a public restart route. The helper's `--dry-run` exposes its
reserved command order, but missing endpoints prevent complete execution/reuse.

Internal installed roles now support preparation, real model training and exact
resume, retained evaluation, checkpoint selection, authorized assessment scoring,
and separate visible-only replay proposals/training-only native audits. Retained
checkpoint descriptors identify exact weights, committed exposure and pins.
Best-loss and behavior selection use the same candidate set; assessment consumes
a frozen authorization and active shared exposure debit without ranking. Replay
search persists reservations before model work and retains empty discoveries and
interruption costs. These adapters have CPU/fixture tests. Their existence does
not complete T046/T052/T053/T056–T059/T061–T062 or qualify actual native/HIP work.

As of 2026-10-31, a failing P4 initializer-authority counterexample reopens
T011/T044. The current study/preparation contracts only permit random initialization
and cannot bind the required common retained literal initializer. Resolve that
Phase-B contract/provenance gate before registering the remaining public routes.
See [local results](validation/local-checks.md), [algorithm review](validation/algorithm-review.md),
and [implementation audit](checklists/implementation.md). E0→P0→P1 and later P2/P3/P4
remain separate operator-authorized research; none has been launched here.
