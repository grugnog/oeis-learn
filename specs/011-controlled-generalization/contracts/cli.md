# Canonical CLI contract

## Status and compatibility

All commands below using `generalization` are registered by this feature. Installed-image/native/HIP qualification remains an execution gate, not a claim established by registration. Existing `validate --suite pipeline`, `pilot-diagnose` and `pilot-followup` commands remain compatible; no historical default silently acquires new semantics.

Portable entrypoint remains `python3 scripts/foundation_run.py --engine docker|podman ...`. Host routing uses standard-library code and launches existing isolated roles; no direct host PyTorch/native dependencies or runtime socket in containers.

## Implemented endpoints

Common flags: `--manifest FILE --output NEW_DIRECTORY`; immutable manifest identity is authority. Runtime arguments may choose declared cells or lower resource limits only through a newly sealed effective invocation. Unknown flags/arms/checkpoints fail before costly work. Source run inputs and private artifacts must lie within explicitly mounted workspace roots.

| Command | Inputs / options | Required behavior |
| --- | --- | --- |
| `generalization preflight` | Manifest, artifact references, `--output` | Read-only inspection plus new preflight report: resolved stage list, budgets, compatibility, missing artifacts, exact CLI plan; no learner work or assessment exposure. |
| `generalization qualify` | Acceptance manifest, `--output` | Run canonical runner's CPU/native and HIP qualification once; issue original PASS producer with immutable payloads and actual device evidence. |
| `generalization prepare` | Study manifest, `--qualification PRODUCER`, `--output` | Bounded corpus/target/shard preparation, coverage/isolation certificate or shortfall. Never publish assessment targets to learner/generator. |
| `generalization evaluate` | Manifest, `--checkpoints ID...`, optional declared `--proposal-bank ID`, qualification, output | Retained checkpoint loading, visible generation/fitting, frozen selection, isolated scoring. Private artifact absence is explicit. |
| `generalization train` | Manifest, `--arms NAME... --seeds INTEGER...`, qualification, output; optional `--resume FULL_CHECKPOINT` | Run only selected registered arms/seeds; reject weights-only resume, changed data/order/budgets or undeclared combinations. |
| `generalization replay` | Manifest, declared initializer/training-task roster, qualification, output | One bounded search/admission/distillation round; preserve continued-SFT control and separate search costs. |
| `generalization report` | Manifest, `--inputs DIRECTORY --output` | Reconcile source/contract/seed/denominator identities and package complete or partial evidence. Never infer missing observations. |
| `generalization verify-bundle` | `--archive FILE --output NEW_DIRECTORY` | Existing safe archive checks plus new contract/status invariants; read-only with respect to source archive. |
| `generalization acceptance` | Acceptance session manifest, qualification, `--case NAME --inputs PREPARED_ROOT`, output | Small canonical fixture case: retained-eval, refresh, parameter-targets, replay-resume or failure-packaging. Uses production paths and synthetic fixture truth, never research assessment. |

Existing `pilot-followup` gains opt-in `--arms uniform` and `--qualification PRODUCER` without changing its default three-arm behavior. P0 uses the historical recipe/selection path. Existing `pilot-diagnose evaluate` may gain qualification and generalized checkpoint descriptors through an explicit new profile; old ordinal-based behavior remains available.

## Serial helper

Proposed invocation:

```bash
python3 scripts/generalization_hip_serial.py --engine docker \
  --manifest run-state/inputs/generalization-acceptance.json \
  --output run-state/evidence/generalization-acceptance-001
```

The helper plans then invokes `preflight`, `qualify`, `prepare`, required `acceptance` cases, `report`, `verify-bundle` in order. Every consumer receives the **original** qualification producer root; it independently verifies coverage/compatibility and performs lightweight current image/device inspection. Helper logic is argument forwarding, serial process supervision, ledger and bounded output capture; acceptance/algorithm decisions live in canonical endpoints.

Add `--dry-run` for the command list/budgets without expensive execution. A workflow restart uses a new output root and `--reuse-run OLD_ROOT`; it validates each reusable completed stage against unchanged inputs and records actual producer refs. There is no unchecked `--start-step` or `--skip-tests`. Learner resume is separate and must satisfy its full-state contract.

If the canonical report endpoint itself fails, use existing minimal recovery packaging to preserve terminal ledger/log evidence, mark PACKAGE_FAILED, and return failure. Never print or package credentials/private checkpoint bytes.

## Status and exit contract

Preserve canonical runner status mapping: PASS exits0, FAIL exits1, BLOCKED/PARTIAL/INCONCLUSIVE exits3, INTERRUPTED exits130. Invalid argument handling retains the host parser's established exit2 convention. Store precise reason codes (MISSING_ARTIFACT, QUOTA_SHORTFALL, INCOMPATIBLE_QUALIFICATION, BUDGET_EXHAUSTED, INTEGRITY_FAILURE, PACKAGE_FAILED) alongside status rather than inventing competing numeric meanings. A valid negative scientific outcome may have software PASS; exit0 alone is not a generalization claim. Packaging failure cannot produce PASS.

Output paths are confined checkout/workspace-relative roots. A directory output produces a sibling `<output>.zip`, matching existing packaging. Use fresh directories and archive paths; do not overwrite terminal results.

`--runtime FILE` optionally selects the sealed sidecar for an ordinary study.
Acceptance sessions bind all four child study/runtime references themselves.
Global `--image` must match the frozen CPU image when supplied; report and
verify-bundle require an immutable CPU image, and the helper forwards it.
Research reports consume one explicit aggregate evidence root; acceptance reports
assemble the session's original case proof closures. A complete session exposes
`report/qualified/` for subsequent research qualification consumption.
