# Versioned authority and evidence contracts

Implement strict validators and contract tests for these proposed formats before wiring endpoints. JSON objects reject unknown authority keys; extensions require a new version. All external paths must be confined to declared roots and regular files; no traversal/symlink escape. Reuse existing canonical serialization/sealing and safe archive handling.

## Study contract: `generalization-study/v1`

Required fields: `version`, `identity`, `recipe`, `track`, `hypothesis`, `changing_factors`, `arms`, `seeds`, `source_closure`, `model`, `precision`, `data`, `selection`, `assessment`, `inference`, `budgets`, `checkpoints`, `progression`.

- `track` is strict generic-only for this feature. No arbitrary assisted inheritance.
- `data` binds generator law, descriptors, manifests, target format/sharing policy, exclusions and role-specific inventories.
- `selection` names authorized supervision, eligibility, score/tie rules and fixed checkpoint candidates.
- `assessment` names sealed cohort, authorized frozen comparison cells and exposure ledger; it contains no raw truth in the learner view.
- `inference` binds 20 visible/100 total terms, candidate budgets, drain policy, controls, fit limits and visible-only choice rule.
- `budgets` binds target tokens, updates where capped, aggregate and per-operation seconds, memory, draws, replay, storage and logs.
- `progression` records screen/confirmation/stop decisions, permitted selection-only feedback and fresh-assessment requirements. All seed/arm subsets selected by CLI must be subsets of the frozen roster.

## Qualification: `generalization-qualification/v2`

Separate authority from study contract. Required fields: `version`, `identity`, `suite_selection`, `covered_capabilities`, `source_closure`, `build_dependencies`, `images`, `runtime`, `hardware`, `precision`, `resources`, `engine_security`, `producer`, `artifact_inventory`, `status`.

`resources` binds the effective memory ceiling, CPU count, Torch intra/inter-op threads, native threads and learner count. It does not reuse study token/time/data budgets. `engine_security` binds engine version, OCI runtime, rootless/SELinux mode and the enforced network, root filesystem, privilege and capability restrictions. v1 producers omit these facts and are rejected by this prospective profile.

`images` uses inspected immutable image IDs, not tags. `hardware/runtime` includes actual HIP availability, device identity/properties and observed driver/runtime/toolchain facts. `producer` references original PASS stage receipts and their dependencies. Artifacts include real weight-load and finite forward/backward parameter-change evidence. A CPU-only substitute fails.

Base qualification covers the shared environment/runtime/model path. New feature smoke cases add their own acceptance receipts; full research execution also requires the completed feature-acceptance manifest. A smoke consumer does not need a receipt claiming it already passed its own test.

Reuse requires: intact original PASS payloads; current whole-package source/build closure equality; freshly inspected image/runtime/device match; matching precision/security/resource contract; producer coverage explicitly includes all required consumer acceptance capabilities. Study datasets/budgets are not equality keys for shared qualification, but each study receipt retains its complete authority separately. Unknown changed files invalidate reuse. Initial implementation supports exact matching, not inferred resource supersets or arbitrary producer chains.

## Preparation authority: `generalization-preparation/v1`

Required fields: version, identity, generator/law and descriptors, role roster, exact quotas, exclusion policy, seeds, draw/time/storage ceilings and proposed study settings. Freeze this before materialization. Success emits cohort/shard manifests and a new fully resolved `study.json`; shortfall emits no trainable study. Existing frozen studies are never patched with new artifact identities.

## Dataset and state formats

- `generalization-cohort/v1`: role manifest and payload inventory with frozen quotas, coverage observations, family/group policy and exclusions. Public prompt records contain only opaque id, visible exact indices/terms and authorized representation fields; no target/family/stratum data.
- `generalization-refresh/v1`: fixed descriptor-set identity, ordered immutable shard hashes, numeric law, generation seed/draw/rejection ledger, cursor and exhaustion semantics.
- `generalization-checkpoint/v1`: atomic complete-state inventory and compatibility identity, milestone/best/latest/selected pins, token/update counts and pending-batch state. Full restore includes all learning/RNG/data/replay/selection/budget state, or rejects.
- `generalization-replay/v1`: training-only target roster, independent exact verification, original/final source and grounding, origin checkpoint, admission order, task cap/global cap, sampler cursor and eviction history. Old `synthesis-replay/v1` is not silently reinterpreted.

## Bundle: `generalization-evidence/v1`

Required logical members: study/effective invocation contracts; qualification producer refs plus verified original evidence needed offline; stage ledger; data/checkpoint/source identities; coverage and rejection counts; per-attempt results; family/seed/aggregate summaries; selection/exposure decisions; measured costs and bounds; concise scoped interpretation; inventory and omissions.

Public source bodies are generated candidates only where permitted. Hidden target programs/terms and optimizer/checkpoint state are excluded; public weight identities are hashes. Archive verification rederives totals and status from records, verifies every required size/hash, and rejects a stale summary or missing mandatory evidence. Full scientific completion and software PASS are separate fields.

Terminal failure packaging preserves completed measurements without merging incompatible manifests. A partial bundle may pass integrity validation while carrying FAILED/PARTIAL software status. A fresh-root restart references old verified producers and new execution receipts; it never mutates old terminal evidence.

### Measurement reconciliation payloads

`effective_invocation` uses `generalization-measurement-plan/v1`: exactly `version`,
`study`, `required_stages`, and `roster`. Each opaque roster key denotes one frozen
measurement cell/task; its record contains exactly `family`, `seed`, `cell`, and
`attempts`. All registered seeds must appear, including seeds with no completed
measurements. Attempt caps cannot exceed the study's inference cap. The stage roster
must equal the invocation's ledger roster.

`attempts` uses `generalization-attempts/v1`: exactly `version` and an `attempts`
array. Each attempt contains `row`, zero-based `ordinal`, `outcome` (true, false or
null), and `queue_us`, `service_us`, `cleanup_us` (nonnegative integer microseconds or
null for unavailable). Unknown fields, duplicate ordinals and unregistered rows fail.
An exact success resolves a row true; only all planned attempts resolving false
resolve it false. Otherwise it remains unresolved. Invalid/duplicate *draws* still
occupy their own ordinal; duplicate ordinal records are corruption.

The verifier recomputes aggregate summaries, cell/seed/family summaries, bounds,
coverage and cost totals. Family summaries are nested by cell, decimal seed key,
then family; seed summaries omit the last level. All use the aggregate summary's
exact rational-string representation. Costs contain `measured_us` and `unavailable`
counts per phase; missing timing is never zero-filled. Coverage contains expected and
recorded attempts, missing attempts, expected rows and resolved rows. Completion
bounds are not statistical confidence intervals. Old placeholder payloads fail this
semantic check even if the outer v1 shape and every file hash are valid.

Stage entries must resolve to original verified producer/stage evidence with matching
status. Qualification payload dependencies must be available transitively in the
public inventory. A partial bundle may have verified integrity with PARTIAL software
status; missing measurements remain in the frozen denominator.


### Public archive layout

The report adapter preserves original hash-bound evidence under `payloads/` without
rewriting producer references. Root `evidence.json` is a sealed
`generalization-archive/v1` descriptor with exactly `version`, `identity`,
`payload_root` (`payloads`) and `evidence` (an ArtifactRef for
`payloads/evidence.json`). The payload evidence manifest retains the
`generalization-evidence/v1` schema. Root `summary.json` is the pipeline status;
`payloads/summary.json`, when present, is the independently reconciled scientific
summary. Neither is substituted for the other. The archive verifier validates the
outer inventory, then the original evidence closure and all derived measurements.
Private or unsupported mandatory inventory paths fail packaging rather than being
silently omitted. A generic failure-recovery archive is not qualifying evidence.
