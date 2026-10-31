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


### Feature acceptance before research

`generalization-feature-acceptance/v2` is a closed sealed record with exactly
`version`, `identity`, `status` (PASS), `qualification` (the original qualification
producer identity), and `cases`. `cases` maps exactly the five names below to
ArtifactRefs for original canonical `pipeline-stage/v1` receipts. Legacy v1,
names-only lists, aggregate flags, missing cases and REUSED wrappers cannot qualify
research. The base qualification producer remains immutable and does not need to
claim it already passed these later feature tests.

Each original stage must be PASS, have stage id `acceptance.<case>`, and have binding
exactly `{case, qualification}`. Its sealed files inventory must contain
`case-result.json` and every named artifact below as `<label>.json` relative to the
stage receipt directory. All inventory bytes and hashes are verified. Required
artifacts cannot be substituted with a different label's path.

`case-result.json` is a closed sealed `generalization-acceptance-result/v1` record
with `version`, `identity`, `case`, `qualification`, `status` (PASS), `checks`, and
`artifacts`. `checks` must have exactly the listed keys, each boolean true.
`artifacts` maps exactly the listed labels to the same ArtifactRefs bound by the
original stage inventory.

| Case | Required checks | Required public artifacts |
| --- | --- | --- |
| retained-eval | weights_loaded, visible_only, matched_budgets | checkpoint_identity, evaluation |
| refresh | fixed_support, fresh_shards, finite_update | preparation, training |
| parameter-targets | literal_update, parameterized_update, grounding_verified | literal_training, parameterized_training, target_audit |
| replay-resume | training_only_admission, paired_control, full_state_resume, finite_update | admission, continued_sft, verified_replay, resume |
| failure-packaging | failure_preserved, archive_verified, private_payloads_excluded | failure_ledger, archive_verification |

The failure-packaging case is PASS only when the injected failure remains recorded
as failure and the resulting partial archive verifies; it does not promote the
injected failed workflow to PASS. Public artifacts contain scoped results and
identities, never private tensors or sealed targets. Their nested ArtifactRefs are
verified recursively; private paths, conflicting references, missing payloads and
tampering fail closed. Total proof closure is limited to 512 MiB/10,000 files;
feature, stage and case-result documents are limited to 1 MiB each.

`assemble_feature` validates every original before issuing the aggregate manifest;
`verify_feature` validates it for consumption. `feature_artifacts` returns the exact
verified closure for isolated role projection, including stage-inventory files
whose hashes alone are not discoverable as ordinary ArtifactRefs. The consumer
still independently verifies the current source/image/device/resource facts.
These are integrity and provenance checks over trusted canonical producer outputs,
not cryptographic attestation or proof that arbitrary user-authored PASS records
represent real execution. Producers must write the checks from actual production
path observations; synthetic fixtures establish only validation behavior.

## Retained initializer authority

`generalization-study/v2` and `generalization-preparation/v2` additionally allow
`model.initialization = {"kind":"retained","artifact":ArtifactRef}`. Version 1
continues to permit only `random`. P4 requires the retained form, paired
`continued_sft`/`verified_replay` arms and literal targets. One study-level initializer
binds both arms; a runtime flag cannot override it.

The referenced sealed `generalization-initializer/v1` has exactly `version`,
`identity`, `study`, `checkpoint`, `selection`, `producer`, `ancestry`, and `parent`.
The four named documents are ArtifactRefs. References inside each initializer are
relative to that initializer document's directory. `checkpoint` is a full checkpoint
descriptor; only its exact weights are needed for initialization, not optimizer/RNG
payloads. `selection` contains the complete original checkpoint-selection result;
behavior ordering is independently recomputed over its frozen candidate set.

`producer` is an original PASS `pipeline-stage/v1` receipt with stage
`generalization.initializer`, a nonempty attempt and binding exactly
`{study: origin-study-identity, checkpoint: selected-descriptor-identity,
ancestry: ancestry-list, parent: parent-initializer-identity-or-null}`. Its inventory
binds the original study, descriptor, selection result and weight bytes. REUSED
receipts do not establish an origin. Hashes bind trusted canonical producer evidence;
they are not cryptographic execution attestation.

Ancestry must be mechanical training from random initialization, optionally including
verified training replay. Origin and destination must agree on model architecture,
profile and precision, with literal source/target representation. A random-origin
checkpoint cannot have an imported parent. Otherwise `parent` must equal the origin
study's retained authority reference and its verified weight hash must match the
checkpoint's parent. Every ancestor is reverified, with a 32-generation depth ceiling
and cyclic/repeated authority rejection. `initializers.verify_initializer` returns
exact weights/root plus verified lineage; preparation publication invokes it before
returning a trainable resolved study.

## Reduced acceptance session and archive

A sealed `generalization-acceptance-session/v1` has exactly `version`, `identity`
and `cases`. The four keys `retained-eval`, `refresh`, `parameter-targets` and
`replay-resume` each bind `{manifest: ArtifactRef, runtime: ArtifactRef}` relative
to the session file. All use `purpose: acceptance`, one image/source scope, at
most one seed, 4,096 target tokens and 900 learner seconds. The failure-packaging
case uses the refresh case's environment. Each case executes its own frozen
study; a P1 study is never silently reinterpreted as P3/P4.

`generalization-acceptance-archive/v1` is a sealed software-only inventory with
`session`, `qualification`, `feature`, `cases`, `inventory`, `software_status`,
`scientific_status`, `omissions`, and `workflow`. Qualification is the original
v2 authority reference; cases reference original stage receipts. `feature` is
null unless every case and the serial workflow succeeded. `workflow` is an
optional reference to the sealed pre-packaging serial ledger. Verification checks
all original closures, every available PASS case's obligations, missing cases,
serial failures and archive hashes. Scientific status is always `UNMEASURED`;
this format cannot represent research yields. It uses the same safe outer ZIP
and `generalization-archive/v1` payload wrapper as recorded research evidence.

## Frozen runtime I/O sidecar

`generalization-runtime/v1` is a closed sealed record with `study` (study identity),
`purpose` (`acceptance` or `research`), `recipe` (ArtifactRef), `images` (immutable
qualification then GPU image IDs), `preparation`, `training`, `evaluation`,
`replay`, `assessment`, `feature` and `storage`. Artifact paths are relative to the
study file's directory, even when `--runtime` names a separate file. Storage has
exactly `study_bytes`, `checkpoint_bytes`, `shard_bytes`, `raw_bytes` and
`sharing_bytes`; the hard ceilings apply to every command, not only preflight.

The study model profile equals the recipe version. Its architecture hash is the
seed-free `recipe.model` object augmented with `model_profile` and `inputs` when
those resolved recipe sections exist. Precision is FP32. The custodian measures
actual CPU model tensor sizes and uses the B storage policy to reserve optimizer,
metadata, replay and required pinned generations before learning.

Before preparation, `preparation` is the preparation authority reference. Training
maps each arm to `{cells: [names], data_mode: static|refresh, batch_size: integer}`.
Evaluation is null or `{role: selection|assessment, cells: [names], checkpoints:
[ArtifactRef], seconds, bank_seeds}`; replay is null or `{cell, seconds}`. Cells
must belong to the named role. Preparation resolves these names to exact shards,
prompt/target references and the replay task reference, writes a new study and
runtime, then sets `preparation` to null. It never overwrites input authority.

Resolved training views replace `cells` with `shards: [ArtifactRef]`; evaluation
replaces `cells` with `prompts` and `targets`; replay replaces `cell` with `tasks`.
P1 static support must be a strict subset of refreshed variants with the same
structural support. Refresh rows must belong to the frozen descriptor/counter
stream. P3 retains ordered literal/parameterized concrete pairs and separately
verified grounding proofs. P4 requires identical static views, the common verified
literal initializer, bounded training tasks and current-policy selection evaluation.

Assessment must reference the study's frozen authorization and is debited through
the durable shared exposure ledger before prompt exposure. Selection passes only
chosen identities back to the learner for pin persistence. Canonical training
records its origin; eligible literal selections can publish an original initializer
producer and complete ancestry closure. Missing weights/full state/origin metadata
remain explicitly unavailable rather than reconstructed.

`generalization-command-result/v2` binds the sealed effective invocation and every
retained output byte to the original result. Reuse requires an exact compatible
original PASS, current qualification inspection, intact output inventory and a new
root. Resuming learning still requires a full-state checkpoint. Command receipt
archives are explicitly not feature qualification. Session restart recomputes CPU
preflight/preparation with an explicit repeat reason and reuses verified original
PASS case evidence; it never silently reruns an incomplete learner.

## Reconstructed executable authority (v3/v2)

New executable inputs require `generalization-study/v3` with an `execution` digest
and `generalization-runtime/v2`. The digest covers every runtime field except its
own identity/version, study link and preparation link (the latter binds the study
and would otherwise be circular). It therefore binds the complete recipe bytes,
images, purpose, training/evaluation/replay views, feature reference and storage.
Changing a sidecar and resealing it alone cannot change a frozen experiment.
Older study/runtime versions remain interpretable as historical authority but must
be explicitly reassembled before canonical execution; they are never silently upgraded.

`generalization-preparation/v3` adds `proposed_study`, the exact proposed study
identity. The custodian checks this before materialization, preventing changes to
representation, sharing, inference or progression between assembly and preparation.
Resolving cell names produces new study/runtime identities. All non-path execution
choices and study data policies must remain identical. E0 explicitly has zero target
tokens, no milestones and no training views; it reserves no learner snapshots.

P1 must preserve the family proportions and unique-source support of its static
subset. P2 uses equal admitted-example counts. P3 retains every unsupported row
with its original literal learning target and original pair/disposition audit;
acceptance additionally needs at least one supported parameterized target. P4
acceptance requires finite-update proofs for both arms as well as complete replay
resume comparison. Historical continuous pooled-loss selection is kept distinct.

## Legacy uniform weights-only import

`generalization-checkpoint/v2` is explicitly `kind: weights_only`. It retains the
checkpoint v1 fields, but `state`, `initialization` and `parent` are null and a
`provenance` ArtifactRef is mandatory. Null initialization means original random
weight bytes are unavailable, not that a new random checkpoint was generated.
Pins are exactly `historical_best` or `historical_latest`; `compatibility` binds the
entire original recipe. These descriptors cannot restore learner state or satisfy
`generalization-initializer/v1` provenance.

The sealed `generalization-legacy-import/v1` binds `training`, `prepared`, `recipe`,
`history`, `producer`, `report` and `label`. Import checks the original PASS uniform
producer, corpus/recipe hashes, generic-only literal/no-replay profile, actual private
weight hashes and producer-inventoried validation history. Best exposure is derived
from the original strict-improvement pooled nonpadding dev loss (first minimum),
latest exposure from the final history/training record. The original report contains
the exact sealed training record. Hashes preserve trusted original evidence; they
are not cryptographic execution attestation. The import roster explicitly reports
`resume_available: false` and `initializer_available: false`.

The standard-library assembly tool emits ASSEMBLED/IMPORTED, never qualification
PASS. Native admission, tensor compatibility, coverage, resources and HIP still
require the canonical gates. `source-closure` uses the same `collect_snapshot`
function as qualification. Rebuild images and record closure after final code changes.

## Acceptance selection dependency and legacy initialization

A v3 P4 study may freeze `model.initialization` as exactly
`{kind: acceptance_selection, case: retained-eval, study: <original E0 identity>}`.
This is an unresolved dependency, never learner initialization. Only acceptance
purpose permits it; canonical preparation may materialize its cohorts, but training
must reject it. Session v2 requires the named E0 child, identical model/precision
and recipe, and resolves only after its actual selection publishes original proof.
Resolution changes only initialization and its consequent study/runtime identities.
Every other prepared choice remains byte-identical. Research and generic reuse
cannot consume an unresolved dependency.

`generalization-initializer/v2` adds exactly `legacy` to the original v1 fields.
It requires an audited weights-only uniform checkpoint, its complete original
legacy provenance, a newly measured selection decision, an E0 source study with
matching architecture and recipe, and an original PASS publication receipt binding
and inventorying these payloads. Historical best-loss meaning and null resume state
remain unchanged. This is a new initialization, never reconstructed optimizer state.
Version 1 retains its original full-state-derived meaning.

Acceptance replay may freeze `proposal_policy` as
`mechanical-neutral-expression-diagnostic/v1`: one charged mechanical proposal per
training task removes a declared outer neutral addition. It uses no selection or
assessment truth and still passes canonical native/reference 100-term admission,
strict shortening, replay limits and finite update/resume proofs. Reports identify
its diagnostic origin and do not claim a model discovery. Research rejects this
policy and retains the model-generated eight-attempt default. Mechanical setup
uses fixed generic sampler/refresh draws with finite exclusions; it cannot weaken
quotas or borrow rows when construction or native admission fails.
