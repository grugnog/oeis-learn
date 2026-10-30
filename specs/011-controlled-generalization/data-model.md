# Data model and access boundaries

All entities are versioned. Canonical JSON identity excludes only its own identity field; payload files carry SHA-256 and byte size. Exact integers and indices use existing canonical decimal-string rules. References are immutable identities, not mutable filenames alone. Unknown authority fields and incompatible versions fail closed.

| Entity | Required fields and relationships | Validation / transitions |
| --- | --- | --- |
| StudyContract | id, version, experiment_stage, hypothesis, arms, changing_factors, paired seeds, model/precision, source closure, cohort and selection-policy refs, tokens/attempts/time/storage, checkpoints, advance/stop rules | DRAFT → FROZEN → RUNNING → COMPLETE/PARTIAL/FAILED; authority immutable after freeze; revision creates new id. |
| CohortManifest | role, family rule, source/variant ids, group ids, quota table, observed coverage, exclusions, exact indices, public/private payload inventories, preparation ledger | PREPARING → FROZEN or SHORTFALL. Only complete frozen manifests train/score. Historical manifests keep their version. |
| StructuralDescriptor | generator version, supported AST/control skeleton, family id, numeric sampling law, allowed variant domain | Fixed throughout within-support refresh. Counter increases cannot silently introduce descriptors. |
| RefreshShard | descriptor-set id, seed/ordinal, admitted row ids, tokens, source/target hashes, coverage and rejection counts, exclusion-policy id | Immutable and independently verified; capacity exhausted → EXHAUSTED, no silent reuse/support expansion. |
| TargetPair | concrete source, literal tokens, explicit parameter source/tokens, binding, sharing policy, protected positions, disposition, reference evidence | Round-trip exact agreement required; UNSUPPORTED kept in original stratum; literal/control bindings cannot be fabricated. |
| ProposalRecord | study/checkpoint/bank/prompt refs, ordinal, intervention, source hash/body when permitted, parse/structure attrs, parameter count, phase timings/status, final assignment/source | CREATED → GENERATED/GENERATION_FAILED → VERIFIED/REJECTED/UNRESOLVED; every attempted draw charged, including duplicate/invalid. |
| ProposalBank | model hash, absent-memory intervention, seeds, ordered proposal refs, generation cost, reuse count | Frozen without recipient target refs; target-specific fitting references bank and visible prompt separately. |
| CheckpointRecord | weights hash, full-state hash, initialization/parent ids, tokens/updates, milestone pins, role/selection score, compatibility | Committed only at complete update boundary; weights-only is separate type. Rejected partial write cannot become latest. |
| LearnerState | weights, optimizer, scheduler, Python/NumPy/Torch/device RNG, data/shard cursor and pending batch or pre-sampling RNG, replay snapshot, checkpoint pins/history, consumed ledger, immutable contract | RUNNING → CHECKPOINTED → INTERRUPTED → RESTORED, or TERMINAL. Restore requires exact compatible state and never replenishes consumed budget. |
| ReplayEntry | training task id, track, parent/origin checkpoint, original/final source, assignment, tokens, exact 100-term audit evidence, production/reference identities, admission order | CANDIDATE → ADMITTED/REJECTED → RETAINED/EVICTED. Deterministic oldest admission then identity eviction; per-task cap applied before global cap. |
| SelectionDecision | selection-cohort id, eligible checkpoint ids, completion, structural/parameter macro lower bounds, tie-breaks, selected id or inconclusive reason | Frozen before assessment; no assessment score fields permitted. |
| AssessmentAuthorization | frozen study, candidate checkpoint/portfolio selections, cohort id, permitted measurement cells, exposure ledger | SEALED → AUTHORIZED → SCORED. Adaptive follow-up cannot reuse consumed cohort as fresh confirmation. |
| QualificationContract/Receipt | suites/coverage ids, source closure, immutable image ids, runtime/device facts, precision/resources/security, original producer and artifact hashes, real-update evidence | Original PASS required for reuse; current consumer inspection and coverage compatibility required. Never infer PASS from REUSED alone. |
| StageReceipt/Bundle | experiment and qualification refs, command/start/end/status, prerequisites, bounded log ref, payload inventory, scientific completion and software status, omissions | Append-only transitions; COMPLETE/PARTIAL/FAILED packaging valid, but only full required stage PASS/verified REUSED permits overall software PASS. |

## Role permissions

| Role | Allowed data | Forbidden data |
| --- | --- | --- |
| Preparation custodian | Generated source/100-term oracle for building splits, coverage and exclusions | Exporting protected labels/programs/terms into learner-visible prompts |
| Learner | Mechanical training targets; admitted training-only replay | Selection targets, assessment continuations/programs/protected labels, OEIS training solutions |
| Candidate generator/fitter | Registered visible terms/indices, own candidates, frozen model/policy | Selection or assessment continuations and target programs |
| Selection scorer/controller | Selection-supervision truth and frozen candidate results | Assessment truth/results as inputs to choosing checkpoints, candidates or progression |
| Assessment scorer | Authorized frozen candidates and sealed assessment truth | Feeding results into the current study's training/selection |
| Public bundle/report | Scoped results, permitted generated candidates, hashes, coverage aggregates after frozen scoring | Optimizer/private state, sealed truth or target programs, credentials |

Implement these boundaries with separate files and mounts, not a single payload whose keys callers promise to ignore. Family labels needed for scoring stay in scorer manifests; candidate signatures are computed from candidates. Training parameter-holdout families may overlap by design, but variant/duplicate/prefix/shift groups cannot cross roles. Structural holdout families must be excluded from training and distinct between selection and assessment.

## Selection and measurement invariants

For fixed N, successes S and unresolved U: lower=S/N, upper=(S+U)/N. Resolved failures=N-S-U; completion=(N-U)/N. Macro results average per-family bounds over the frozen nonempty family roster; never average only families with completed measurements. Paired differences must preserve paired rows/seeds and report unresolved possibilities. Confidence intervals, if provided, name their resampling unit and do not substitute row counts for independent seeds.

Default checkpoint ordering: completion eligible on each primary selection split, greatest structural macro confirmed yield, then parameter macro confirmed yield, then earlier token exposure, then stable checkpoint identity if still tied. Ineligible sets yield no promotion. Oracle union is scorer-only coverage, not a candidate-selection policy.

## Checkpoint and replay invariants

Charge actual target tokens after replay substitution, excluding BOS/PAD/input terms. At a token cap, do not consume a pending batch irreversibly; save it or preserve pre-sampling RNG/cursor. Pin fixed milestones, best/latest/selected; account their private storage before starting. A newly selected short target is not proof of a unique program. A replay entry's full audit must be independently reproducible against its training oracle; a user-supplied admitted=true field alone is insufficient.
