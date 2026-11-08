# Data model — progressive mathematical discovery

All following types are proposed closed versioned records. Implementation must install JSON schemas with additionalProperties=false and semantic validators; markdown tables are the design, not executable schema validation.

## Common envelope

Every authority object has `version`, `identity`, `created_at`, `parents`, `purpose` (acceptance/research), `track` (synthetic_strict/oeis_derived/external_assisted) and content-bound payload. Identity is the existing canonical SHA-256 seal excluding its own identity field. Artifact references use confined relative path, byte count and SHA-256. Reject symlinks, duplicate/conflicting paths, unknown keys, nonfinite numbers and booleans substituted for integers. Times use integer microseconds; authoritative values use canonical signed decimal integer strings. Timestamps identify history, not cross-boot elapsed duration.

| Entity / version | Principal payload and relationships | Validation |
| --- | --- | --- |
| discovery-blueprint/v1 | Requested stages, candidate settings, source locations, proposed caps | May be unresolved; explicitly non-executable |
| discovery-campaign/v1 | Resolved study policies, source/image/qualification refs, task roster/DAG, cumulative caps, selected stage subset, transition rules | Immutable; every node/edge bounded; all dependencies and resources resolved before freeze |
| discovery-task/v1 | Campaign, stage, arm/seed, initializer/compatible parent, token target, active cap, resource/reserve envelope, roster/library/stream policy | <=86,400s for training; no new entitlement from duplicate identity/output |
| discovery-runtime/v1 | Source closure, image pair, device/precision/model-shape capabilities, execution policy, role projections, qualification refs | Exact compatible original producers; no fixtures as research authority |
| discovery-cohort/v1 | Role, stratum, source/offset, exact visible and private truth refs, related groups, exposure history, task profile | Role-specific projections; sufficient support; exclusive group policy and no silent horizon change |
| discovery-exposure/v1 | Group identities, historical role/outcome access, source relations, reasons/limitations | Append-only; cannot erase exposure by changing seed or cohort name |
| discovery-coverage/v1 | Task/operator class, private constructive witness ref, language/resource profile, verification and supported/over-budget/unresolved classification | Search failure never equals inexpressible; witness not a training target |
| discovery-stream/v1 | Generator/numeric law, bins, seed/counter range, draw/rejection ledger, shards, membership and fingerprints | Every ordinal accounted; independent verification; no arbitrary post-outcome redraw |
| discovery-trial/v1 | Search-stage/ordinal, exact parameters, seed, qualification, resource coordinate, proposal/result refs, state | Parameters sealed before execution; one result per proposal; confirmation never sampler feedback |
| discovery-state/v1 | Weights/optimizer/scheduler/RNG refs, learning compatibility, stream/pool/library cuts, pending update and cost journal positions | Atomic complete-update state; full-state vs weights-only explicit; exact compatible lineage |
| discovery-admission/v1 | Candidate, source/member, visible/admission commitments, native/reference evidence, concrete/paired target refs, length/cost, validity | Training membership only; no named formula as candidate authority; no prefix-only pass |
| discovery-archive/v1 | Ordered admitted targets, per-task alternatives, weights/evictions, ancestry, validity cut | Capacity and diversity rules; every consumed entry tied to checkpoint lineage |
| discovery-library/v1 | Components, typed holes, expanded AST refs, dependency DAG, support groups, bounds, verification | Training-only origins, acyclic dependencies, hygienic expansion, positive budgets |
| discovery-source/v1 | Immutable source, access terms, indices/value semantics, grouping, transformation, profile, verification, suitability | Unsupported data rejected; bounded finite-support semantics explicit |
| discovery-decision/v1 | Compared refs/seeds/panels, counts/deltas/censoring, interval policy and looks, thresholds, status/reasons | Recomputed from outcomes; outcome-driven threshold change invalid |
| discovery-dossier/v1 | Sealed candidate, profile/domain, tested ranges, known-method comparisons, proof and novelty refs, claim labels | Evidence scopes independent; unsupported claims cannot be promoted |
| discovery-quarantine/v1 | Refutation, root artifacts, dependency cut, affected targets/library/checkpoints/claims | Append-only; transitive closure complete; cannot retroactively rewrite original bytes |
| discovery-report/v1 | Inventory, stage/task/trial/decision tables, qualification closure, metrics, first failure, omissions | Offline reconciliation; required missing evidence prevents VERIFIED |

## State machines

Campaign: DRAFT → FROZEN → QUALIFIED → ACCEPTED → RUNNING → PAUSED / COMPLETE / STOPPED / INCOMPLETE / FAILED. QUALIFIED and ACCEPTED require exact runtime/feature receipts. A completed software rehearsal does not imply a positive scientific decision. Stages omitted from an explicitly selected campaign subset are NOT_SELECTED, never COMPLETE.

Task: RESERVED → RUNNING → PAUSED → RUNNING → COMPLETE / PARTIAL / FAILED / CANCELLED. Resume is the same task with a new invocation identity and remaining reservations. At a terminal partial boundary a different preallocated continuation task may start; it inherits the immutable learning compatibility and remaining campaign exposure. A stage needing an additional unallocated task becomes INCOMPLETE.

Trial: PROPOSED → RESERVED → RUNNING → COMPLETE / PRUNED / FAILED / PARTIAL / INFEASIBLE. INTERRUPTED invocation is not a fresh trial. Same ordinal always returns the same proposal; a duplicate tell must agree byte-for-byte or be rejected. Partial same-task recovery is allowed only within remaining trial limits.

Admission transaction: RESERVED → EXECUTED → VERIFIED → COMMITTED or REJECTED. Only committed target/archive cuts can train. Rejections and unfinished attempts remain charged. Library publication uses the same transaction pattern.

Claim axes are independent: finite_agreement/conjecture/bounded_verified/proved; novelty not_reviewed/known/uncertain/potentially_novel/independently_confirmed. A proof of a polynomial identity does not establish novelty or all-input machine safety. Quarantine overrides eligibility without deleting the original labels/history.

## Recovery cut and journal

One checkpoint cut binds every dynamic object consumed by an optimizer update: weights, optimizer/scheduler, framework/device RNG, allocation/reward state, shard cursor and generator watermark, target archive and component library versions, charged/useful/discarded exposure, cost journal position and selection history. Every update records consumed entry/group identities in a bounded shard-backed ledger sufficient for transitive quarantine.

Reserve before work, append supervisor-backed elapsed/debit observations, atomically commit the new checkpoint and latest pointer. On crash restore the previous cut, retain all later spent charges, and classify repeated work as discarded/replayed. Never restore a newer library or target archive onto older model state implicitly. A frozen update policy permits dynamic versions; changing policy itself creates a new compatible-child decision or a rejected resume.

## Relationships and private projections

Campaign → tasks → invocation receipts → checkpoints/decisions. Cohort groups reference exposure history across all campaigns. Stream/admission records → target archive versions → consumed-update ledger → checkpoints. Training targets → extracted components → libraries → generated candidates → dossiers. Refutation walks this graph in reverse to enumerate affected descendants.

Custodian holds source definitions and truth. Generator sees only visible prompts; admission sees only training continuation truth; scorer sees the selected development/confirmation/assessment truth. Learner receives approved targets and training-only probes. Proposal role receives permissible scalar observations only. Public export includes task outcome/group hashes for independent metric recalculation without private target values. A content hash proves byte binding, not permission to mount those bytes.
