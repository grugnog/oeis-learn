# Data model: pipeline validation, scheduling and study evidence

**Status**: Planned schemas. JSON uses existing canonical serialization and SHA-256 identity helpers; exact integers remain canonical decimal strings where the existing contracts require them. Unknown keys fail closed for versioned authority-bearing records. Timing values are finite, nonnegative observations and never substitute for identities.

## Identity rules

Every authority-bearing document has `version`, `identity` and an immutable canonical payload. Compute identity over the payload excluding its own identity and mutable presentation fields. Paths are workspace-confined references plus content hash; a path string alone is never an input identity. Large/private payloads are not embedded in public bundles.

Record both full source snapshot and the explicitly versioned dependency closure used for a stage. Bind image digest, installed package inventory, native binary, guest, numerical/engine/fit/reference policies and workload hashes as relevant. Code paths that ignore an effective policy in contract identity must be corrected before reusing their artifacts. Old v1 identity and seal validators stay available; never rewrite an old record to look like v2.

### ValidationContract — `pipeline-validation/v1`

| Field | Meaning / validation |
| --- | --- |
| `scope`, `phases`, `stage_registry_version` | `engineering`; phase1–5 or `all-ready`; registered implementation versions only |
| `authority` | Explicitly denies learning/resume/final evaluation; permits only declared diagnostic disposable updates and bounded operations fixtures |
| `source`, `input_refs`, `weights_ref` | Producer/current source and closed dependencies, compatible completed run/prepared inputs, frozen selected weights hashes |
| `engines`, `resource_policy` | Primary engine, requested portability coverage, effective CPU/memory/device policy and finite limits |
| `workloads`, `comparisons` | Hashes of frozen manifests and prospective candidate/measurement policies |
| `budgets` | Global, per-engine build, phase, stage, role, cleanup/export, disk/log/archive and job limits; reservations must fit the parent |
| `reuse_refs` | Prior immutable receipt/bundle hashes, never permission to inherit a whole-run result |
| `excluded_authorities` | Training, T096 retry/resume, assisted-data invention and final access are rejected, not simply unused |

### ComparisonContract — `pipeline-comparison/v1`

Fields: parent validation identity; workload membership; baseline/candidate source and policy hashes; primary metric; thresholds; seed/order roster; warmup/calibration/confirmation count; cold/warm cache policy; row/attempt/control denominators; token, service and group deadlines; numerical tolerances; telemetry/trace settings; frozen exclusion manifest; expected artifact/aggregate work; allowed outcome interpretations.

If calibration derives finite reservations, store the derivation-rule hash, complete calibration receipt hashes and resulting child contract before the first candidate measurement. A candidate run cannot mutate its parent's expectations. Confirmation IDs differ from exploratory IDs; imported old samples never become fresh confirmation.

### ResourcePolicy — `pipeline-resource/v1`

Fields: effective CPU count/quota/cpuset; physical/currently available/engine/ancestor memory; user-authorized fraction0.90; selected historical16/capacity64/workstation90 profile, resolved enforcement ceiling, separate expected active peak reservation and safety headroom; device identity; native slot count and affinity; Torch/native thread limits; active/pending/result/lookahead counts and byte charges; per-slot and aggregate RSS watchdogs; role-wide cache caps; candidate ResourceProfile hash; stream/artifact/archive limits; external/cleanup deadlines. Missing sensors use explicit unavailable reasons. Distinguish observed RSS/shared-memory counters from enforced container limits.

Resource policy changes acquire a new hash. A validation command may lower an invocation ceiling, but it cannot silently reinterpret a hardware comparison performed under different limits. At most one active compute role receives the full envelope.

## Invocation and stage records

### Invocation — `pipeline-invocation/v1`

Fields: unique invocation ID; resolved contract hash; started/ended wall timestamps; monotonic origin/durations; selected scope; unavailable/unrequested registry stages; parent reuse links; phase/gate status; actual build/startup/compute/archive costs; consumed/reserved budgets including conservative charges for unacknowledged work; interrupted flag; output/ZIP paths and integrity; explicit historical T093/T096 dispositions.

An invocation owns a fresh output root. Old invocation records are immutable. Repeating a failed test uses a new invocation, not an edit to a terminal status. A missing terminal record is interrupted/partial, never inferred complete.

### StageReceipt — `pipeline-stage/v1`

Fields: stage ID/version; attempt ID; state/outcome/reason; exact executable/argument vector with no secrets; prerequisite receipt hashes; current source and dependency closure; image/artifact identities; workload/weights/policies; engine/hardware/resource identity where relevant; expected/started/completed work; elapsed/CPU/resource observations; mandatory file hashes; JUnit counts; omissions; reused-from identity; child role audits and cleanup result.

Transitions:

- `PENDING → READY → RUNNING → PASS | FAIL | PARTIAL | INCONCLUSIVE`.
- `PENDING | READY → BLOCKED` on failed/missing prerequisites.
- `PENDING → REUSED` only after validating a complete matching PASS receipt; preserve its original producer/measurement identity.
- `UNAVAILABLE` and `NOT_SELECTED` describe registry/scope availability; neither counts as execution.

All terminal states are immutable. Stage-level PASS does not imply scientific success; a valid report of a failed performance hypothesis can complete a reporting stage while `adoption=NO_ADOPTION`. Top-level status retains separate `selected_scope`, `engineering`, `learning`, `research`, `portability` and `long_run_admission` results.

### DependencyClosure

Fields: versioned rule ID; explicit source/build/config/test/adapter paths and hashes; prerequisite artifact/receipt identities; exclusions with rationale; closure hash. An unmapped changed path invalidates relevant receipts conservatively. A whole-source/package check remains mandatory for the current image. Timing reuse additionally requires identical effective hardware and comparison conditions; reports disclose age and producer source even when compatible.

## Logical work and evidence

### LogicalJob — `pipeline-job/v1`

Fields: contract hash; model snapshot hash; group/mode/arm; stable prompt ID and ordinal; attempt ordinal/method/seed; candidate original/final source hash when available; numerical/fit/selection/archive policy hashes; logical work reservation; expected outcomes/outputs; parent group/prompt identity.

`job_id = hash(contract, snapshot, mode, arm, prompt_id, attempt_ordinal, method, seed)`. Absolute clocks, process IDs and arrival order are not inputs. Retain the old timestamp-derived attempt ID only as a legacy cross-reference. Duplicate logical IDs within one manifest are invalid. A retry cannot create a fresh attempt allowance by changing its presentation name.

### JobExecution — `pipeline-job-execution/v1`

Fields: logical job ID; invocation/stage ID; queue/slot/global generation; enqueue/dispatch/finish/owner-observed times; original absolute parent and service deadlines; reserved/consumed work; generation/raw/fit/reference dispositions; worker/resource failures; source/assignment/verification refs; cache provenance; publication ordinal; late/duplicate/stale flags.

Normal state path: `PLANNED → RESERVED → GENERATING → NATIVE_QUEUED → NATIVE_RUNNING → REFERENCE → READY_TO_PUBLISH → COMMITTED`. Skip inapplicable stages explicitly, for example templates do not use neural generation. Any nonterminal state can end as `UNSTARTED`, `INCOMPLETE`, `CANCELLED`, `RESOURCE_EXHAUSTED`, `INVALID` or `FAILED` with a stage/reason. A conclusive unsupported or visible mismatch is a completed operation but not an accepted solution. A terminal disposition is committed exactly once; late duplicates append a rejection observation without changing it.

Record distinct notions of scheduled work completion, generation completeness and scientific scoring coverage. A fully consumed token budget can have a recorded terminal operation while generation remains incomplete. That does not make it verified or automatically suitable for fixed-work equivalence; comparisons must verify matching actual work and classify deadline/unknown cases separately.

### ArmEvidence and SelectionSeal

Arm evidence records scheduled/started/completed attempts, required stage dispositions, all failure reasons, selected candidate ref or absence reason, visible fit and independently completed100-index production/reference scoring. Compact records exist for every prompt/arm before verbose examples.

`visible-only/v2` semantic seal binds prompt/snapshot/policies, selected source, stable attempt provenance, original/transformed hashes, exact assignment and independent verification hash. Measured duration is in a separate evidence record so it cannot change the semantic identity. v1 seals remain validated by their original schema. No selection plus unknown/resource-exhausted generation/fit is incomplete; conclusive no-fit can be a complete negative. Scoring a sealed selection requires both production/reference to complete before its scoring deadline.

### CacheEntry and Reservation

Fields: semantic key, namespace/arm, producer contract and job, payload hash/byte charge, exact completion/discovery status, resource and solver/reference policies, admitted assignment provenance, timestamps as observations, reservation/publication state. Only complete qualified payloads may be reusable complete fit results; partial discoveries are diagnostics. A hit still undergoes current-deadline independent admission. Cache publication is atomic under one owner and cannot exceed aggregate reserved bytes.

### CheckpointBoundary

Use the existing `bigint_state.CheckpointStore` and complete learning state. Add effective scheduling/selection/archive/measurement contract hashes, stable completed logical frontier, cumulative reservations/debits and a drained/cancelled synthesis-block receipt. A resumable boundary has zero unaccounted active/reserved candidate work. Preserve optimizer, RNG, data order, curriculum, replay and selection. Reject a changed-policy resume; weights-only diagnostic snapshots are explicitly nonresumable.

## Study and gate records

### StudyDecision — `pipeline-study/v1`

Fields: kind (`learnability`, `representation`, `curriculum`, `replay`, `assistance`, `encoder`, `final`); prospective decision ID; frozen hypotheses/arm roster/checkpoints/configs/seeds; strict/assisted lineage and reviewed inputs; prior decision hashes; scope-specific cost receipt; mandatory qualification identities; fixed denominators and20/100 horizons; shared/per-arm budgets; selection/stop/promotion rules; authority and final-plan reference when applicable. Invalid/unreviewed authority fails before mounting data or starting training.

States: `DRAFT → FROZEN → ADMITTED → RUNNING → COMPLETE | NEGATIVE | INCONCLUSIVE | INTERRUPTED`. Missing gates yield `BLOCKED`, not a successful scientific result. A manifest may document an explicit prospective deferral; it cannot mark the unexecuted study complete. Final cohort retains existing `sealed → authorized → exposed` lifecycle, with exposure recorded before prompt access and no transition back to sealed.

### QualificationReport

Map each gate and carryover task to current evidence, compatible historical evidence, missing prerequisites, disposition and closure condition. Record T093 as deferred, T096 as historical negative and separately named new attempts if ever admitted. Separate engineering readiness from learning success, research completion and permission for a longer run. A no-go is a completed report, not a promotion.

### BundleManifest

Fields: invocation/contract identity; archive version; each member's relative path, byte count and SHA-256; mandatory/optional designation; omissions with reason; producer scope/source; excluded private categories. Reject path traversal, symlinks, unreadable mandatory records, hash/size mismatch or contradictory summary state. Core job/stage evidence takes priority over optional logs/traces. The archive is evidence, not a checkpoint and not a source of new execution authority.
