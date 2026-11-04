# Data model and transaction design

New versioned records extend the existing sealed identity/closed-field conventions.
Exact version names below are planned contracts, not claims that validators exist.
Keep immutable historical studies and receipts readable with their original policy.
Reject unknown fields/versions at new authority boundaries; do not use permissive
wrapper dictionaries to bypass canonical validation.

| Entity | Required content and relationships | Validation / visibility |
| --- | --- | --- |
| `PAC-screen/v1` experiment | Arms, comparison factors, seed/initializer, pool/probe/target identities, curriculum policy, budget graph, metrics and control disposition | Frozen before outcomes; no old P3 relabeling; public safe projection |
| Control compatibility receipt | Original study/source/image identities; checkpoint/initializer refs; ordered data/model/optimizer/exposure/diagnostic checks; MATCHED or REFERENCE_ONLY with reasons | Missing proof is not match; private paths/weights excluded from shared output |
| Family inventory | 32 IDs; original row offsets/hashes; probe/anchor lists; sidecar hash; source/exclusion disposition totals | No empty family; exact counts; learner receives training role only |
| Target pair proof | Concrete ID, both sources/token views, assignment/sharing, protected dependency disposition, independent ground/execute receipt | Equal underlying rows; fallback retains original literal; private source handling follows existing dataset boundary |
| Observation | Unique run/arm/segment/block/panel ID; pre/post model-update bindings; loss sums/counts, signed raw gain, charged-token cost, selection-time probability, scaler quantiles/reward and censored reason | Same panel around one contiguous family block; common trends are monitoring, not reward credit; no false zero for unavailable |
| Review state | 128 anchors: acquisition flag, queue level, last committed exposure, due update, grade/event ID and one obligation each | Grade belongs to that anchor; initial acquisition and incidental exposure separate; no promotion without committed update |
| Adaptive state | Policy/inventory IDs, stable logweights, action count, valid kernel-update count, selection probability, bandit/scaler RNGs, 4,096-value reward reservoir/seen count, quantile rule, cycle/block cursors, per-family row RNG/permutation/cursor and probe rotation, consecutive censored-reward/invalid-grade counters, pending pause/terminal disposition | Named kernel and epsilon-derived probability bounds; reject old visit-scheduler state; complete identity-bound restore |
| Pending transaction | Block ID/family, frozen selection probability, pre-probe IDs/results, planned/completed inner updates, pending post-probe/reward phase, current batch IDs/tokens/debit and review grade | At most one pending learner update within one pending block; never repeat completed gradients or apply feedback twice |
| Complete checkpoint | Existing model/optimizer/scheduler/device and host RNG + data + entire adaptive/review/probe state + token/time ledger | Atomic generation manifest hashes every component; weights-only is not resumable |
| Progress event | Monotone sequence, completed exposure, reserved/discarded work, timing windows, checkpoint time and resource/progress status | Small bounded journal; not authority to restore newer scheduler state |
| Stage receipt | Input/source/image/policy/helper hashes, parent receipts, command identity, process/application/cleanup status, artifacts and omissions | Reuse only complete compatible closure; primary/secondary errors both preserved |

## Transaction and recovery ordering

1. At a new bandit block boundary choose the family once, retain its selection-time
   probability and record the pre-probe. Freeze the selected row/panel cursors and
   block length. Reserve each exact batch debit before its optimizer update.
2. Commit each successful optimizer update, exposure and incidental anchor effects
   together. Snapshotting may occur at an update boundary inside a block, but the
   checkpoint must retain its pre-probe, chosen action and completed inner cursor.
   Never substitute a new family or insert review while that block is unfinished.
3. After all eight updates, measure the same post-probe and atomically commit the
   reward/scaler/kernel update exactly once. If paused here, resume this observation
   before any new gradients. Missing observation leaves q/scaler/valid clock unchanged;
   action/debit counters and its censored event remain. A terminal partial block
   cannot masquerade as a completed rewarded block.
4. In the review lane, record an acquired anchor's grade before its optimizer update;
   only the successful update commits the grade-driven queue transition and new due
   time. Initial acquisition and incidental exposure cannot invent a recall grade.
5. At a snapshot boundary synchronize device work and serialize model/optimizer,
   RNG, all pending block/probe/scaler/review state and ledgers together. Atomically
   publish the complete manifest, retaining the previous verified generation.
6. Resume restores that complete generation. Later journal entries can debit spent
   but lost tokens/time, not advance its model-bound policy. Mark lost events discarded
   and use distinct resume-segment identities for recomputed work. Preserve valid
   completed kernel updates and queue services without duplicate application.

Clean boundary pause/resume must reproduce weights, optimizer, ordered decisions,
probe outcomes, reservoir contents/counters, queue state and next batch under the
qualified deterministic backend. An
interrupted unsaved segment may lose up to the chosen snapshot interval; its
consumed budget stays charged, so useful final exposure may be lower. Validate
both behaviors explicitly rather than demanding identical final weights after
irreversibly debiting different crash work.

## State transitions

Experiment: DRAFT -> FROZEN -> PREFLIGHT_PASS -> ACCEPTANCE_PASS -> optional
RESEARCH_RUNNING -> COMPLETE or PARTIAL/FAIL. A failed preflight never transitions
to research. Operational COMPLETE/PASS and scientific INTERPRETABLE/INCOMPLETE/
UNMEASURED are distinct fields.

Stage: PLANNED -> RUNNING -> application terminal -> process terminal -> cleanup
acknowledged -> sealed receipt. Application PASS alone is insufficient. Any error
retains first-cause order; export adds its own outcome rather than overwriting it.
Read-only recovery can add a new export receipt referring to an original failure;
it never mutates that failure into success.

## Retention and disclosure

Private tensors, exact training/probe payloads, hidden evaluation truth and host
paths remain private. Export pseudonymous IDs, sums/counts, policy decisions,
bounded before/after cases allowed by existing roles, original artifact hashes and
explicit omissions. Aggregate progress can be deterministically compacted while
preserving total counts/cost and whole-run coverage. Cap observation windows,
review queues, raw logs, snapshots and copies in the storage graph. No cleanup of
original operator runs is authorized by this design.
