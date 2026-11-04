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
| Observation | Unique `(run, arm, update, panel, family)` ID; checkpoint/update binding; counts/sums; loss/semantic/completion/cost fields; censored reason | Finite values, actual denominators, known role; no false zero for unavailable |
| Review state | Four anchors/family, cursor, interval, next due update, success streak, last observation ID and pending obligation | Bounded, deterministic ordering; no duplicate review/feedback application |
| Adaptive state | Policy/inventory IDs, RNG/cursors, slot, bounded observations/EMA, frozen cost profile, probabilities and counters | Closed version; floor/cap/queue feasibility; full identity-bound restore |
| Pending transaction | Update ID, chosen family/channel, row IDs, exact tokens, selection probability, pre/post decision state, reserved debit | At most one pending learner update; matching batch on resume |
| Complete checkpoint | Existing model/optimizer/scheduler/device and host RNG + data + entire adaptive/review/probe state + token/time ledger | Atomic generation manifest hashes every component; weights-only is not resumable |
| Progress event | Monotone sequence, completed exposure, reserved/discarded work, timing windows, checkpoint time and resource/progress status | Small bounded journal; not authority to restore newer scheduler state |
| Stage receipt | Input/source/image/policy/helper hashes, parent receipts, command identity, process/application/cleanup status, artifacts and omissions | Reuse only complete compatible closure; primary/secondary errors both preserved |

## Transaction and recovery ordering

1. At a committed update boundary, choose the next batch from committed policy/RNG;
   reserve exact token debit and persist the bounded pending-work journal. Do not
   persist a full model here merely to record the reservation.
2. Execute one optimizer update. Only after success commit model/optimizer cursor
   and exposure, then any due measurement and its uniquely identified feedback.
   Feedback can influence the next decision only after this commit boundary.
3. Scheduled review advances only after its real optimizer update succeeds. Probe
   events bind the observed model update, never a newer or older unlabeled model.
   A failed/censored panel records its disposition without fabricated outcomes.
4. At a snapshot boundary, synchronize required device work, serialize all state
   from the same committed boundary and atomically publish a complete generation
   manifest. Keep the previous complete generation until the new one verifies.
5. Resume selects the newest verified complete generation. Restore all RNG, panel
   cursors and pending choices together. Later journal entries can establish spent
   but lost tokens/time; they cannot replace model-bound adaptive state. Mark lost
   events discarded and assign new resume-segment identities to recomputed work.

Clean boundary pause/resume must reproduce weights, optimizer, ordered decisions,
probe outcomes and next batch under the qualified deterministic backend. An
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
