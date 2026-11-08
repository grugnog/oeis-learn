# Execution, isolation and recovery — discovery-runtime/v1

Proposed contract; not implemented. Authority: FR-004–006, FR-022–025, FR-035, FR-038–041.

## Frozen graph and resources

The stdlib host controls the engine and supervises registered operations; each installed role gets explicit read-only inputs and one owned output. No role receives an engine socket, the workspace root or network except allowlisted source acquisition. One learner/generator GPU role runs at a time. All stage IDs, dependencies, decision edges, continuation slots, resource envelopes and maximum visit counts are frozen. No generic arbitrary-code DAG or `start-step` bypass.

Before launch, resolve image/source/hardware/shape/numeric qualification, exact workstation paths, all input hashes and finite ceilings. A proposal blueprint cannot confer execution authority. Reuse original compatible receipts only after current-facts checks; closure invalidation remains conservative. Register 013 contracts, source modules, test inventories, dependencies and package assets in `experiments/source_snapshot.py` and `pipeline_validation/closures.py`.

Each training task has cumulative active lifetime <=86,400 seconds, including projections, startup, online search/admission, updates, checkpoints and finalization. Define active intervals from owned supervisor monotonic clocks and boot identities. Record calendar downtime separately. A clean paused interval consumes no active time. If an unclean interval lacks trustworthy elapsed/termination proof, conservatively consume its remaining reservation and stop that task; never infer spare entitlement from wall timestamps.

Reserve startup and finalization before dispatch and before every new update/search. Set reserve from measured worst observed checkpoint/publication/cleanup costs with a frozen margin; default minimum finalization reserve is 30 minutes for research, raised if measurement requires it. Reject plans whose envelope cannot fit startup, useful work and reserves. Native service/scientific portfolio caps remain independent and cannot borrow these generous operational reserves.

Initial maximum research reservation, subject to outcome-blind calibration and explicit freeze:

| Phase | Maximum task roster | Training-task active upper bound |
| --- | --- | ---: |
| Diversity | 2 arms × 2 seeds × 1 task of 12h | 48h |
| Tuning screen | 16 trials × 2h | 32h |
| Finalist confirmation | baseline + at most 3 finalists × 2 seeds × 12h | 96h |
| Synthetic scaling | 2 selected seeds × at most 4 continuation tasks × 24h, shared across 100M/400M milestones | 192h |
| OEIS pilot | 1 cycle × 2 arms (replay/continued synthetic) × 2 seeds × 24h | 96h |
| Component pilot | library/no-library × 2 seeds × 12h | 48h |
| External-source mixture pilot | with/without source × 2 seeds × 6h | 24h |

This upper bound is **536 training-task hours**, not an ETA or an automatic booking. Reserve at most 24h acquisition, 24h mechanical/source preparation, 12h qualification/acceptance, 72h evaluation/analysis and 4h packaging outside those tasks: maximum campaign active sum **672h**, with measured concurrent overlaps reported separately. Heavy independent qualification or a source-specific image build that cannot fit must receive a new freeze before launch. Early failures/gates release unused reservation but never erase spent cost. The operator may freeze a smaller stage subset; the full feature remains implemented and testable. A further OEIS cycle is a new prospective study, not an unbounded repeat edge.

Tokens, draw attempts, search attempts, native work, CPU/GPU active time, total task time, disk, archive bytes and checkpoint storage each have independent caps. Sum predeclared child budgets; do not add already included role time twice to task lifetime. A supervisor enforces the remaining enclosing deadline even if a worker blocks. Freeze a peak-live storage graph using measured checkpoint size, two atomic snapshot generations, retained milestone pins, prepared shards and packaging reserve. Stop before writing if free space/inodes or task cleanup reserve are insufficient.

## Role channels

| Role | Allowed inputs | Excluded inputs |
| --- | --- | --- |
| Acquisition/preparation custodian | Pinned sources, grouping/exposure registry, mechanical generation policy, source-specific reference tools | Model outcomes for cohort construction; engine socket |
| Learner | Mechanical shards, admitted training targets, train-only reward probes, compatible full state/library | Development/confirmation/assessment truth, raw source definitions/metadata |
| Candidate generator/fitter | Visible prompts, checkpoint/library, budgets | Training admission continuations, score truth, witness solutions |
| Training admission | Candidate source and designated training-only continuations; native/reference versions | Development/confirmation/assessment truth |
| Development/confirmation scorer | Frozen candidates and selected panel truth | Permission to update training targets or sampler |
| Assessment scorer | Locked final candidates and assessment truth after explicit assess action | Learning or adaptive selection authority |
| Tuning proposal role | Frozen search space and permitted development scalar history | Candidate truth, confirmation/assessment outcomes, engine or GPU |
| Component proposer/verifier | Verified training archive, typed source and bounded independent checker | Evaluation-derived targets or witness programs |
| Claim reviewer | Sealed candidate dossier and permitted independent verification/literature evidence | Authority to retroactively train on review evidence |
| Offline verifier | Public bounded evidence with hashes/metrics/receipts | Private truth, model weights or execution authority |

Every artifact channel validates its role/type/identity, not just a path. Reject path traversal, symlinks, aliasing and same-file truth leakage across role projections. Source formulas may be used by the custodian to compute terms; they never become learner targets under the initial profiles.

## State and continuation

Separate immutable learning compatibility from mutable task entitlement. Compatibility covers model/numerical policy, optimizer semantics/schedule, language, training/allocation and archive/library update rules. Campaign identity covers budgets, selection edges and allowed task roster. A checkpoint binds both plus the dynamic data/archive/library cut. Changing task output paths or its remaining seconds does not change learning state or grant entitlement.

Atomic checkpoint cut contains weights, optimizer/scheduler, all RNG states, pending complete-update state, charged/useful/discarded token ledger, shard/draw/rejection cursor, sampler/bin state, reward probes, admission archive version, component dictionary/version, cost journal position and milestone history. Accumulation checkpoints occur at optimizer-update boundaries; interrupted microbatches are charged discarded work. Keep the previous checkpoint valid until the new cut and references are durable.

Use write-ahead reserve → perform → verify → commit for batch, target admission and library publication. Pending target bytes not covered by a committed admission record cannot train. Recovery reconciles new progress debits with the older full checkpoint and does not roll exposure back. Retried draws/verification remain charged. Idempotency key = campaign/task/operation/ordinal; replay returns the original committed output rather than applying it again.

`resume` continues the same task's remaining allowance in a new owned invocation directory. A preallocated next task starts only after compatible boundary publication and the declared dependency/gate. No direct reopening of a terminal historical 012 run. A weights-only import starts a labeled child experiment with new optimizer/state; it does not claim exact continuation.

Refutation appends a quarantine record and enumerates descendants through target → archive/library → checkpoint → dossier. Historical bytes and previous results remain readable but invalid for the affected claim. Resume from a clean ancestor retains spent cost and contamination history.

## Qualification and evidence

Reuse the real qualification producer and extend feature acceptance for accumulation, refreshed streams, proposal replay, task-boundary recovery, OEIS channels and component vocabulary changes. Every selected model family/shape/dropout path requires finite actual-HIP updates, full save/restore and deterministic continuation under its qualified policy. Existing hardware-only sizing tests are insufficient. Include repeated cleanup/restart and bounded hangs; do not use CPU fallback for a GPU gate.

Progress <=60s includes phase, actual committed/charged exposure and remaining reservations, or explicit unknown/stalled state. Do not rely on the legacy 128-event phase log as a permanent heartbeat. Record queue and service clocks, batch sizes, memory/CPU counters and available mapped render-node sensors; unavailable sensors remain null with reasons. Overlapping timers cannot be summed into wall time.

`report` produces one closed-inventory archive, maximum 512 MiB uncompressed metadata, with member byte/hash seals, stage/trial receipts, exposure/metrics/gate inputs, clocks, qualification references, omissions and original producer error/traceback. Exclude weights, full private continuations, raw corpora and unrestricted logs. Required reconciliation tables cannot be dropped to fit: partition task tables deterministically or return REPORT_INCOMPLETE and preserve local output. Failure packaging must preserve the first failure as well as any collector error. A reference hash alone cannot make missing producer authority verified.

Offline verification checks exact inventory, seals, reference closure, stage ancestry, qualification scope, budgets, cohort/counts, aggregate metric calculations and decision rules. `mechanical_status`, `scientific_status`, `qualification_status` and `claim_status` are independent fields. A successful process or package cannot upgrade research or proof status.
