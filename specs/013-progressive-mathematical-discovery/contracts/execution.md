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
| Synthetic scaling | 2 selected seeds, at most 8 total routine slots (each <=24h; aggregate <=168h), plus <=24h diagnostic pool | 192h |
| OEIS pilot | 1 cycle × 2 arms (replay/continued synthetic) × 2 seeds × 24h | 96h |
| Component pilot | library/no-library × 2 seeds × 12h | 48h |
| External-source mixture pilot | with/without source × 2 seeds × 6h | 24h |

The diagnostic pool is suballocated from scaling, not added: at most four diagnostic tasks, each <=12h and aggregate <=24h, with concrete triggers, ancillary reservations and maximum visits frozen under [contingencies.md](contingencies.md). A smaller stage subset does not inherit this pool automatically.

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

`resume` continues the same task's remaining allowance in a new owned invocation directory. A preallocated next task starts only after compatible boundary publication and the declared dependency/gate or a separately frozen exploratory trigger. The evidence verdict and action remain distinct; exploratory work never marks a failed gate passed. No direct reopening of a terminal historical 012 run. A weights-only import starts a labeled child experiment with new optimizer/state; it does not claim exact continuation.

Refutation appends a quarantine record and enumerates descendants through target → archive/library → checkpoint → dossier. Historical bytes and previous results remain readable but invalid for the affected claim. Resume from a clean ancestor retains spent cost and contamination history.

## Qualification and evidence

Reuse the real qualification producer and extend feature acceptance for accumulation, refreshed streams, proposal replay, task-boundary recovery, OEIS channels and component vocabulary changes. Every selected model family/shape/dropout path requires finite actual-HIP updates, full save/restore and deterministic continuation under its qualified policy. Existing hardware-only sizing tests are insufficient. Include repeated cleanup/restart and bounded hangs; do not use CPU fallback for a GPU gate.

Progress <=60s includes phase, actual committed/charged exposure and remaining reservations, or explicit unknown/stalled state. Do not rely on the legacy 128-event phase log as a permanent heartbeat. Record queue and service clocks, batch sizes, memory/CPU counters and available mapped render-node sensors; unavailable sensors remain null with reasons. Overlapping timers cannot be summed into wall time.

`report` produces one closed-inventory archive, maximum 512 MiB uncompressed metadata, with member byte/hash seals, stage/trial receipts, exposure/metrics/gate inputs, clocks, qualification references, omissions and original producer error/traceback. Exclude weights, full private continuations, raw corpora and unrestricted logs. Required reconciliation tables cannot be dropped to fit: partition task tables deterministically or return REPORT_INCOMPLETE and preserve local output. Failure packaging must preserve the first failure as well as any collector error. A reference hash alone cannot make missing producer authority verified.

Offline verification checks exact inventory, seals, reference closure, stage ancestry, qualification scope, budgets, cohort/counts, aggregate metric calculations and decision rules. `mechanical_status`, `scientific_status`, `qualification_status` and `claim_status` are independent fields. A successful process or package cannot upgrade research or proof status.

Exact iteration calibration records (`progressive-iteration-calibration/v1`) bind
the current source/image and prepared learning compatibility. They measure two
complete logical updates, including initial Adam allocation, two boundary probes,
and durable full-state save plus verified restore at maximal prepared target and
numeric input sizes. Freeze requires at least twice each measured phase, and
includes the measured tensor closure in peak-live storage. Research requires an
actual-HIP calibration; CPU records remain diagnostic. Generic short-case timing
cannot authorize an exact research iteration. Both learner preflight and the
write-ahead debit refuse work unless the complete next iteration and finalization
can fit. The independent supervisor hard deadline remains in force.

## Prospective measured runtime amendment (2026-11-12)

The maintainer requested resolution of the actual workstation planning deficits:
full-domain exact calibration projects about 13.15 hours before overhead, versus
the original 12-hour allowance; selected initial-round evaluation projects about
330.49 hours, versus the original 136-hour ancillary envelope. The original
536/136/672-hour defaults and old frozen records remain unchanged.

A fresh, explicitly selected `progressive-research-round-design/v3` and
`progressive-research-reservation/v2` may reserve the same capability, diversity
and tuning design under the following operational limits. These are ceilings,
not required work or expected duration; no trial, look, panel member, token,
learning setting, per-prompt 60-second portfolio, 4-second fit, or assessment rule
changes. This amendment does not authorize later scaling/OEIS/component/source
branches without their separate finite reservation and eligibility.

| Boundary | New prospective ceiling |
| --- | --- |
| Exact-input calibration | 72 hours aggregate; existing independent per-case watchdog/cleanup remains |
| Enclosing preparation command | 72 hours + 30 minutes dispatch/publication headroom |
| Screening | 96 hours shared per trial, including all three looks and 30-minute finalization |
| Screening training child | 4 hours; cumulative tokens remain 1M → 2M → 4M |
| Diversity/confirmation training | 24 hours per task; 25M targets and 400M schedule unchanged |
| Evaluation role | At most 72 hours, with a frozen member-based allocation |
| Selected acquisition/preparation/qualification/evaluation/packaging pools | 24 / 96 / 24 / 3,000 / 4 hours |
| Selected aggregate ceiling | 3,600 hours, with at most 536 training-task hours |

Generation allocates 300 seconds/member for primary methods or 900 for the full
control set; scoring allocates 30 seconds/member. These operational estimates
include measured safety margin and previously unexercised services, plus the
original role finalization reserve. They do not enlarge each program's scientific
budget. The exact 376-role construction reserves 480 training-task hours and
2,685.99 evaluation/analysis hours, 3,265.45 hours including setup/reporting. Shared
trial evaluation is counted once under evaluation, rather than called training.
The much smaller unpadded measured initial-round projection remains approximately
367–401 hours plus preparation; conditional early stops may reduce it.

The new campaign marker must bind the original revised reservation and measured
execution contract at freeze. Generic/legacy roles retain six-hour guards and
old study/calibration versions retain their original bounds. Only validated
installed discovery evaluation authority receives the expanded deadline.
Training never exceeds 24 hours. Both sampled evaluation and exact-input measured
feasibility remain mandatory; raising ceilings cannot manufacture qualification
or bypass infeasibility. Rollback means selecting the historical version in a
fresh unexecuted preparation, never editing an existing freeze or refunding costs.

## Prospective measured storage amendment (2026-11-12)

Measured-resource policy v3 may reduce disk partitions from original exact-input
calibration v2. The producer measures allocated Adam state once and separately
identifies weights. Each task reserves its largest permitted calibrated shape,
128 MiB context per generation, two publication generations and applicable first
milestone pins. The initial 25M round reserves one milestone pin; screening tasks
below 25M need none. Evaluation projections reserve weights rather than optimizer
state. All role metadata and every retained task partition remain charged.

Clean terminal publication may retire a redundant predecessor, preserving the
latest full state, milestone pins and immutable recovery history. Copy-on-write
projections must have distinct inodes and an ordinary-copy fallback; their full
logical bytes remain reserved. No projection with an unreclaimed worker may be
removed. Shared package bytes must remain physically free but are not duplicated
inside each task quota. Runtime checks protect the next atomic publication, while
the complete campaign storage requirement remains a freeze gate.

These changes do not reduce token horizons, trial count, panels, precision,
runtime limits, recovery authority or required historical evidence. Old frozen
campaigns and calibration records are not rewritten. A fresh source-bound
preparation must establish the new reservation before training.

## Positive service timing amendment (2026-11-13)

Prospective evaluation requests v2 must exercise successful full-horizon scoring
with four size-selected original training programs, excluding reserved probes,
and two isolated repeats each. Bind original native/reference results, source,
image and support; partial or mismatched service evidence blocks freeze. These
witnesses measure resources only and cannot establish learned capability.

For scoring feasibility, extrapolate the greater of measured model scoring and
positive service timing using the generation task's actual three/eight-arm scope.
Charge every selected arm independently, retain owner overhead, and apply the
existing fourfold margin. Samples are estimates, not guarantees: existing runtime
clocks and all other resource limits remain authoritative. Legacy requests retain
their old unmeasured-path reservation on replay; do not rewrite historical results.

An unused preparation owner stopped specifically by the legacy scoring-only
feasibility gate may be replaced after original training-only inputs, successful
qualification, complete measurements, clean termination and absence of campaign
construction are verified. Preserve evidence and consumed diagnostic costs, fence
the old owner, and obtain current-source qualification again. No scientific
allowance or historical qualification authority transfers.

## Qualified service-role correction

Evaluation request v3 binds the CPU image for positive native/reference services
and the GPU image for model timing, both from original current qualification.
Historical census native identity remains unchanged; runtime binary equality is
still checked exactly in the CPU service role. The serial CPU stage is charged
against the existing aggregate evaluation timing allowance. Case success alone
cannot authorize freeze: original stage completion, process exit and acknowledged
cleanup must also verify. Preserve failed stages and continue independent GPU
cases only after safe CPU termination.

A historical v2 attempt whose eight services all failed the CPU/GPU native identity
check before any fitting/scoring may replace unused preparation ownership after
its original training membership, successful qualification, 14 complete independent
model timings and clean termination verify. Retain costs/history and require new
source-bound qualification. No post-construction or scientific work qualifies.

## Construction validation and timeout recovery

Immutable support may reuse successful row/codec validation separately for native
and short-profile scopes. Mutable inputs must be revalidated; task, profile, seed,
batch, initializer, arm and token-supply constraints remain independent. This is
an implementation optimization with byte-identical canonical package identities,
not a reduced calibration or scientific domain.

Construction must emit public phase/count progress and diagnostic stack traces
without private inputs. A terminated original two-hour metadata-only construction
attempt may release unused preparation ownership after successful qualification,
complete feasible training-only timings, exact command identity and acknowledged
process-group termination verify. Reject completed construction, calibration,
frozen campaigns, execution output and scientific work. Preserve partial metadata,
original journals/costs and require current-source requalification.
