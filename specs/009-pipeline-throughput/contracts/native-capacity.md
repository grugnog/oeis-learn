# Native CPU capacity screen v1

After the accepted hardware ladder, isolated GPU128 replay reaches about90.5%
sampled busy while CPU use is about1.16/16 equivalents. This separately registered
engineering investigation concerns native CPU scaling, not further GPU size growth.
Use `pipeline-native-capacity-validation/v1`, scope `engineering-native-capacity`,
registry `pipeline-native-capacity-registry/v1`, phases1/3, workload
`native18-slot-waves/v1`, comparisons `native1-2-4/v1`. Original scopes are unchanged.

## Prospective work and admission

Freeze the existing18-row native corpus before any timing:64/256/512/1024/4096-bit
identity and degree2/4 cancellation cases, affine, nonlinear and no-fit strata.
There is no cold-selected subset in this scope. All18 rows must complete and
preserve admitted assignments/verification for a comparison to be scored.
A complete negative fit may be valid work; an incomplete/timeout result is not.

Retain one old queued-eight/one-worker diagnostic with its original admission
and four-second service clock; it is not a baseline or a candidate retry.
The new `slot-waves/v1` policy constructs at most `workers` requests per wave.
A later wave acquires its one allowance only when admitted after the preceding
wave drains. This is backpressure before admission, not renewal after queueing.
After admission every request has the same immutable `min(parent_deadline, now+4)`
deadline, including dispatch/queue/owner observation. The parent clock never moves.
Every logical row is attempted at most once per fresh trial; pending/expired work
is explicit and never silently retried. The production scheduler/default is unchanged.

## Fixed experiment roster and bounds

Every trial has a fresh client/pool and empty private native caches. Shared
role-wide memory/CPU/byte and native per-slot/aggregate limits remain enforced.
Run wave1 cold, excluded wave1 warmup and exactly three complete wave1
calibrations. Each diagnostic/baseline has at most180s internal and the whole
isolated CPU role has4500s external (4470s internal), inside phase3's7200s.
Unknown/incomplete baselines block both candidate comparisons. Do not narrow work.

Before the first candidate observation, seal one child binding source, parent
manifest, full corpus, exact18-row baseline semantics, all five baseline receipts,
calibration, workers[2,4], policies, slots, fixed stop/selection rules and memory
ceiling. Child reservation is `ceil(max(30,2*max(three baseline intervals)))`,
maximum270s; over-cap blocks rather than clamps. Reserve ten such trials plus
two observed trials and60s for remaining cleanup/reporting before starting them.
Every launch requires its complete reservation plus report/cleanup time remaining.
There are no extra samples/retries. Both named sizes are measured independently;
an unsupported2-worker gain does not remove the predeclared4-worker hypothesis.

Each comparison has one excluded warmup and two fresh AB/BA pairs, A=wave1,
B=wave2 or4, ordinary timing with the mandatory watchdog. A pair ratio requires
both full18-row trials, all baseline semantics and clean terminal ledgers.
Select a candidate only when both reductions are at least5%; choose fastest
eligible median, preferring fewer workers within5%, retaining1 when unsupported.
These two-pair component screens cannot grant five-pair cold/warm adoption.
Two separately named observed-only wave1/wave4 trials follow ordinary comparisons
when the child exists; aggregate1Hz data describe CPU/process/memory usage but
never enter scored ratios. Observation overhead is uncalibrated for this CPU scope.

## Evidence and authority

Plan every nested slot before dispatch. Retain durable transitions, trial seals,
all18 results, per-row admission/dispatch/native completion/owner-observation clocks,
terminal logical-job debits, native service/pool counters, explicit never-started
rows and block reasons. Record both primary wall interval (fresh client through
trial export) and the inner native interval; role/image startup stays separate.
Raw/native caches remain private and excluded. Whole-role watchdog, device-free
CPU isolation and five-second external cleanup are mandatory. Nonquiescent workers,
resource failure and interruption stop globally, with partial evidence archived.

Shared preparation includes host tests, selected engine/current qualification image
and CPU/native regression fixtures. No HIP/GPU role, checkpoint, hidden truth,
learning, production default, full-pipeline cost/adoption, multi-day or final-data
authority is granted. One command creates one archive. No duplicate GPU ladder.
