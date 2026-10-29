# Combined hardware experiments: finite capacity ladder v1

The user requests combining useful hardware experiments to minimize handoffs.
The accepted batch8/16 archive b075bcee4348130494859e56110ad69c1c0318d0ae32d54b8e4d377b7786a5a5,
source10ea8ca7c2cc459cacdd9ec886e8ae198b9032b8bf725a98600d276c4e6912e4,
report7cc536cc6b34fe48113a592dc401e7adb4960529f7149df71a16a88888fac0ed,
shows48.96–49.00% lower complete interval and about1.71GiB sampled memory at16.
This meets the prior condition for a named32 comparison. A new prospective scope
also preregisters64/128 as conditional capacity hypotheses, with fixed stop rules;
the old manifest/32-row results are unchanged. No open-ended size search follows.

Use `pipeline-hardware-validation/v1`, scope `engineering-hardware`, registry
`pipeline-hardware-registry/v1`, phases1/2, workload `replay-capacity128/v1`,
comparisons `telemetry-overhead/v1` and `batch16-capacity/v1`. All numerical,
authority, phase/global/build/disk/cleanup/resource ceilings remain. One invocation
shares image checks, CPU/HIP fixtures, visible-only inputs, reference and archive.
Isolated timing stays serial; intentional native/GPU overlap is separate work.

## Frozen equal work and observation policy

Freeze the same32 distinct-ID twenty-term visible rows and cyclic mapping
`logical_row % 32` into128 logical rows, each with1024 supplied BOS tokens.
Encoder memories are physically replicated; each row has an independent KV cache.
This is a disclosed capacity fixture with32 unique visible inputs, not128 distinct
prompts or generated programs. Every arm processes131072 supplied tokens and640
row boundaries at1/32/128/513/1024. The CPU full-prefix oracle for the32 unique
rows is expanded through the frozen mapping; all128 GPU row outputs must verify.
Eager FP32, hoisted/v1, weights, row order and numerical/top-one rules match.

Before primary timing, exactly one fresh ordinary batch16 role then one fresh
aggregate1Hz role complete the same work. Both retain the mandatory resource
watchdog/isolation checks. Use verified intervals to calculate telemetry/ordinary
minus1. If overhead exceeds5%, use ordinary primary timing for all subsequent
roles; otherwise use aggregate1Hz. The observed role remains available for
utilization inspection. Role/host startup durations remain separately reported.
One pair is a bounded policy calibration, not population confidence or evidence
that memory-watchdog overhead is zero. Unknown/incomplete observations cannot
select a primary mode. No token trace or per-token synchronization is introduced.

## Baseline, finite branches and stopping

Run one batch16 cold role, excluded warmup and exactly three complete batch16
calibrations in the selected mode. Each has600s external/570s internal maximum.
For each prospective branch32/64/128, seal a child from these same baseline-only
receipts, immutable source/validation/reference/workload/weights, selected mode,
actual2/1 threads/device/memory, prior evidence basis, candidate allocation and
ordered slots **before its first candidate observation**. The roster, derivation
and stop rules are already fixed by this parent contract.

Each branch has one excluded candidate warmup and exactly two fresh AB/BA pairs;
A=batch16, B=that named size. Component reservation is
`ceil(max(60,2*max(three verified baseline intervals)))`, capped at280s. Add30s
internal setup and30s outer startup/cleanup, so external maximum340s. A branch
reserves1800s, including all five roles, cleanup and reporting; remaining parent
time must cover the next full child reservation. No clamping, extra sample or retry.

Predict peak as the larger of8GiB plus12×weight bytes and
`candidate_batch/16 × max observed baseline role memory upper bound`. Add the
larger of1GiB or10% peak as headroom; leave at least10% of the enforcement ceiling
unreserved. Current availability is rechecked at dispatch. Missing accounting or
capacity failure stops expansion; a prediction is not a measured candidate peak.

After a complete branch, advance only if both gains versus16 are at least5% and
the candidate median is at least5% faster than the preceding eligible size's
median. For32 the preceding size is baseline16. These screening rules are not
production adoption thresholds. A complete negative result stops larger branches;
their role ledgers explicitly say NOT_SELECTED. Incomplete/parity/resource failure
stops dependent work, preserving partial/unstarted records. No favorable subset
can produce a gain. Investigation/reporting PASS may describe a complete negative
decision; unexecuted roles never become executed success.

Select an experimental size only from complete qualified outcomes: fastest
candidate median, preferring the smaller size within5%; baseline16 stays available.
No production default changes. Every report denies optimization adoption and
research/learning admission. No training-batch/model-size or algorithm change.

## Bounds and evidence

Phase2 reserves600s HIP,720s observation roles,3000s baseline roles,5400s for three
branches and90s reports:9810s, below10800s. CPU reference600s belongs to phase1.
Existing shared per-engine14400s build,86400s global,900s archive/300s emergency,
five-second role cleanup and artifact/memory ceilings remain.

Every branch plans all five nested role ledgers before dispatch and binds public
artifacts/child identities. Durable640-row plans/events and terminal states remain.
Private raw memories/oracles/weights are excluded. One archive retains selected
mode, complete/negative/skipped/failed branches, CPU/GPU/memory observations or
explicit missing reasons, startup/cleanup and source-bound qualification.
Ordinary fallback lacks per-candidate sampled utilization; do not invent it.
This is forced replay component evidence, not full-pipeline confirmation, learning
quality, a multi-day training decision or final-data authority.
