# Accepted native CPU capacity screen

The corrected `009-native-capacity-20261025T040709Z.zip` passes at producer
`176ecc9e2f5a9ff83107d4d74dd8ecc65505cdbe`. All517 listed members plus the
manifest verify, and its709-entry source closure matches exactly:
`sha256:4534bfdc83f9faf7132830a4d281c55fb8e6c52f129f2f4e3339fcbe7504308c`.
The3,256,466-byte archive SHA256 is
`5449f91f92665a6be42060ab3521f21ecf6363e2888f725a76e2f93de70dc7b2`.
[Machine-readable review](native-capacity-operator-20261025T040709Z.json) retains
all receipt identities, child/calibration/pairs, trial counters and clock checks.

136 host tests and655 CPU tests pass, with one explicitly deselected case and17
passing subtests. All three actual packaged-native1/2/4 correctness fixtures pass.
No HIP role is selected by this CPU-only scope. All six outer stages and18 native
slots pass, including a complete queued diagnostic and all17 wave trials. Each
has18 complete rows:324 result rows,306 strict wave-job/debit records and1224 wave
clock events. Total invocation436.09s (7m16s), including53.62s inside the native
performance role. There are no missing or favorable-subset comparisons.

## Component scaling

Every ordinary arm uses the same full18-row corpus, a fresh client/pool and empty
private native caches. Primary intervals include client shutdown and trial export;
image/role startup stays separately observed. Warmups are excluded.

| Workers | AB:1-worker → candidate (s) | BA:1-worker → candidate (s) | Paired interval reductions | Candidate median (s) |
| --- | --- | --- | --- | --- |
|2|3.54046 →2.03062|3.53268 →1.84340|42.65%,47.82%|1.93701|
|4|4.80475 →1.40715|3.55266 →1.11444|70.71%,68.63%|1.26079|

Both named worker hypotheses pass the frozen exploratory5% rule. Four workers are
the experimental selection:69.67% median paired interval reduction, roughly3.3×
paired throughput, with12.79–16.15 completed rows/s. The slower4.80475s serial
control is retained; it is not discarded. No population confidence or full-pipeline
speedup is inferred from two component pairs. Production defaults remain serial.

Three baseline calibrations are3.60103/3.62261/4.06922s. Their frozen reservation
is30s, derived only from complete baseline receipts. The queued diagnostic also
completes all18 rows in3.50642s on this collection; it is neither a paired control
nor evidence that historical queue expiry did not happen.

All306 wave jobs preserve immutable admission/deadline/dispatch/completion/owner
clocks and exactly-once terminal charges. The largest admission-to-dispatch wait
is4.49ms. Every later wave is admitted after the preceding wave's last owner
observation; no allowance renews at dispatch. Each trial has its complete18-row
plan,72 events and terminal status. Admitted assignments and verification agree
with the frozen full-corpus baseline semantics.

The separate observed-only1/4-worker trials have4/2 aggregate1Hz samples and
approximately0.95/3.27 CPU equivalents over their sampled cgroup intervals,
against a16-equivalent quota. This is sparse, short-interval utilization evidence;
CPU observation overhead remains uncalibrated and these trials never enter ratios.
The4-worker observed slot batch counts are5/5/4/4. The whole-role sampled memory
upper bound is257,986,560 bytes (246.04MiB), with a112.45GiB enforcement ceiling.
All memory accounting qualifies, no breach occurs, device-free isolation passes,
and external cleanup is0.104s within5s. Native thread counters are retained
separately from the one-configurable-numerical-thread policy.

## Qualification and next hardware work

Review independently reruns bundle verification, source/receipt/file seals,
complete semantics and every wave clock/debit/event relationship. Baseline child
calibration, both comparison reports, tie-aware selection and all18 final slot
identities recompute exactly. No raw model/hidden data is needed in this CPU scope.

The already-completed producer Foundation CI40160282432 succeeds; Container/native
CI40160282342 is cancelled, not a passing portability result. No new CI is awaited.
Current operator CPU/package qualification is accepted for this affected source.
009:N004 and T110 close here; original counts78 complete/18 runtime pending/15
research held or blocked. The earlier failed archive remains recorded.

The native4 and earlier GPU128 component results motivate a separate bounded
co-scheduling/parity/resource and repeatability investigation. They do not yet
qualify useful generated-program throughput, full-pipeline cold/warm confirmation,
Docker portability, persistent-trainer stability or a multi-day training run.
No unchanged native screen or GPU ladder repeat is requested. Learning-performance
and curriculum work remain held by user priority.
