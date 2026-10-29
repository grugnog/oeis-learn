# Updated pilot preflight: primary plumbing PASS, long-run NO-GO

Reviewed the operator's `009-learning-pilot-preflight-20261026T213151Z.zip` with the
[read-only reproducible audit](pilot_preflight_review.py). The [machine review](learning-pilot-operator-20261026T213151Z.json)
verifies all391 inventoried payloads, no omissions/hash mismatches, and exact current
runtime source `sha256:2b57e82976f9aaa322c105e0180edffcb38bda3141e32749652e554dcffbb043`.
The full invocation takes14507.200s. No runtime source or scientific policy is
changed by this review; historical source-bound failures remain valid history.

## Work that now succeeds

The full524288-counter cache is reused with the expected v2 binding/index hash;
no reference math is regenerated. Preparation takes6488.094s and admits all33024
selected programs through native/reference agreement. Counts are32768/128/128;
the actual eight exposed OEIS development inputs are used. Structural development
remains128 A-stage rows/18families; parameter development has no C-stage rows.
These are limitations of the registered corpus, not broad structural generalization.

Pinned qualification passes802 CPU tests (JUnit count includes subtests),23 HIP,
45 policy,123 retirement and1115 workflow cases with9 skips. Primary isolated
operations/recovery and numerical execution pass. Docker is explicitly unavailable
as an independent engine: the command reports Podman metadata. Do not claim Docker
portability or request a new host engine installation.

All three actual104496662-parameter snapshots and the15+15-minute full-state
restart complete. Training reaches4710updates,4568750nonpadding tokens,
32768unique examples and2.299effective token epochs in1812.988learner-role seconds.
Every native boundary is drained (active=reserved=0). This passes the short
plumbing test without a learning-success or loss-decrease threshold.

Teacher-forced token agreement improves; it is a diagnostic, not independent
sequence continuation success. Incomplete hidden-scored outcomes remain unknown.
No learning ceiling/extrapolation is supported by these three snapshots.

## Exact remaining blocker

| Snapshot | COMMITTED | INCOMPLETE | UNSTARTED | Full attempt cost observation |
| --- | ---: | ---: | ---: | --- |
| Initialization |2368|1184|1776|Incomplete|
| After first learner leg |3085|2243|0|Complete|
| After second learner leg |3016|2185|127|Incomplete|

All5328 registered attempts per snapshot remain in the public ledger. The
initialization correct arm attempts only592 of2368 slots;1776 remain unstarted.
The last correct arm leaves127 unstarted. Grammar dead ends/token caps, native
four-second deadlines and generation deadlines are separately retained. Neither
an invalid decode nor a missing selected program means that an unstarted attempt
has been timed. Financial/cost completeness and scientific scoring completeness
are distinct: the middle observation is cost-complete despite incomplete science.

The implementation deliberately caps each portfolio by its original60s AND a
finite proportional group/split allocation. Those clocks explain the unstarted
work; the archive does not establish a scheduler or GPU fault. The admission gate
correctly refuses these observations. Do not relabel them complete or retry the
same unchanged preflight until it happens to finish more attempts.

As a diagnostic only, applying the existing doubled componentwise envelope to
the incomplete observations yields generator4425s + scorer762s per snapshot,
or31122s over six snapshots. This already exceeds the registered28800s evaluation
reservation by2322s. It is NOT a complete/admissible forecast. Setup/corpus costs
remain separately charged; original local reference-generation total stays UNKNOWN.

## Disposition and next implementation

L004 and exact-source T110 close. L013 closes with a concrete **NO-GO** for the
current long-run profile. L012's complete measured-cost handoff, G005's full cost
clause and long-run L014/L015 remain open. Original counts become88complete /
8engineering pending /15research held. Accepted hardware/default/history and
missing independent Docker remain unchanged. No long run is admitted.

Preserve the entire target directory, especially prepared data, latest full-state
checkpoint, all three diagnostic weight snapshots, and evaluation settings. The
public ZIP intentionally excludes these private payloads and cannot substitute
for them. No additional cache download or reference/native regeneration is needed.

The next code package is described in the appended [prospective calibration plan](learning-pilot-plan.md#scoped-calibration-follow-up).
It must complete local checks before any new target command. No target command is
issued by this review; no repeated training, size/batch ladder or hardware soak.
