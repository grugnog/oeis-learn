# Fresh paired pilot handoff — 2026-11-02 Pacific

T139–T142 replace the infeasible v1 design and add gates before expensive work.
Use the single consolidated command at the top of [quickstart.md](../quickstart.md)
on existing branch `011-controlled-generalization`, PR#62. No HIP/container run was
performed here, and CI was not awaited. Current-source acceptance remains required.

## Design and purpose

Compare repeating a fixed training corpus with receiving fresh examples from the
same fixed family support. Both arms start from identical random weights using the
imported recipe configuration, seed20261016, literal targets, batch16 and no replay.
Historical weights are used only by software acceptance, never the research arms.

| Item | Frozen v2 definition |
| --- | --- |
| Support | 32 generic families: A4/B6/C2/D20, equal proportions. |
| Coefficients | Independent uniform integers[-1024,1024], explicitly versioned. Historical[-16,16] streams remain unchanged. |
| Descriptor screen |16 source-only variants; no19-difference window may recur in more than two variants. This removes fixed arithmetic segments, not all behavioral duplicates. |
| Static | Repeat the first32,768-row shard. |
| Refreshed |12 available shards,32,768 unique sources each;1,024/family/shard. Stop at the frozen token/time cap, not necessarily after consuming every available row. |
| Protected data |128 rows each for selection/assessment × parameter/structural, with unchanged exact20-term cross-role window exclusions and coverage quotas. |
| Training |25M target-token and6-hour cap/arm; retain4/8/16/25M milestones and latest. |
| Selection |256 rows, seven existing controls, latest available checkpoint/arm;6-hour generation and6-hour scoring envelope/arm. |
| Interpretation |Single-seed exploratory; matched exposure and95% primary completion required for comparison. Assessment stays sealed; no automatic confirmation. |

This is a new distribution conditioned on numeric variation, not a retrospective
claim about historical generalization. Both arms share the new law/roster; only
refresh changes between them. Source feasibility uses no model or assessment scores.

## Gates in one invocation

1. **Full CPU reference feasibility:** all four protected cells and all12 shards,
   including exact isolation, per-family quotas, cross-shard source uniqueness and
   actual encoded target-token capacity. A failure starts no engine/HIP work.
2. **Current-source acceptance:** original canonical suites and five feature cases.
   Old-source PASS records remain historical evidence, not qualification of changes.
3. **Canonical native preparation:** independently check every surviving candidate;
   require selected row order to match source feasibility before any smoke learner.
4. **Same-model paired smoke:** one row/family in each of two32-row shards. An exact
   computed cap (<=16,384 tokens) allows at least three batch16 updates and actual
   refreshed-row consumption. Each learner gets900 seconds. Canonical proofs inspect
   full state for finite changed weights; initial weight hashes must match. One fixed
   selection prompt/stratum exercises all seven controls, with1800 seconds per
   generator/scorer. Recorded negatives/censoring are valid; missing/incomplete
   scoring or failed accounting blocks full training. No accuracy threshold.
5. **Full paired research and export:** unchanged learner/inference budgets, existing
   isolated generator/scorer roles and canonical report/verify-bundle. Smoke weights
   never initialize research, and smoke measurements never enter its denominator.

The 96-hour outer ceiling after builds includes stage/setup/cleanup reservations and
one hour protected for finalization. Feasibility and smoke each get six hours;
native research preparation gets 24 hours inside a 28-hour enclosing command. These are
ceilings, not expected durations. Full learner/generator throughput is unmeasured;
25M completion within six learner hours is not promised.

CPU feasibility uses up to four ordered processes with64 rows in flight. Native
preparation uses up to three reference workers and one existing native client inside
four qualified CPUs, batching up to eight native executions/request. Per-program
100-term agreement remains mandatory. Reference values and coverage share one
execution; exclusions precede native work. GPU work remains serial. Worker failures
and deadlines terminate/reap owned children; no hidden retries or quota reduction.

Each study root retains its64GiB cap and32GiB checkpoint cap. The separate smoke
root retains diagnostic state; image/input/proof projections need additional free
space. Existing measured storage preflight and runtime watchdogs remain authoritative.

## Actual local evidence

The original v1 first shard admitted20,634/32,768 rows, with11 empty families,
43,817 cross-role exclusions and1,085 duplicate sources. Acceptance passed; neither
research learner ran. More time could not repair that frozen candidate pool.

The local reference feasibility check recorded all16 cells FEASIBLE,393,216 selected training examples, every1,024/family
quota met, and30,525,661 target tokens. It examined794,604 total candidates, with
2,280 training rejections. Four-worker CPU time was2,018.96 seconds (33m39s), including
selection/publication of metadata. This is not a hardware/native runtime estimate.

The recorded pass began before the final orchestration amendment. Its complete
source/cohort/shard/token inputs are byte-for-byte equal to the final protocol;
only the smoke gate and campaign ceiling changed. The local review checked exact equality of those data inputs. Detailed per-cell
records remain local; this public guide contains aggregate operational evidence only. Native batching was not executed
by this reference-only pass and has separate serial/batch differential tests.

Final affected regression:446 passed,1 unavailable-native test skipped and1 GPU
test deselected, after the final native-disagreement/worker-exit fixes. The broader
pre-final-review run had509 passes,6 skips and1 GPU deselection. Twelve
additional foundation tests failed at import, all solely because this environment
lacks `wasmtime`; none failed an implementation assertion. Those tests remain
required in the pinned container. The temporary runner/helper suite passed56 tests,
including each failed boundary, missing scoring, nonfinite-proof failure, no fresh
updates, initialization mismatch, archive verification and protected-data redaction.
No host dependency was installed to substitute for the pinned environment.

## Evidence and failure handling

The command prints `<RUN>/report.zip` or `<RUN>/recovery.zip`. Share that archive
on either success or failure and retain the private run tree locally. Recovery
includes completed source-feasibility cells/deficits, native preparation rejection
counts and bounded smoke/role control records, excluding target programs and terms.
The read-only `diagnose` command can export an existing run without new hardware.
Native/reference mismatch, different selected rows, missing proof, unacknowledged
worker exit or missing canonical verification cannot promote the run to success. A native
disagreement ends preparation after the current bounded reference batch, even if
other candidates could fill the quota; later shards are not executed.


## 2026-11-03 preparation timeout and repair

The original recovery archive for `fresh-p1-v2-20261102T162615Z` passes the
canonical archive inventory/hash verifier. Its controller records a successful
source-feasibility stage (1,027.91 seconds), acceptance exit0 and preflight PASS.
The inner preparation process records TimeoutError, returncode -9 and elapsed
21,570.25 seconds; reclamation was acknowledged in0.19 seconds. The outer command's
remaining two hours were host-stage headroom, not additional inner-role work time.

Four protected cells and the first training cell have original PASS ledgers.
No native/reference disagreement appears in those completed ledgers. The archive
has no ledger for the interrupted cell, so its exact progress/cause within native
preparation cannot be reconstructed. No research learner started; science remains
UNMEASURED. This is a timeout, not the earlier coefficient/quota failure.

Code audit found per-candidate compilation of the unchanged BigInt guest plus
repeated reading/hashing of the installed worker executable, despite a persistent
worker and batched transport. T143 retains one installed artifact snapshot and one
compiled shared runtime per native worker generation. Every candidate still has a
fresh candidate module, stores/instances, clocks, fuel and memory limits; all source
and native checks remain mandatory. No result cache, new distribution or budget
increase is introduced. The same audit found the inner study reset its six-hour
clock rather than consuming the role allowance; setup/draw costs now consume that
allowance and publication/exit retain bounded reserves.

T144 writes atomic per-cell diagnostic snapshots on first phase entry and every30
seconds during ongoing scanning, plus selection/terminal snapshots. These expose
counts and elapsed reference/native/selection durations without source/target terms.
A last snapshot may precede interruption by the current bounded batch; it is not a
resume checkpoint or qualification. Recovery includes it even without a cell ledger.

Both images must be rebuilt before the next consolidated command; changed native
and Python sources invalidate old qualification reuse. The data design and72-hour
campaign ceiling remain unchanged. A speedup and completion within six hours are
unmeasured; phase timings from the next run will establish actual native throughput.
Rust reuse/differential tests are part of the existing pinned-container gate and
were not run locally because Cargo and Docker/Podman are unavailable. No HIP or CI
wait was performed in this repair session.

Affected local verification:109 Python tests passed in52.64 seconds. The updated
implementation disposition binding validates all68 existing records. These are
software checks, not a native/HIP performance measurement.


## CPU/native throughput precheck (T145–T146)

Before another full run, execute `fresh_pilot.py throughput` as documented in
`../quickstart.md`. This is CPU-only and produces `report.zip` on PASS, BLOCKED or
failure. The 90-minute diagnostic includes complete protected cohorts, one full
training shard, middle/end stream samples and cold-worker restart consistency.
It uses canonical native batching, preparation, exclusions and payload publication
at64GiB/four CPUs, with three ordered reference workers. The timing gate uses the
slowest training-cell time/draw across all 12 shards with a 3× margin, triples the
protected cost and adds 7200 seconds for unmeasured full-study overhead. The bound
is 86280 seconds, preserving 30-second cleanup, 30-second process exit and 60-second
study reporting reserves inside the 24-hour role envelope. These T147–T148
limits supersede the earlier six-hour/30-minute limits recorded above. The v2
throughput policy and new helper/source identities require a rebuilt CPU image
and a fresh probe. The optional preparation authority `work_seconds` is independent
of the unchanged six-hour learner budget. Legacy authorities keep their defaults.

The full runner now requires `--throughput PATH` pointing to an original PASS
output, no older than72 hours and compatible with its image/source/helpers/engine
facts/operator host/resources. All original metric/process/cleanup hashes are
verified; late image changes are rejected. A failed or partial probe never permits
a full run. The ZIP is a diagnostic export, not reusable qualification or a study.
It contains aggregate metrics and bounded logs; generated terms stay private.

Local verification:108 affected Python tests passed, followed by23 focused tests
including the exact headroom boundary. Coverage includes the canonical
preparation path with an independent reference stand-in, actual CPU-only engine
command construction, timeout/corruption/cleanup recovery and stale/tampered proof
rejection. CLI help/dry-run remain stdlib-only. Native timing and actual target
performance remain NOT_RUN here; the container engine is unavailable. No CI wait
or HIP execution was performed. This gate supports preparation-runtime readiness,
not a statistical confidence percentage or guarantee of learner/evaluation success;
current-source acceptance and the existing same-model paired smoke remain mandatory.


T147–T148 margin amendment validation: 138 affected tests passed. The registered
24-hour preparation authority reaches the isolated role without the old six-hour
clamp; published learner budgets stay unchanged. Boundary tests exercise both
sides of the 3× plus two-hour gate, and all stage reservations plus final reporting
fit within 96 hours. The host CLI runs under `python -S` and all 26 documented Bash
blocks parse. These are software checks; actual CPU/native throughput still awaits
the operator probe. No HIP command or CI wait was performed.


## Prepared-run recovery reviewed 2026-11-03 Pacific (T149–T150)

`recovery(3).zip` passed canonical archive inventory/hash verification. Original
workflow `sha256:3a093092474e604018242d32f939eac24498244e3df6c184eaebe01354937313`
records acceptance exit 0, source feasibility PASS in 1022 seconds, preflight PASS,
and native preparation PASS in 6675 seconds. All 16 cell ledgers passed. The smoke
failed after 2.07 seconds with `ValueError: insufficient storage reservation`;
the final preparation disk monitor reported 65,664,630,784 bytes free (61.2 GiB).
The canonical runtime needs the full 64 GiB reservation still available. No full
learner started, so the scientific status remains UNMEASURED.

T149 also repairs the smoke's fixed-candidate blueprint, which was inconsistent
with its declared refreshed arm. A new canonical assembly/preparation/resolved
runtime test exercises the boundary previously hidden by mocked assembly. The
small diagnostic retains two original training cursors and 2048 draws per cursor,
32 families, original numeric law, protected cohorts and native verification.
The full scientific data, model, seed and exposure remain unchanged.

T150 continues only this verified pre-learning boundary in a disjoint run root.
Original preparation inventories are streamed and hashed, original selection order
is compared to the source-feasibility record, and qualification/feature evidence,
production closure, image pair, host/resources and fresh original throughput proof
are checked. Canonical consumers still re-inspect qualification compatibility.
224 GiB free is required up front; no destructive cleanup is performed. The exact
command is in the quickstart. Successful earlier stages are explicitly reused;
the parent FAIL workflow and original references remain in success/failure exports.
No image rebuild or repeat of full preparation/acceptance is required when all
these checks pass. The known old controller hash is the sole throughput-helper
migration exception; measured code and production source must remain identical.


Local continuation validation: 114 helper tests passed, then 52 focused tests after
the final disk-policy/FEASIBLE-status audit. The 224 GiB reserve covers two 64 GiB
output roots while leaving the canonical 64 GiB free-space requirement plus a
32 GiB margin. The runtime remains stdlib-only on the host; all 27 documented Bash
blocks parse. Production source identity still equals
`sha256:df68fb0a2a70ed6025ea06084e900f0dbaced51037498a8d18bd3367868eb58b`,
the actual reviewed throughput binding. CI implementation binding is unchanged.
No local HIP, image build or CI wait was performed. Actual continuation remains
pending on the operator machine after space is freed.


## Small-smoke token-cap correction (T151–T152)

Recovery(4) passed canonical ZIP verification. Continuation reused feasibility,
acceptance and all large native-prepared cells. Free disk was 715933028352 bytes,
well above the required reservation. Small preflight/native preparation passed;
the static learner exited successfully after two real updates, followed by the
wrapper error `smoke did not consume beyond first shard`. Full learners never
started. This is a diagnostic budget/planning defect, not evidence of HIP failure.

The old 3723-token cap was derived by quota-reducing the large parent's selected
rows. Small native preparation selects from the exact 2048-draw prefix instead;
row identities and program lengths can differ. The repair source-selects those
exact prefixes with canonical exclusions/refresh admission, computes each 16-row
batch charge, and freezes enough tokens for both arms' first three updates below
the existing 16384-token ceiling and available refresh capacity. Native-prepared
source order, encoded targets and per-batch charges must match before model work.
The actual batching/Ledger regression reproduces two updates under an insufficient
parent-derived cap and three under the exact cap. A canonical preparation fixture
also exercises actual PreparedBatches consumption beyond the first shard.

Use a new continuation output with the original 20261103T035534Z prepared parent;
do not resume the failed diagnostic learner or use its weights for research.
Qualified production source, images, scientific model/data/token budgets and
original acceptance stay unchanged. No new local HIP run or CI wait is performed.

Local validation: 17 smoke tests passed with canonical prepared data and actual
PreparedBatches/Ledger exposure; seven focused budget/native-plan counterexamples
passed; then 121 helper regressions passed after wiring the pre-learner guard.
The already-exercised canonical fixture was excluded from that last run. Source
and CI bindings remain unchanged; host dry-run and documented Bash syntax pass.
