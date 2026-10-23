# Measurement-only performance screen — T094

Latest2026-10-23: [the corrected HIP pipeline profile](phase-5-pipeline-profile-rerun-review.json)
is reviewed; W005 is complete and T094 remains open. All280 archived file hashes,
five actual-GPU trace exports and three role audits verify. The partial observed
portfolio spends162.63s decoding (50.51%) and148.08s on native requests (45.99%).
Native spans include solver/compilation/checks/IPC, not pure WASM. The average
aggregate CPU use is0.98 of16 allowed core-equivalents; sampled GPU busy is31.44%
during decoding and5.26% during native work. Ordinary500-update/256-validation-row
measurements take32.99s/1.33s and separate CPU scoring18.37s. This diagnostic mix
does not establish the whole-training fractions or a complete cost projection.

The later-token probe records7168 small GPU kernel events and224 `nonzero`
operations over eight steps;140.32ms host wall contains18.06ms kernel union.
Repeated boolean indexing in cached decoding is a concrete candidate for a
focused fixed-input HIP comparison before scheduler changes. Trace perturbation
and duplicate-flow warnings preclude a baseline or per-op causal speedup claim.
A later bounded native-worker/overlap pilot is plausible but requires explicit
ordering/deadline/memory design. See the
[full review and proposed next step](phase-5-pipeline-profile-plan.md#reviewed-corrected-hip-collection--2026-10-23).
No production optimization, training admission or T096 reassessment is made.

The following receipts retain their original source and historical scope.

Status: current-source native/HIP measurements are reviewed in
[the scoped HIP receipt](phase-5-scoped-hip-review.json); the earlier
[batching review](phase-5-batched-hip-review.json) remains historical evidence. All12 frozen-state
mixed-prompt generation groups pass parity, unchanged-weight and deadline checks.
Diverse32 cost admission passes at7,926.648s versus11,520s. Complete-screen cost
admission fails independently at29,637.220s versus5,760s. Fifteen current shared
small-value slowdown cases range from23.24x to73.81x and still require
investigation, so T094 remains open. Local CPU tests alone establish no
accelerator throughput or utilization result.

`scripts/benchmark_bigint.py` calls production APIs. Native observational timings
split parse/lower/candidate compilation from fresh-store execution, literal/input
loading and serialization; runtime-module compilation and subprocess overhead
are separate. Public production records/limits remain unchanged. The observation
wrapper has a native parity test; both observation and ordinary execution use the
same internal run method. No dependency or arithmetic/profile upgrade is made.

The corpus has shared old/new add/sub cases (including64/128/255-bit literals),
supplied64/256/512/1024/4096-bit signed values, full multiplication, cancellation,
loop/branch workloads, a16384-bit generated result and expected overflow. No
16384-bit input/literal is loaded. Batches1/20/100 use repeated exact inputs as
microbenchmarks, not dataset denominators. Each result is checked against direct
Python expectations and the independent source interpreter. Classified fuel,
deadline or memory exclusions remain explicit and cannot count as successful
arithmetic throughput. Cold and repeated warm samples retain exact phase times;
warm reuse covers the runtime module, not candidate/value-result caching.

The optional HIP neural screen measures restored batch8 updates, full teacher forcing,
production cached generation and the complete durable checkpoint store. Capacity
fixtures reach4096-bit inputs,4097-bit differences and1024-token targets. A separate
non-scoring tensor fixture matches the prepared dataset's maximum input/target
shape for cost projection. Measurements are explicitly separate from training
accuracy or learnability evidence. Native fit and reference p95 costs, Linux
reaped-child peak RSS, per-request fuel, real parameter/device counts and source/
image/compiler/lock/binary hashes are retained. Three repetitions are a bounded
engineering estimate, not a population confidence claim.

Protocol `fixed-state-batched/v2` separates fixed calibration from repeated
timings, checks frozen parameter hashes and compares singleton/batched seeded
generation under mixed prompts and all conditioning controls. Both the short-
and long-output synthetic calibration states are included. Generation p95 is
the worst per-state/control/temperature p95. See the linked follow-up for exact
protocol, batching limits, cost accounting and the targeted actual-HIP command.

Shared small-value median execution slowdown above5x appears in
`slowdown_investigations`. Review its cold compilation, prepare/execute and batch
scaling before long runs. Such a measurement cannot justify weaker semantics;
algorithm changes return to Phase 2. See [phase-5.md](phase-5.md) for commands.

## Scope follow-up and sequencing — 2026-10-21

PR #59 changes admission accounting and execution panels, not the arithmetic
runtime or measured generation algorithm. Both new cost scopes retain every
registered control. The historical p95s motivate the finite diverse32 resource
revision in [screen-preflight.md](screen-preflight.md); a fresh source/image HIP
receipt must confirm it before the one bounded diagnostic.

The reviewed 15 small-value ratios are execution ratios, already excluding
candidate preparation and module compilation. New execution includes fresh
stores, bigint arena/import calls, input loading and serialization, whereas the
retained subset uses the old scalar representation. Those differences identify
profiling targets; the ratios alone cannot attribute the slowdown to any one
component. The targeted preflight retains cold/warm preparation/execution and
batch1/20/100 records for that investigation. Per the merged scope decision,
broad runtime optimization is deferred until after the single diverse32 result;
the unresolved slowdown remains T094 evidence and prevents closing T098.

## Current CPU evidence and bounded next profiling step — 2026-10-21

Reviewed the75-case CPU report from completed workflow39686633400, source
`sha256:942afe05c9cbe3ba95f839ed071ca7a44855b1222d2e0b03d82f0c39eeeef562`.
Archive CRC/SHA256, canonical report hash and shared source/native/image bindings
verify in [the execution review](phase-5-execution-review.json). The15 shared
small-value ratios are26.09–100.91x on this CI CPU; this is separate evidence
from the earlier23.24–73.81x HIP-host ratios, not a comparable regression trend.

Warm shared-small-add execution illustrates the scaling:

| Batch | Bigint execution | Retained execution | Bigint prepare | Bigint per term |
| --- | --- | --- | --- | --- |
| 1 | 0.717ms | 0.0257ms | 0.550ms | 0.717ms |
| 20 | 13.936ms | 0.1553ms | 0.893ms | 0.697ms |
| 100 | 68.831ms | 0.6925ms | 0.967ms | 0.688ms |

The255-bit shared case takes68.893ms at batch100, close to the small case's
68.831ms. Runtime compilation is separately0.176–0.181s; it is excluded from
these execution ratios. Preparation at batch100 is about1.4% of bigint
execution time. Thus compilation alone cannot explain the measured execution
gap, and batching does not amortize the dominant new cost: it remains nearly
constant per term. This is an inference from the measured scaling, not an
attribution to a particular function.

Source inspection identifies per-term operations worth measuring: fresh store
and deadline-thread lifetime; guest instance/init; literal/index byte transport;
linker/function registrations and candidate instantiation; compute; output
serialization and teardown. Existing prepare/execute totals cannot separate
them, so T094 stays open. In particular, these data do not justify blaming
bigint arithmetic or changing safety/resource semantics.

The next bounded engineering step is observation-only native profiling:

1. Add opt-in qualification measurements around those phases in the existing
   production run implementation. Ordinary execution remains the same path with
   observation disabled. Preserve fresh stores, epoch deadlines, fuel, guards,
   imports, numeric limits, output validation and failed-value suppression.
2. Use the existing non-scoring corpus and unchanged three-sample schedule
   (one cold, two warm), batch1/20/100 and retained controls. Include legal wide
   values and expected overflow/resource exclusions. Bound the CPU role to ten
   minutes, with separate existing cleanup allowance and a fresh result archive;
   no training, checkpoint, hidden truth or HIP mount is needed.
3. Verify measured/unmeasured output, failure, fuel and coverage parity; report
   profiler overhead separately and retain unattributed time rather than forcing
   phase sums. A measurement with meaningful observer distortion cannot be used
   as precise causal attribution. Record source/binary/image and compiler/lock
   identities and inspect/cleanup evidence.
4. Rank observed phase costs before proposing a production optimization. Any
   change to runtime or deadline behavior returns to Phase2 with independent
   counterexamples and a frozen comparison. Expanded language/solver support,
   selection or budget scheduling is outside this profiling step.

No profiling run or causal performance fix is claimed by this review. It narrows
the next implementation target using already collected evidence. The successful
deadline replay does not close T094 or authorize another learning attempt.


## Opt-in phase profiler implementation

Freeze observation policy `bigint-runtime-profile/v1`: use the existing corpus,
batches1/20/100 and three paired repetitions (first profiled-first, then alternate
order). Compare ordinary aggregate measurement with opt-in detailed observation
on the same runtime implementation. Preserve every production resource/semantic
check and independently validate both outputs. Record phase call counts and
seconds, attempted/finished terms, unattributed execution time, paired parity
and observer overhead; do not claim causal attribution before reviewing them.
The dedicated CPU role has600seconds work plus the existing separate cleanup
allowance, retains partial/failure reports and produces a bounded archive. It
receives no dataset, weights, hidden scoring or GPU capability.

- [X] P001 Add native and Python regressions for phase accounting, early failure,
  measured/control semantic/fuel/coverage parity and malformed profile evidence.
- [X] P002 Add opt-in native timings around the existing execution path, leaving
  ordinary aggregate measurement available as the paired control.
- [X] P003 Summarize phase costs, order and observer overhead with explicit
  unattributed time; retain failures and report resource exclusions honestly.
- [X] P004 Wire an archived CPU-only host/CI profiling role with600seconds work,
  source/image/binary binding and actual inspect/cleanup records.
- [X] P005 Run available checks, push incrementally and provide CPU repro commands;
  leave actual native profiling evidence pending instead of awaiting new CI.

T094 remains open until actual phase/observer-overhead results are reviewed.
A production optimization is outside this measurement-only implementation.

Implementation is pushed in PR59. Native stages are `literal_decode`,
`store_setup`, `clock_start`, `guest_init`, `input_load`, `link_instantiate`,
`compute`, `serialize`, `clock_stop` and `store_drop`; failed phases retain
calls/time and normal failure suppression. The paired report independently
compares every Evaluation field and hashes retained values after comparison.
Ordinary controls have no detailed clocks; retained slowdown uses their warm
execution time, so detailed observation cost is reported separately. Warm phase
medians provide a descriptive ranking, not a production optimization decision.

The dedicated `--runtime-profile-only` helper and CI partition use the existing
shared image and actual audited launcher. See the [command](phase-5.md#bounded-cpu-runtime-profiling)
and [local validation receipt](phase-5-runtime-profile-local.json). P001–P005
record implementation/checks, not actual native measurement qualification.
Native compilation and the75-row corpus remain pending new CI or workstation
execution. T094 is deliberately unchecked until those results are reviewed.

## Reviewed workstation CPU profile — 2026-10-22

The uploaded `008-runtime-profile-20261022T112858Z.zip` is reviewed in
[the source-bound receipt](phase-5-runtime-profile-review.json): all archive
hashes, report identity and225 profiled/control result pairs verify across75
rows. The recorded source is `sha256:ca83192e992a165e88dc139eea7539bc78491c0640caf1878f9bce108eedd533`.
Offline/read-only/output-only inspection passes and cleanup is0.040418s against5s.

Shared-small batch100 ordinary execution is0.219898/0.220840ms per term for
add/cancel, with retained slowdowns72.529x/71.045x. Input loading is74.41%/74.32%
of detailed execution; computation is0.217%/0.220%. Detailed/control execution
ratios are0.993875/0.991463 for these rows. Other single-term samples have larger
variance (overall ratio range0.675–1.410), so this is descriptive attribution
from two warm samples, not a precise causal cost model. The generated16384-bit
batch100 hits aggregate fuel at82/100 on all three repetitions and is excluded
from successful full-batch throughput. Expected overflow stays classified.

These data identify input loading as the first target, not its internal cause.
Source inspection shows repeated typed host-export resolution during each
transfer and status check. The focused cache and independent ordinary baseline
are frozen in [phase-2-host-lookups.md](phase-2-host-lookups.md). New CI and
workstation comparison must establish its result; no speedup or T094 completion
is inferred from source inspection alone.

### Run the focused CPU comparison

```bash
lookup_out="tmp-validation/008-host-lookups-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/qualify_bigint_phase5.py --engine podman \
  --runtime-profile-only --compare-host-lookups --output "$lookup_out"
```

Share `$lookup_out.zip`. A fresh current-checkout image is built/reused; explicit
`--image` must match the checkout. Build time is outside600s work plus5s cleanup.
Use Docker with `--engine docker`. Each row contains the ordinary current call,
the detailed current call and an ordinary lookup-per-call baseline; baseline
order alternates around the existing pair. First samples remain separate from
the two warm samples; recorded order describes shared process warmup. Full
Evaluation parity is mandatory across all three, and retained small-value ratios
still use ordinary current execution. `host_lookups.cached_to_lookup_execution_ratio`
compares ordinary execution, while `profiling.warm_median_execution_ratio`
continues to measure detailed/ordinary observer distortion. Neither ratio turns
a resource exclusion into a successful evaluation.

The comparison uses CPU Rust/Wasmtime execution, which also runs inside HIP-backed
learning. HIP is needed for neural updates/decoding or end-to-end throughput,
not this candidate-executor microbenchmark. No data, weights, fitting, hidden
scoring, GPU capability or learning retry is requested. Original T096 remains
negative and promotion remains blocked.

## Host-lookup result and finer input-load observation — 2026-10-22

The [reviewed lookup archive](phase-5-host-lookups-review.json) verifies75 rows,
225 three-way groups, exact output hashes, report/source bindings and actual
isolation/cleanup (0.047662s against5s). At batch100 small-add/cancel use
21.838/22.269ms with lookup-per-call versus21.782/21.784ms cached:0.25%/2.18%
less time. The retained gap remains about71–73x, and input loading remains74%.
The4096-bit multiplication row uses44.775ms versus29.350ms (34.45% less);
the successful generated16384-bit batch20 uses38.67% less time. Batch100 of
that latter case still exhausts aggregate fuel at82 terms and remains excluded.
A batch100 loop row is5.25% slower with caching, whereas its batch20 row is9.63%
faster; two warm samples cannot establish a stable small difference. No broad
throughput superiority or new end-to-end HIP claim follows.

The next bounded step is `bigint-input-load-profile/v1`, frozen in
[phase-5-input-load.md](phase-5-input-load.md). It measures index parsing,
begin-load, magnitude encoding, the entire byte-transfer loop, finish-load,
status checks and sealing inside the existing input-load phase. It records
attempted/returned byte calls and started/finished values, including early
failure; no byte gets its own clock. Nested sums/residual must match the parent
and do not contribute again to aggregate execution. The current ordinary pair
still measures observer distortion; the old lookup strategy remains an optional
third ordinary baseline.

Use the [current CPU command](../quickstart.md#current-next-step-cpu-input-load-attribution).
The same corpus, resource ceilings, source/image binding and failure archives
apply; malformed native detail also retains a bounded raw log. Actual new-image
native regression and subphase measurements are pending, so T094 stays open.

## Reviewed production policy CPU qualification — 2026-10-22

[The corrected-source adoption review](phase-5-engine-adoption-review.json)
passes334 arithmetic,37 fitting and64 containment tests with no skips. All27
archive entries and exact source/image/policy bindings verify. Every one of75
comparison summaries recomputes,450 detailed/ordinary pairs and225 cross-setting
groups agree, and independently expected value hashes match. Per arm,213 samples
succeed;3 aggregate-fuel and9 integer-limit samples remain classified exclusions.
Podman roles are offline/read-only/output-only, with cleanup below0.040s.

Production64KiB ordinary batch100 small add/cancel takes4.165/4.220ms versus
22.694/21.880ms at zero reservation (5.45/5.18x). Full4096-bit multiplication
improves29.606 to11.931ms (2.48x). Corpus maximum per-process VmHWM is19.25MiB
zero versus18.15MiB production, including compilation/JIT and observations;
this is not aggregate or guest-only resident memory.

The remaining small add/cancel retained execution gap is13.29/13.40x. Three
batch1 rows are slower, up to2.05x; two warm observations do not resolve latency
uncertainty. Timer start/stop now accounts for roughly half of profiled small
add/cancel execution. Runtime code creates and joins a deadline thread per
term. This supports investigating timer lifecycle next, not claiming that a
replacement is safe or yields a particular speedup. Preserve per-term/aggregate
deadlines, fuel, fresh stores, cleanup and concurrent-engine isolation.

T094 stays open. This supplies CPU adoption evidence, not new Docker/HIP,
neural throughput or whole-workload cost admission. Original negative T096,
its consumed attempt and the promotion block remain unchanged.
