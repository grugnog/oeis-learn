# Measurement-only performance screen — T094

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

