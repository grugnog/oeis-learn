# Measurement-only performance screen — T094

Status: current-source native/HIP measurements are reviewed in
[phase-5-generation.md](phase-5-generation.md#reviewed-current-source-hip-run--2026-10-21)
and [the immutable receipt](phase-5-batched-hip-review.json). All12 frozen-state
mixed-prompt generation groups pass parity, unchanged-weight and deadline checks.
Complete-screen cost admission fails at47,749.33s versus5,760s. Fifteen shared
small-value slowdown cases range from23.87x to78.28x and still require
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

