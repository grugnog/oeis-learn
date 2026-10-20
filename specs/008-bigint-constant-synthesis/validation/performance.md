# Measurement-only performance screen — T094

Status: harness implemented; actual native/throughput measurements pending CI,
and actual neural measurements pending the current-image HIP run. No speed,
utilization or resource-coverage superiority is claimed from local unit tests.

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

The optional HIP neural screen measures batch8 updates, full teacher forcing,
production cached generation and the complete durable checkpoint store. Capacity
fixtures reach4096-bit inputs,4097-bit differences and1024-token targets. A separate
non-scoring tensor fixture matches the prepared dataset's maximum input/target
shape for cost projection. Measurements are explicitly separate from training
accuracy or learnability evidence. Native fit and reference p95 costs, Linux
reaped-child peak RSS, per-request fuel, real parameter/device counts and source/
image/compiler/lock/binary hashes are retained. Three repetitions are a bounded
engineering estimate, not a population confidence claim.

Shared small-value median execution slowdown above5x appears in
`slowdown_investigations`. Review its cold compilation, prepare/execute and batch
scaling before long runs. Such a measurement cannot justify weaker semantics;
algorithm changes return to Phase 2. See [phase-5.md](phase-5.md) for commands.
