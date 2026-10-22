# First-byte and memory-growth observation — 2026-10-22

The [reviewed nested input profile](phase-5-input-load-review.json) verifies75
configurations/225 pairs, independent output hashes and exact parent/nested
accounting. Small batch100 byte transfer takes about158.9us per term for three
calls, versus175.5us for512 calls in4096-bit multiplication. It accounts for
98.3% of small-case input loading, about73% of execution. This suggests fixed
first-use work. First allocation/memory growth is a hypothesis, not a finding.

Freeze observation-only `bigint-input-load-profile/v2`, retaining the existing
seven subphases, counters and ordinary control. Add `bigint-first-byte-probe/v1`:

- Time just the first actual byte call per fresh term, including traps. No other
  byte gets its own clock. Remaining transfer time is parent minus first-call
  time; it includes later calls, loop and observation overhead, not pure memcpy.
- Retain first-call attempted/returned counts and partial transfer evidence.
- Observe the existing ResourceLimiter callbacks without changing decisions.
  Partition requests before the first byte, during it and after it until input
  loading ends. Record allowed/denied requests and `memory_grow_failed` callbacks.
  Allowed requests are not labelled successful physical allocation or copy time.
  Failure callbacks are counted independently, without assuming a paired request
  or that the operating system failed to allocate memory.
- Bound raw growth-event samples to four per store and16 per evaluation, with
  exact omission counts. Record current/desired bytes, request order, term and
  phase; count all requests even when their detailed samples are omitted.
- Keep v1 evidence readable. Current native v2 reports require the new record;
  old v1 records cannot supply missing first-byte attribution.
- Preserve all guest calls/arguments, fuel, memory limits/reservations, fresh
  stores, deadlines, failure mapping and cleanup. Ordinary execution gets no
  probe counters/clocks. No allocator, memory policy, guest ABI or solver change.
- Reuse `--runtime-profile-only --profile-input-load`, the same25-case corpus,
  batches1/20/100, three repetitions,600s CPU work plus5s cleanup and bounded
  result archives. HIP, data, weights, hidden scoring and training are excluded.

- [X] B001 Add native/Python tests for first versus later calls, real guest growth,
  traps/denials, unchanged fuel/results, bounded traces and malformed evidence.
- [X] B002 Add optional native limiter/first-call telemetry on the existing path.
- [X] B003 Validate and summarize v2 while preserving v1 readers and archived failures.
- [X] B004 Run available checks, push incrementally and provide CPU repro commands;
  leave new-image native regression and hypothesis confirmation pending.

T094 remains open. No performance fix or first-allocation cause is claimed.
The negative T096, remaining checkpoint/operations gates and promotion block stay.


Implementation: the current input-detail path emits v2 and carries optional
limiter counters plus a fixed four-request buffer per store. One timer wraps
only the first lazy host adapter dispatch/guest byte call per fresh term.
Snapshots and counters remain outside that timer. Finalizing input observation
on both success and error preserves first-call traps and partial receipts.
Initial memory creation is counted; compute/serialization growth after input
completion is excluded. No new flag, dependency, image layer or runtime resource
identity is introduced. The Python reader retains v1 without inventing fields,
rejects mixed policies and requires v2 for current native collection. Invalid
raw native output still reaches the existing bounded failure archive.

B001–B004 mark implementation and available checks only, not actual execution
of the new native source or confirmation of the hypothesis. New-image native
regressions, first-call attribution and observer distortion require CI/the CPU
command in [quickstart](../quickstart.md). The local receipt is
[phase-5-first-byte-local.json](phase-5-first-byte-local.json). T094 stays open.


## Reviewed user CPU result — 2026-10-22

[phase-5-first-byte-review.json](phase-5-first-byte-review.json) verifies archive
CRC, all seven manifest hashes, current source/report binding, independent
expected value hashes and all225 paired summaries. The benchmark executes the
new source successfully; a standalone native test-suite receipt is not included.
Every one of8670 observed first calls has one allowed64KiB growth request; no
request is denied and no failure callback fires. The sampled transition is
17 to18 pages. No later growth occurs during input loading. Failed-term probes
remain counted without treating their results as successful throughput.

Warm batch100 small-add/cancel first calls take160.93/161.50us per term,
99.85% of byte-transfer time and about73% of detailed execution. Remaining
transfer costs about0.24us per term.4096-bit multiplication spends157.92us in
its first call and13.98us in the remaining loop. Observer ratios on these rows
are1.0049/1.0079/1.0049. Batch1 varies materially; two warm samples are descriptive.

The guest starts its transfer Vec empty, then allocates on the first push.
The host forces dynamic memories with zero guard/reserved growth space. Pinned
[Wasmtime20.0.2](https://github.com/bytecodealliance/wasmtime/blob/v20.0.2/crates/runtime/src/memory.rs)
allocates a new mapping and copies prior accessible memory in `grow_to` when
existing reservation is exceeded. Relocation/copy is therefore a strong
explanation consistent with the evidence, not directly timed or causally proven.

The next focused work is a separately identified qualification-only comparison
of zero versus a small bounded growth reservation. Preserve guest bytes/calls,
fresh state, fuel and visible memory limits; include ordinary execution,
first-byte timing, total lifecycle, RSS and cleanup. Verify exact outputs and
failure semantics, and bind configuration/source/image identities. Do not
change production memory policy before reviewing the comparison. This stays
CPU-only. No new learning attempt or wider synthesis spec is warranted by this
result. T094 and promotion remain open/blocked; original negative T096 is intact.
