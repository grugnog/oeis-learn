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
  loading ends. Record allowed/denied requests and allocation-failure callbacks.
  Allowed requests are not labelled successful physical allocation or copy time.
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

- [ ] B001 Add native/Python tests for first versus later calls, real guest growth,
  traps/denials, unchanged fuel/results, bounded traces and malformed evidence.
- [ ] B002 Add optional native limiter/first-call telemetry on the existing path.
- [ ] B003 Validate and summarize v2 while preserving v1 readers and archived failures.
- [ ] B004 Run available checks, push incrementally and provide CPU repro commands;
  leave new-image native regression and hypothesis confirmation pending.

T094 remains open. No performance fix or first-allocation cause is claimed.
The negative T096, remaining checkpoint/operations gates and promotion block stay.
