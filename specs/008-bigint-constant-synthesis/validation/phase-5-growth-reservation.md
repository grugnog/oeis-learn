# Bounded memory-growth comparison — 2026-10-22

T094 follow-up to [the reviewed first-byte evidence](phase-5-first-byte-review.json).
Every observed first byte requests64KiB growth; the host reserves no spare
capacity. Pinned Wasmtime20.0.2 relocates/copies a dynamic memory when its mapping
cannot accommodate growth. This is a strong mechanism hypothesis, not a measured
copy time or evidence for a production change.

Freeze `bigint-growth-reservation-comparison/v1` before execution:

- Exactly two settings: production zero spare growth bytes and a qualification-only
  65,536-byte reservation. Retain dynamic memory, zero guards, pinned engine/guest,
  guest calls, fresh stores, memory/arena caps, fuel, deadlines and failure mapping.
  No configurable reservation size or production worker/config option is added.
- Run each setting in a separate native process with the existing detailed versus
  ordinary pair, three repetitions, retaining the first/cold sample separately.
  Alternate zero-first/reserved-first across the fixed25-case by1/20/100 corpus.
  This yields450 pairs, not independent training/data examples.
- Bind both explicit engine configuration manifests/hashes, identical guest hash,
  semantic resource identity, source/image and measurement binary. Control engine
  configuration has a distinct qualification purpose; it cannot qualify production.
- Compare exact result/failure/fuel/coverage against ordinary controls and across
  settings. Keep all exclusions, including aggregate-fuel and expected overflow.
  Preserve bounded failed raw native output and partial/final archives.
- Report cold/warm prepare/execute/total and runtime compilation separately for
  both settings, first-byte/nested phases, observer distortion and per-process
  Linux VmHWM. Peak RSS includes JIT/compilation and both observations; it is not
  aggregate RSS or a pure guest allocation measurement. The role's memory cap and
  5s cleanup bound remain enforced. Extra virtual reservation is at most64KiB per
  mapping beyond its visible size; it does not authorize extra guest linear memory.
- Use new `--compare-memory-growth` only with CPU `--runtime-profile-only` and
  `--profile-input-load`. Reject simultaneous host-lookup comparison to keep the
  experiment bounded. Keep the existing600s work cap, offline/read-only role and
  output-only mount. No data, weights, HIP, fitting, training or hidden scoring.
- Shared runtime-profile CI uses this comparison through the existing built image;
  ordinary host-lookup coverage remains in native/contract tests. No new build
  layer, dependency lock or independent image build is needed.

- [X] G001 Add failing evidence/scope tests and native cap/fuel regression source.
- [X] G002 Add the fixed qualification constructor and explicit engine/RSS records.
- [X] G003 Add bounded separate-process comparison, validation and host/CI routing.
- [X] G004 Verify available checks, publish incremental commits and CPU commands.

Acceptance for this implementation is trustworthy comparison evidence, not an
assumed speedup. Production adoption requires review of exact semantic parity,
ordinary execution/total/lifecycle time, memory/cleanup and observer cost.
No inference of HIP or end-to-end throughput follows from this CPU comparison.
T094 remains open; original negative T096 and the promotion block remain intact.


Implementation and available-check status: G001–G004 are complete for code and
metadata validation only. The production constructor still selects zero; the
only alternate constructor fixes64KiB and is selected solely by the bounded
native benchmark request. Python rejects changed/unbound manifests, mismatched
results/fuel/coverage/failures, missing v2 probes and invalid per-process RSS.
Each setting is independently checked against the exact arithmetic reference.
The raw report is capped at4MiB; the active-setting journal preserves progress
on later failure and the existing archive cap/omissions remain enforced.

The new native unit source checks unchanged initial visible memory, equal fuel,
accepted growth and identical limiter denial with both reservations. Packaged
contract modes cover both setting orders, and the runtime-profile CI partition
covers all batches including aggregate-fuel exhaustion. No new-source native
execution, measured speedup or production adoption is claimed by test-code
presence. See [phase-5-growth-reservation-local.json](phase-5-growth-reservation-local.json)
for available checks and [quickstart](../quickstart.md) for the CPU command.


## Reviewed user CPU result — 2026-10-22

[phase-5-growth-reservation-review.json](phase-5-growth-reservation-review.json)
verifies CRC and all nine manifest hashes, source/report binding, engine and
comparison identities, independently expected output hashes,450 detailed/ordinary
pairs and225 cross-setting groups. All summaries recompute exactly. Both settings
have identical results, failure classification, fuel, coverage and semantic
resources. The benchmark executes successfully; a separate native regression-suite
receipt is not included. Initial visible memory/growth requests are unchanged.

Warm batch100 ordinary execution:

| Case | Zero |64KiB | Speedup |
| --- | ---: | ---: | ---: |
| Small addition |21.94ms |4.16ms |5.27x |
| Small cancellation |22.38ms |4.28ms |5.23x |
|4096-bit multiplication |29.69ms |12.04ms |2.46x |

First-byte cost falls from about160us to2.7–3.1us per term on these rows.
Small-case total time also improves4.85–4.87x; initialization does not absorb the
savings. Store destruction falls from about2.1ms to0.52ms per100 terms. This
additional lifecycle saving explains why execution improves beyond the earlier
first-call-only estimate. Per-process maximum RSS across the corpus is19.25MiB
versus18.31MiB; this is not an aggregate/guest-only memory measurement and some
individual reserved rows have slightly higher RSS. Cleanup is0.03994s against5s.
Isolation remains offline/read-only with no GPU and only the output bind writable.

These are strong controlled results for a focused production-engine change, not
broad latency or HIP evidence. Three batch1 rows are slower, up to2.16x, with
substantial observer/order variation; two warm samples do not settle single-term
behavior. Reserved small-value execution still takes13.4–13.7x the retained backend.
The generated16384 batch100 fuel exclusion remains at82 completed terms.

Recommend a focused64KiB production policy with explicit engine identity, retained
zero control and native/container regression qualification. No production policy
is changed by this review. T094 remains open for the remaining slowdown/measurement
work; original negative T096 and promotion block are unchanged.
