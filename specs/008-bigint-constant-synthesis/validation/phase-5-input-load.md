# Input-load attribution follow-up — 2026-10-22

The [reviewed host-lookup comparison](phase-5-host-lookups-review.json) verifies
225 three-way groups and independent exact output hashes across75 configurations.
At batch100 cached small-add/cancel differ by0.25%/2.18% from lookup-per-call;
4096-bit full multiplication uses34.45% less time. Generated16384-bit batch20
uses38.67% less time; batch100 remains an aggregate-fuel exclusion at82 terms.
These bounded two-warm-sample observations are not end-to-end HIP throughput.

Input loading remains about74% of small-case detailed execution. Repeated
lookups explain little of that fixed overhead. Freeze the next step as
observation-only `bigint-input-load-profile/v1`, on the existing run path:

- Keep the ten parent phases and optional ordinary lookup baseline unchanged.
- Opt in to nested timing for index parsing, begin-load, magnitude encoding,
  the entire byte-transfer loop, finish-load, status checks and sealing.
- Do not clock each byte. Record literal/index values started, values finished,
  byte calls attempted/returned and per-subphase calls/time. Returned calls do
  not imply accepted bytes after a sticky guest error.
- Attribute nested times only within `input_load`; retain its residual and do
  not add nested values again to total execution. Literal source parsing stays
  in the existing outer `literal_decode`, outside this nested measurement.
- Preserve every guest call, argument, failure position, fuel, deadline, fresh
  state, cached-handle policy, numeric/resource identity and cleanup. No guest
  ABI, arithmetic, solver/model/data/selection or resource-policy change.
- Continue pairing the detailed run with an ordinary run to measure observer
  distortion. The optional third ordinary lookup baseline remains independent.
- Use the same25-case corpus, batches1/20/100, three repetitions,600s CPU work
  plus5s cleanup and fresh bounded archives. No inputs, weights or HIP mount.

- [ ] I001 Add native/Python counterexamples for nested accounting, early errors,
  byte/value counters, malformed evidence and unchanged result/fuel parity.
- [ ] I002 Add opt-in nested measurements in the native runtime/example, leaving
  ordinary and existing aggregate-only profiling available.
- [ ] I003 Validate/summarize nested evidence, wire the portable helper/shared
  CI flag and retain mismatch/partial reports.
- [ ] I004 Run available checks, record source-bound local evidence and provide
  CPU commands, leaving actual new-image native results pending without CI waits.

T094 remains open until actual subphase costs and observer distortion are
reviewed. This step does not promise a further optimization or admit training.
T093/T097/T098 and the original negative T096 remain unchanged.
