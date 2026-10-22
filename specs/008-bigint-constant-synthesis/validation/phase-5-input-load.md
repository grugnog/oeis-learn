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

- [X] I001 Add native/Python counterexamples for nested accounting, early errors,
  byte/value counters, malformed evidence and unchanged result/fuel parity.
- [X] I002 Add opt-in nested measurements in the native runtime/example, leaving
  ordinary and existing aggregate-only profiling available.
- [X] I003 Validate/summarize nested evidence, wire the portable helper/shared
  CI flag and retain mismatch/partial reports.
- [X] I004 Run available checks, record source-bound local evidence and provide
  CPU commands, leaving actual new-image native results pending without CI waits.

T094 remains open until actual subphase costs and observer distortion are
reviewed. This step does not promise a further optimization or admit training.
T093/T097/T098 and the original negative T096 remain unchanged.

## Implemented observation and available checks

The optional native `evaluate_input_profiled` wrapper uses the same production
run/load helpers. Existing aggregate-only observation and ordinary execution do
not read nested clocks. The seven fixed subphases include first-use export
resolution in their respective guest-call phase; finish-load also includes sign
selection in its original position. Magnitude encoding remains after begin-load,
as before. Literal source parsing is outside the input-load parent. Byte-loop
counters are separate from the one timer around the loop; value completion means
its finish/status checks returned successfully, not candidate scoring completion.

Native regression source covers exact guest call/fuel parity, successful/partial
byte counts and empty nested profiles after early guest initialization failure.
The packaged corpus now has five mode combinations, including nested profiling
with/without the ordinary lookup baseline. Missing native artifacts fail in CI.
Python evidence checks reject malformed policies, missing/extra phases, NaN or
negative timings, double counting, impossible phase/value/byte counts and wrong
scope. A bounded raw `failed-native.log` survives malformed native JSON/profile
validation; the valid failed summary identifies the active case.

Available checks pass103 Python tests, with five explicit packaged-native
contract skips. Ruff/formatting, workflow YAML/Bash/embedded Python, mocked
command routing, standard-library-only host help and documented shell commands
are checked in [the local receipt](phase-5-input-load-local.json). There is no
local Rust/container toolchain: native compilation/regressions and actual nested
costs/observer distortion remain pending. No native or HIP result is fabricated,
and new CI is left running without polling or waiting.

Use the [current quickstart command](../quickstart.md#current-next-step-cpu-input-load-attribution).
I001–I004 are complete as implementation/local-check tasks. T094 is open until
the new results are reviewed; full Phase5 promotion is still blocked.
