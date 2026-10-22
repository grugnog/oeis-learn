# Focused host function lookup follow-up — 2026-10-22

The [reviewed CPU profile](phase-5-runtime-profile-review.json) identifies input
loading as about74% of shared-small batch100 execution; ordinary cost is about
0.22ms/term,71–73x the retained executor. Profile/control parity holds for225
pairs across75 rows. Observer ratios for these two rows differ by less than1%;
single-term outliers are noisy. The generated16384-bit batch100 correctly
exhausts aggregate fuel after82 terms. No cost or coverage claim hides that
exclusion. CPU program execution also runs during HIP-backed learning; this
receipt does not measure neural work or complete training throughput.

## Frozen bounded change and comparison

Reopen only the Phase2 host runtime adapter implementation for an exact-call
optimization: cache typed host export handles lazily within one fresh term.
Keep first-use resolution order, every guest call and argument, status check,
WASM bytes, fuel, numeric/resource identity, deadlines, stores/instances and
cleanup. Do not cache values or reuse state between terms. Do not add a new
candidate import, guest bulk API, native arithmetic or resource escalation.
Source/native binary identity changes normally; historical receipts stay bound
to their original sources. No performance improvement is assumed.

The qualification-only comparison retains the earlier lookup-per-call behavior
on the same run method. Each sample keeps the existing profiled/ordinary pair
and optionally adds an ordinary lookup-per-call baseline. Baseline order
alternates before/after that pair. Full values, outcome/reason, fuel, coverage
and resource identity must match across all three. Ordinary optimized versus
ordinary baseline timings measure the proposed change; existing profile/control
ratios separately describe observation distortion. Same25-case corpus,
batches1/20/100, three repetitions,600s CPU work plus5s cleanup, bounded archives
and fresh output identities. No data, checkpoint, fitting, neural work or HIP
mount. A limit discrepancy is a failed comparison, not a speedup.

- [ ] L001 Add counterexamples for lazy missing/mistyped exports, sticky failures,
  exact guest call/fuel parity, fresh-term handles, and baseline evidence rejection.
- [ ] L002 Cache the bounded host function handles per term in
  `crates/oeis_synthesis_core/src/runtime.rs`; retain a qualification-only old path.
- [ ] L003 Add the opt-in three-way CPU comparison and validation to the native
  example, benchmark and portable archived host helper; include it in shared CI.
- [ ] L004 Run available checks, document commands, push commits and leave actual
  new-image performance/native qualification pending rather than awaiting CI.

T094 remains open until actual new-image comparison and safety regressions are
reviewed. A faster CPU executor does not establish end-to-end HIP throughput.
No new T096 attempt, threshold change or Phase6 work is authorized.
