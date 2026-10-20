# Fixed-state measurement and batched generation follow-up

Phase 5 remains blocked. This implementation changes scheduling and measurement,
not the architecture, recipe, numerical limits, sampler distribution, search
attempt count, cadence, score denominator or 80% admission threshold. Its new
source/image still needs actual HIP qualification; old receipts do not qualify it.

## Reviewed workstation evidence

The supplied `008-phase5-hip-20261020T202907Z` run passed arithmetic114, fitting36,
containment64 and actual HIP3 tests. Prepared inputs passed checking and reported
512/128/128 plus tiny32. Cost admission failed at13,930.43 seconds versus5,760
allowed. Learnability was blocked before training. T091, T093, T094, T095, T096,
T097 and T098 remained pending. These are observations of the old source/image,
not new task completions for this implementation.

The supplied `008-phase5-cost-diagnostic-20261020T224008Z` archive collected
successfully but projected183,891.05 seconds. Five repetitions also performed
more synthetic optimizer updates, producing longer programs. Most decode cost
remained14–16ms/token. This exposed a measurement confound, not evidence of a
17.68x hardware regression. The Python profile identifies substantial cached
decoder/linear work and additional grammar work. An anomalous800% device-table
entry and duplicate-flow warning prevent precise utilization attribution.

The fixed-state archive `008-phase5-fixed-state-20261020T225828Z` matched source
`sha256:694d12f51b40ad370f58538b26b972d6104d586d21713795a0cfbde2d20c4470`
and image `sha256:6d7f1b2a719fa6f2e1e0d94c35d30341adb8b11d8792092225769cbf935a79d5`.
All12 archived file hashes checked without disagreement. On AMD Radeon Graphics
gfx1151 / HIP7.2, both synthetic model states retained unchanged parameters and
passed greedy token/disposition parity at batches1/4/8. Median measurements:

| Output tokens | Batch1 seconds/prompt | Batch8 seconds/prompt | Throughput gain |
| --- | ---: | ---: | ---: |
| 64 | 0.923781 | 0.127950 | 7.21985x |
| 1024 | 16.301111 | 3.840528 | 4.24450x |

The older model states followed15 and25 synthetic optimizer steps respectively.
Repeated-prompt greedy throughput does not establish mixed-prompt/stochastic
parity or the full cost of the new implementation. The long-output result still
exceeds the frozen screen budget even after the measured batching gain.

## Implementation and acceptance

`decoder.bigint_generation.sample_batch` runs the existing cached sampler over
up to eight compatible visible prompts. Each request retains its seed, greedy
or0.8 stochastic temperature,1,024-token limit and original deadline. Portfolios
are interleaved at decode boundaries, preserving greedy-then-seven-stochastic
order and serial fitting/admission. A shared slice of the original global
allowance bounds each prompt group; every portfolio retains its60s ceiling.
Completed and token-capped rows retain separate dispositions. Deadline failures
retain partial tokens and cannot become repaired sources or fitted successes.

CPU and mandatory HIP tests compare singleton/batched/reordered mixed prompts
under correct/shuffled/absent conditioning, for greedy and stochastic sampling.
Additional tests check serial fit order, tail groups, all eight attempts, control
denominators, partial timeouts and early EOS beside a token-capped row. Hardware
tests live in the existing `bigint-gpu` suite and fail rather than skip when HIP
is required.

Benchmark protocol `fixed-state-batched/v2` separately calibrates two model
states using three or five updates per fixed capacity/shape fixture. These counts
are fixed protocol constants. Timing repetitions cannot change them. Training
measurements restore parameters, optimizer and RNG before the next sample;
generation has no optimizer updates. Parameter hashes are checked throughout.
Warmup is outside timed generation. Records include model/fixture/memory/output
hashes, seeds, token lengths, termination/failure and single/batch parity.

Shape-matched generation measures different visible prompts, all three controls,
greedy and stochastic calls, batch8 latency and singleton references. Singleton
tails conservatively include the full eight-prompt encoder work. The worst
per-state/control/temperature p95 is used rather than a pooled quantile that
could hide a slower case. Timed deadline failures or parity failures block
admission. No scoring/private data or retained trained weights enter this role.

`fresh.cost_policy.generation_seconds` charges each full group once and tail rows
at the singleton upper bound. Preflight and coordinator role allowances share
this function; separate attempt rounds never combine tail rows into imaginary
full batches. Legacy singleton receipts retain their original calculation.
All scheduled validations and256 end-screen prompts remain included.

## Next actual-device command

Update this PR checkout and run from its root. Reuse the original prepared inputs;
the default builds/reuses an image matching this new source. Do not pass the old
image digest above.

```bash
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --performance-only \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "tmp-validation/008-phase5-batched-$(date -u +%Y%m%dT%H%M%SZ)"
```

This checks the existing inputs, runs actual HIP parity and the full native/neural
cost measurement, and writes the usual bounded ZIP. It skips unrelated CPU and
retirement reruns and records those execution gates as pending. It cannot combine
with preparation, checkpoint diagnosis or learnability. A collected but failed
preflight returns BLOCKED / exit2; execution errors return FAIL / exit1.
The expanded fixed-state/control measurement has the existing one-hour role cap
and can take substantially longer than the previous scalar probe.

Review that ZIP before launching the registered conditional learnability run.
A failed long-output cost remains a resource/recipe preregistration decision;
no budget, cadence, output cap or success threshold is automatically changed.
Matched checkpoint rescue, both-engine evidence, small-value slowdown review
and operations/export review remain separate requirements.

## Local validation

CPU parity and production scheduling/checkpoint/coordinator tests run with a
CPU-only PyTorch installation. The model-state test mutates mock training weights
and verifies that two versus five timing repetitions produce identical calibrated
hashes. These tests do not establish accelerator throughput or learnability.
The broader local set passed **130 tests**, with two native-image skips and one
GPU deselection. After adding partial neural progress reports, the affected set
passed **70 tests**, with the same two skips. Ruff F/E9, Python compilation,
whitespace and stdlib-only host help pass; all eight HIP test cases collect.
Exact local counts and changed-code hashes are in
[phase-5-generation-local.json](phase-5-generation-local.json).
Actual HIP and packaged native qualification are pending the new image.
