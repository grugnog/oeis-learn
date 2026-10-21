# Fixed-state measurement and batched generation follow-up

Phase 5 remains blocked. This implementation changes scheduling and measurement,
not the architecture, recipe, numerical limits, sampler distribution, search
attempt count, cadence, score denominator or 80% admission threshold. Its new
source/image now passes the actual HIP correctness gate (T092), while measured
complete-screen cost admission fails. The current reviewed outcome is below;
older receipts retain their original source/image identities.

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

## Executed actual-device command (reference)

The current-source run below used this command with the original prepared
inputs. It builds/reuses a matching image. The completed result supplies the
needed evidence; do not repeat it unchanged.

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
At that local-only checkpoint, actual HIP and packaged native execution were
pending. The subsequently supplied current-source outcome is recorded below.

## HIP test-package import follow-up

The supplied `008-phase5-batched-20261020T235941Z` archive passed input checking
and the three existing GPU tests, then failed all six new parity cases on
`ModuleNotFoundError: test_bigint_decoding`. None reached decoding; the cost
benchmark did not start. All 14 archived file hashes verify. These failures are
test wiring errors and provide no parity or throughput result for the new source.

The GPU test now imports the helper by its qualified package name,
`tests.integration.test_bigint_decoding`, at module load time. This avoids a
dependency on the integration directory being on `sys.path` and makes import
failures visible during CPU collection, before any HIP-only skip guard. A fresh
subprocess regression imports the GPU test through its package and executes the
shared stochastic parity helper on CPU without adding that directory to its path.
It failed before the fix and now passes. It does not simulate or claim HIP.

Validation: **37 passed**, including the package-import regression, existing CI
regressions, CPU mixed-prompt parity and generation scheduling tests. Ruff F/E9
and whitespace checks pass; all eight GPU cases collect under pytest's
`--import-mode=importlib`. HIP parity and cost were pending at this import-fix
checkpoint. The subsequent run below validates the correction on actual HIP.
The source-bound image rebuild used the existing dependency layers; no
production algorithm, recipe, runtime resource or numerical setting changed
in this import correction.


## Reviewed current-source HIP run — 2026-10-21

The supplied `008-phase5-batched-20261021T001809Z.zip` is **BLOCKED**, with
`qualified:false`. All21 archived file sizes/hashes verify. The expected source
snapshot, GPU result and benchmark agree on
`sha256:1e8d469e97912d3af3e40ad0b255a2df18ca09d13d9cb1209f67580cb9386a52`
and image
`sha256:3932b17d9612f8eb6e5b8128329fa7a6403839e66a49db54668a4becde89896b`.
Full reviewed identities, test names, group observations and cost values are in
[phase-5-batched-hip-review.json](phase-5-batched-hip-review.json).

**T092 is complete:**9 actual HIP tests pass with zero failures/errors/skips,
including finite large-input encoder updates, cached/full decoding,
checkpoint/resume and all six new mixed-prompt parity cases. The test-package
import correction is validated on the device.

All12 benchmark groups (two fixed calibrations × three conditioning controls ×
two temperatures) pass singleton/batch token-disposition parity, unchanged
parameter hashes and deadline completion. Calibration counts remain15 and25
updates independently of timing repetitions. Some synthetic outputs reach the
1024-token cap or grammar dead ends; those dispositions are retained. These are
non-scoring cost fixtures, not a learnability success or failure. The partial
progress report remains labelled PARTIAL; the final benchmark is complete.
Its `status:PASS` means successful measurement collection, not cost admission.

| Projected component | Seconds |
| --- | ---: |
| Training | 241.04 |
| Periodic synthesis | 12,096.84 |
| Full development loss | 107.77 |
| End-screen generation | 32,435.01 |
| End-screen fitting | 2,467.22 |
| End-screen hidden scoring | 394.51 |
| Checkpoints | 6.94 |
| **Total** | **47,749.33** |
| **80% admission allowance** | **5,760.00** |

Checked prepared counts remain512/128/128 plus tiny32, with native admission.
The cost calculation includes all30 scheduled validations and256 end-screen
prompts, with zero omitted prompts. Batch8 p95 is31.6748s; singleton p95 is
17.7881s. Generation accounts for43,837.95s of the projection. The result
exceeds the frozen cost limit by8.29x. T095 stays open and T096 has not run.
Fifteen small-value slowdown investigations range23.87–78.28x, leaving T094
open. The measurement-only scope provides no new T089/T090/T091/T093/T097
completion; T098 and Phase 5 overall remain blocked.

An unchanged rerun is unnecessary: this is now a measured cost blocker, not an
import, parity or missing-evidence failure. Further generation optimization
must retain the registered semantics and rerun affected parity/cost gates.
As a diagnostic bound, holding non-generation costs and the complete recipe
fixed consumes3,911.38s outside generation. Batch8 p95 would need to fall to about
1.34s (23.71x faster) to fit5,760s. This bound is an inference from the measured
phase totals, not a registered recipe or permission to change limits.
Alternatively, a resource/recipe revision must be explicitly preregistered
before training; no budget, cadence, output cap, denominator or threshold is
relaxed automatically. No further training or Phase 6 study is launched.

This evidence commit only updates Spec 008 qualification documentation and its
review receipt. Those paths are outside the implementation snapshot collected
by `experiments/source_snapshot.py`; the tested source/image identity remains
applicable to T092.

## Phase 5 cost decision — 2026-10-21

A limited contract review found no measurement basis for treating the initial
7,200-second value as a throughput requirement. The plan already describes
initial budgets as finite qualification defaults, and the experiment contract
permits a preregistered revision after measurement. Training updates contribute
about241 seconds to the current projection; generation contributes43,837.95
seconds (91.8%). A broad optimization pause would therefore precede evidence
that the model can pass the diverse32 learning gate.

Phase 5 will continue with T095 revised to produce scope-identified cost
receipts. The existing47,749.33-second result remains the failed complete
transfer-screen receipt. It must not be used as the diverse32 estimate because
the current calculation charges256 end-screen prompts where the memorization
path evaluates32, and the exact periodic workload and fixed costs also need
explicit scope accounting. T095 will implement matching cost-policy/coordinator
parameters, measure the exact diverse32 workload on the current source/image and
preregister a finite budget before one T096 run. All denominators, controls,
termination rules and the31/32 gate remain frozen. Passing that diagnostic will
not qualify the later transfer screen.

Substantial generation optimization is deferred until the diverse32 result
shows whether repeated research screens are warranted. The measured full-screen
failure and small-value slowdown investigation remain open evidence, not erased
or reclassified by this decision.
