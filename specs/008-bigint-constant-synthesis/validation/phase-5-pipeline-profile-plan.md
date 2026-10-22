# Whole-pipeline resource and performance profile — proposed2026-10-22

Status: PLAN ONLY. User requested planning before implementation or execution.
This is a focused T094/T097 follow-up within Spec008. No profiling code, new
model training run, qualification attempt or parallel scheduler is implemented
or launched by this document. A broader synthesis/model spec is not needed to
collect this evidence. Production scheduling changes require their own reviewed
scope after the results.

## Decision and questions

Decide where engineering effort will shorten the complete workload: GPU decode,
CPU grammar/sampling/device synchronization, independent CPU candidate work,
or overlap between those stages. Identify memory/thermal/CPU-quota constraints
before choosing concurrency. Stop pursuing WASM microbenchmarks in isolation
until their share of the critical path is measured.

The last reviewed HIP forecast allocates86.3%/89.8% to generation for diverse32/
transfer-screen,3.0%/0.8% to weight updates and about9.9%/9.3% to fitting,
execution and independent verification combined. Those are forecasts, not a
measured wall-time breakdown. Current CPU adoption passes435 tests and the
zero/64KiB comparison, but the remaining13–14x small-value retained gap does not
establish a dominant whole-workload problem. See
[HIP forecast](phase-5-scoped-hip-review.json) and
[CPU adoption](phase-5-engine-adoption-review.json).

T093 historical-model comparison is user-deferred, not passed or required for
this work. A current bigint model used below is a separate frozen analysis
input; it is not conversion of pre007 weights or completion of T093.

## Frozen inputs and meaningful workload

Use the existing selected bigint checkpoint from the completed1800-update
seed20261016 run, plus its original model/codec/input manifests and prepared
training data. Resolve it from the run metadata; verify checkpoint files and
parameter hashes. Load a copy as an explicitly identified weights-only analysis
under the current engine policy. Never resume or mutate the parent contract,
checkpoint, ledger, selections or original negative result. Missing/incompatible
current bigint weights are an explicit input blocker; do not silently substitute
untrained weights or import a historical model.

Freeze a new diagnostic manifest before collection. Include source/image,
policy, hardware/driver/torch versions, checkpoint and data hashes, prompt IDs,
seeds/order, workload counts, profiling settings, quotas and finite allowances.
Use the existing32 diverse **synthetic training prompts** (at least four
structures), all20 visible terms, original tokenizer and1024-token limit. Record
input widths, target/emitted lengths, EOS/cap/dead-end/timeout dispositions and
actual batch sizes. Do not cherry-pick only fast or successful candidates.

| Segment | Fixed work | Purpose |
| --- | --- | --- |
| Startup/warmup | Capture cold role/model/worker initialization once;20 scratch warmup updates and one batch8 greedy decode, excluded from steady-state summaries | Separate startup, allocator/cache and compilation cost |
| Weight updates |500 timed updates, batch8, in20 blocks of25 using a frozen schedule over128 synthetic training examples; restore identical copied weights/optimizer/RNG/data state before each block | Enough sustained work to sample training utilization, with forward/loss/backward/optimizer/metrics costs separated |
| Validation | One full256-row synthetic development teacher-forced loss pass, with fixed order/batch policy | Measure actual validation shape without candidate or hidden-term selection |
| Production portfolio |32 prompts: correct conditioning with greedy plus seven stochastic attempts; shuffled and absent conditioning with greedy only; eight fixed templates per prompt; baseline generation batch8, fitting serial | Exercise real inference, fitting, raw checks and reference admission across the existing diverse set |
| Continuation verification | Execute the frozen selected candidates over their100-index synthetic training horizons in the separate scorer role | Include production/reference verification cost while keeping truth out of generation |
| Short/long trace probes | Separate replay of one short and one long decode from the fixed workload; if no long output exists, use a separately labelled existing fixed-state long-output fixture | Attribute kernel/launch/synchronization costs without pretending a worst-case stress output is typical |

Scratch update states are disposable and never selected, promoted or used for
main inference; every inference pass starts from the selected original weights.
Update blocks measure an operation's throughput, not learning progress. Training
examples may be reused; they are not independent statistical trials.

This diagnostic intentionally uses fewer conditioning-control attempts than a
qualification screen. Label it diagnostic; it cannot create a cost-admission
receipt, replace a complete evaluation, or retry T096. Preserve current60s
portfolio ceilings, per-request limits, fuel, memory and cleanup. Record every
requested/started/completed/unstarted attempt. A capped portfolio is evidence
about its actual path, not permission to extend its time or silently drop rows.

## Collection layout and finite size

Run one portable host harness using the current image and actual HIP. Planned
main budget is85minutes of work, with a90minute external ceiling including
cleanup/archive; image building is recorded separately. Record baseline wall
start/end timestamps independently of budget reservations. Proposed stage caps:
120s capability check,180s warmup,300s scratch updates/validation,3600s paired
portfolio/scoring,600s focused traces,300s analysis/archive. Work is finite;
partial/incomplete results and unstarted stages are retained on any cap.

Two paired measurement passes use identical frozen inputs/seeds and independent
analysis/cache directories: ordinary execution and execution with lightweight
telemetry. Alternate their order across the four groups of eight prompts, while
preserving each arm's own prompt/attempt order and cache history. Restore state
outside timed update windows. Describe cold versus warm worker/compiled/result
cache state and report hit/miss counts; do not let the second arm inherit cached
answers from the first. Reuse existing production cache policy within each arm.

Expected duration is roughly45–60minutes from historical long-output forecasts;
a trained checkpoint may finish sooner. Do not add repetitions just to fill a
clock target. Keep per-group and per-attempt distributions; two passes do not
provide a precise tail-latency estimate. If the cap prevents completion, report
coverage and narrow a separately identified follow-up rather than extending it.

Expose16 CPU-equivalents to the diagnostic roles, subject to actual effective
cpuset/quota; record physical cores, logical CPUs, affinity and throttling.
Keep the current serial fitter, batch8 decoder and existing PyTorch thread
settings. This measures current scheduling with room to observe unused capacity,
not the speedup of a parallel implementation. Separate comparison with the old
four-CPU quota can follow only if measured throttling makes it useful.

## Lightweight telemetry throughout the instrumented pass

Sample at1Hz from a separate bounded collector, joining all records to monotonic
role/group/attempt events. One observation per second is sufficient for sustained
usage; brief kernels and updates need the bounded trace windows below. Record
collector CPU/time, missed samples and sensor coverage. Do not spawn SMI tools
per sample or stream unbounded command output.

| Area | Record | Interpretation |
| --- | --- | --- |
| Whole run | Monotonic wall timeline; role startup, model load, data conversion, generation, fitting, verification, checkpoint/export and explicit waits | Exclusive top-level wall phases plus nested spans; expose residual/unattributed time instead of double-counting |
| CPU | Host per-core busy/idle, process-tree CPU seconds, runnable threads, context switches; cgroup CPU usage/quota/throttled time; I/O and pressure where readable | Report CPU cores used as CPU-seconds/wall-second and as fraction of effective quota; distinguish one busy core from a fully used machine |
| GPU | Selected AMD device busy/memory-busy counters where supported, clocks, temperature and power; PyTorch allocated/reserved/peak bytes | Device busy is sampled activity, not occupancy, FLOP efficiency or proof of bandwidth saturation; report device-wide background activity |
| Memory | Role/native-worker RSS/high-water marks, cgroup current/peak/events, host available/swap; allocator peaks per stage | APU GPU/system counters overlap. Never sum RSS, cgroup usage and GPU allocator bytes as independent consumption |
| Work | Actual emitted/useful tokens per second, batch/sequence lengths, termination, fit/execution/reference counts/outcomes, queue/IPC/cache timings and worker restarts | Distinguish more work from slower operations; include failed/capped work and compare like shapes |

Use stdlib `/proc`, readable cgroup files and AMD sysfs counters, plus existing
PyTorch memory APIs. Map the GPU sensor to the selected render node/PCI identity;
do not guess from the first card. Capture supported sensor names/units/cadences
in the capability report. Missing/ambiguous/invalid readings remain null with a
reason; no fabricated0% utilization. APU power may include CPU package power.
Use only already available container tools as optional sources. No host package,
root access, driver change, privileged launch or runtime socket mount is needed.
If host PID/cgroup sampling is possible, bind each record to verified process
identity; otherwise mark that scope unavailable and retain in-role metrics.

Do not change launch isolation to obtain a sensor. Keep collector diagnostics
outside the model's inputs and preserve separate generation/scoring mounts.

## Additional instrumentation and what to disable

Keep ordinary correctness, independent admission, finite checks, seeds, limits,
cache behavior and result hashes. Preserve all old profiling utilities and
archived readers, but disable detailed WASM phases, per-byte timers, limiter
request traces, zero-reservation comparisons and repeated full CPU qualification
suites in this run. They already answered narrower questions and would distort
or inflate the whole-pipeline profile.

Add coarse host spans around encoder input/features, encoding, complete decode
calls, native request/response, final admission, scoring and logging/checkpoint
I/O. Keep counters per decode/attempt rather than writing one log row per token.
CPU fitting spans must distinguish solver search from production checking where
existing native timing supports it; otherwise mark the unresolved portion.
Do not interpret a native request's wall time as pure WASM time.

The existing `PhaseTimer` synchronizes HIP before/after each measured training
phase. Baseline retains current production behavior; additional whole-run timing
must not synchronize every token/operator. Keep those historical timer values
labelled synchronized/perturbing and nested. First attribute their cost in a
short paired replay. Any later timer-only replacement with asynchronous GPU
events belongs to a separately tested observer-overhead change, not a silent
alteration of this baseline. Preserve required validation even if entropy,
per-parameter finite checks or logging prove expensive; first measure them.

Collect short PyTorch CPU/HIP traces **outside headline throughput measurements**:
one five-update window and one short/long generation window, at most30s and
256 token steps per decode window. Initial traces use operator/range events,
no shapes/stacks/tensor memory histories. Check that GPU kernels are actually
present; a CPU-only trace is incomplete HIP attribution. Add one capped CPU
call-stack or allocation trace only if the basic trace leaves an identified gap.
No full-run operator trace, hardware-counter replay sweep or repeated detailed
memory allocation trace. Extra tracing must stop at the predeclared cap and
cannot change the main workload's outputs or measured runtime.

Candidate range labels: encoder features/forward; decoder prefill/cached step;
grammar support/mask; sampling; device-to-host token/scalar transfers; logging;
loss; backward/optimizer; queue/IPC; solver; production checks; independent
reference. Source inspection finds per-token `.cpu().tolist()` calls and
synchronized training phase timers, but only measured traces can establish their
importance. GPU kernel intervals may overlap; use their interval union rather
than summing kernel durations to infer elapsed device activity. Align device and
host timestamps through the profiler's clock correlation.

## Observer overhead, evidence and decision rules

Compare paired ordinary/lightweight runs by group, emitted tokens, work counts,
outputs and parameter hashes. Report telemetry overhead separately; investigate
median paired slowdown above5% or output/disposition differences. If overhead
is material, keep ordinary timing authoritative and reduce sampling/ranges in
a new labelled pass. Profiler windows are explanatory, never throughput claims.
A missing sensor does not imply idle hardware. Material changes in output length,
timeouts or cache hits invalidate a simple speed ratio and must be explained.

Produce an interactive or standalone timeline plus a compact table of exclusive
wall shares, CPU core-equivalents, GPU busy distributions, memory/pressure,
tokens/sec, native cost, waits and collector coverage for each phase. Include
cold startup, failed work and unallocated residual. Compute the upper bound on
whole-run benefit from accelerating each category before recommending work.

- GPU consistently busy and decode dominates: investigate batch occupancy,
  output-length effects and kernel work before adding CPU workers.
- GPU has gaps aligned with grammar/sampling/transfers: investigate CPU/device
  synchronization and per-token host work, rather than assuming GPU compute is
  saturated or that WASM workers fix decoding.
- GPU waits during appreciable independent native work and CPU has spare cores:
  propose a bounded process-pool/overlap experiment, initially4 then8 workers,
  with16 considered only if headroom/scaling justify it. Do not implement a pool
  merely because16 cores exist. Each proposal must preserve deadlines,
  deterministic selection, bounded aggregate memory and separate engines.
- Memory/CPU quota, thermal or I/O limits correlate with delay: address the
  measured constraint before attributing it to arithmetic or inference.
- Native work is a small wall share: retain the CPU optimization and defer more
  timer/WASM work unless correctness or an explicit cost target requires it.

Archive source/image/policy/data/checkpoint identities, planned/actual work,
phase events, raw1Hz samples, bounded traces, sensor availability/overhead,
results/dispositions and cleanup under a fresh analysis identity. Reserve core
summary/manifest first. Cap raw traces at128MiB total and the result archive at
256MiB, recording hashes/bytes/reasons for omissions. No weights, private source
payloads or final/OEIS scoring data in the ZIP. Both passes and partial failures
must be exportable. Old negative T096 and historical receipts stay unchanged.

## Implementation and execution sequence after plan review

- [ ] W001 Add a separate diagnostic manifest/host harness and analysis identity;
  reject training/resume/final-study authority and missing source/weights. New
  summaries must reflect T093 deferral rather than copying stale blocker lists.
- [ ] W002 Add coarse spans and1Hz resource collection with GPU mapping, CPU quota
  normalization, missing-sensor handling and bounded cleanup/archive tests.
- [ ] W003 Add the fixed-state paired workload and short trace mode; test stable
  seeds/counts/state restoration, no cross-arm cached answers, limits and trace
  bounds before actual HIP execution. Separate mocks/CPU tests from device evidence.
- [ ] W004 Push incrementally and provide one copy-pastable Podman/Docker command
  for capability check plus bounded collection/rerun/archive. No command for this
  proposed harness exists yet. Do not launch it automatically or wait for PR CI.
- [ ] W005 Review actual HIP evidence, publish phase/resource/overhead results and
  recommend one measured next step. This does not itself close T094/T098, admit
  transfer-screen training, retry T096 or require deferred T093.

Prefer the existing pinned PyTorch profiler. Verify capabilities of the installed
PyTorch2.9.1/ROCm7.2.1 image rather than assuming a newer ROCm tool is available.
Relevant primary documentation (reviewed2026-10-22):

- [PyTorch2.9 profiler](https://docs.pytorch.org/docs/2.9/profiler.html): scheduled
  CPU/device activities and optional tracing costs; shape/stack capture can
  introduce additional overhead.
- [AMD PyTorch profiler guide](https://rocm.blogs.amd.com/artificial-intelligence/torch_profiler/README.html):
  CPU/GPU operation profiling on ROCm. Its privileged launch example is not part
  of this plan; retain this project's launch boundary.
- [AMDGPU sensors](https://kernel.org/doc/html/latest/gpu/amdgpu/thermal.html):
  busy counters and sensor semantics, including CPU-inclusive APU power.
