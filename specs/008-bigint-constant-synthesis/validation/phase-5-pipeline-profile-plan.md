# Whole-pipeline resource and performance profile — proposed2026-10-22

Status: REVIEWED diagnostic evidence, W005 complete on2026-10-23. The
[corrected HIP rerun](phase-5-pipeline-profile-rerun-review.json) verifies actual
device traces, isolation, CPU scoring and resource collection; its workload
remains PARTIAL at the original deadlines. No new training experiment,
qualification attempt or parallel scheduler was launched. This is a focused
T094/T097 diagnostic within Spec008; production scheduling changes still require
their own reviewed scope. Earlier pending statements below describe prior stages.

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
| Weight updates |500 timed updates **per arm** (1000 total), batch8, in20 blocks of25 using a frozen schedule over128 synthetic training examples; restore identical copied weights/optimizer/RNG/data state before each block | Enough sustained work to sample training utilization, with forward/loss/backward/optimizer/metrics costs separated |
| Validation | One full256-row synthetic development teacher-forced loss pass per arm, with fixed order/batch policy | Measure actual validation shape without candidate or hidden-term selection |
| Production portfolio |32 prompts: correct conditioning with greedy plus seven stochastic attempts; shuffled and absent conditioning with greedy only; eight fixed templates per prompt; baseline generation batch8, fitting serial | Exercise real inference, fitting, raw checks and reference admission across the existing diverse set |
| Continuation verification | Execute the frozen selected candidates over their100-index synthetic training horizons in the separate scorer role | Include production/reference verification cost while keeping truth out of generation |
| Short/long trace probes | Separate bounded replay of the observed shortest and longest correct-conditioning decode batches; explicitly report missing long-output coverage rather than substituting untrained weights | Attribute kernel/launch/synchronization costs without pretending a worst-case stress output is typical |

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

The work table applies to each measurement arm, except the separate trace probes.
Two paired measurement passes use identical frozen inputs/seeds and independent
analysis/cache directories: ordinary execution and execution with lightweight
telemetry. Alternate their order across the four groups of eight prompts, while
preserving each arm's own prompt/attempt order and cache history. Restore state
outside timed update windows. Describe cold versus warm worker/compiled/result
cache state and report hit/miss counts; do not let the second arm inherit cached
answers from the first. Reuse existing production cache policy within each arm.

Treat45–60minutes as a planning estimate, not a demonstrated runtime: before
launch, compute a conservative bound from both arms, actual deadline scope
(per prompt versus per batch), scoring and role startup. If the fixed work cannot
fit its stage caps, revise and freeze the manifest before execution; never
silently shorten only the second arm.
A trained checkpoint may finish sooner than historical long-output forecasts;
do not add repetitions just to fill a clock target. Keep per-group and per-attempt distributions; two passes do not
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

Capture a short idle/background sample during capability checking and record
competing workloads. Prefer cumulative CPU/cgroup deltas at stage boundaries
as well as1Hz samples: sampling alone can miss short-lived native children.
Use monotonic timestamps for all same-host roles; report sensor sample age and
clock alignment uncertainty. Collector activity must be identifiable separately
from workload activity.

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

Run coarse measurements first. Collect short PyTorch CPU/HIP traces
**outside headline throughput measurements**, targeted using those measurements:
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
Deadline-limited portfolios can do different amounts of work under even a small
observer slowdown. Report this as a measurement effect, not automatically as
semantic nondeterminism. Use paired completed, matching attempts to estimate
per-operation overhead, and separately show unmatched/capped work; do not hide
that work or compare truncated aggregate times as a speedup. Compare update
parameter hashes at corresponding block ends, not against the unchanged parent.
A missing sensor does not imply idle hardware. Material changes in output length,
timeouts or cache hits invalidate a simple speed ratio and must be explained.

Produce an interactive or standalone timeline plus a compact table of exclusive
wall shares, CPU core-equivalents, GPU busy distributions, memory/pressure,
tokens/sec, native cost, waits and collector coverage for each phase. Include
cold startup, failed work and unallocated residual.

**Do not interpret this diagnostic's aggregate wall shares as the training run's
shares.** Its deliberately oversampled updates, reduced control portfolios and
paired/traced repetitions have a different mix. Publish two separate views:
(1) measured diagnostic phase times and utilization; (2) a schedule-weighted
projection using the intended training/evaluation frequency and actual token
length/batch distributions. State assumptions and uncertainty, and leave an
unmeasured category unknown. Exclude observer duplication, trace probes and
state restoration from that projection, while retaining real production startup,
checkpoint/export and waiting costs. Since this plan does not time a real
checkpoint/export, either add one bounded scratch checkpoint/export measurement
within the existing cap or explicitly label that cost unmeasured. Do not use
synthetic long-output stress results as the representative decode distribution.

Compute Amdahl-style upper bounds using the schedule-weighted **critical path**,
not summed CPU time or overlapping device spans. Spare cores establish capacity,
not independent work: document which native requests could be concurrent without
changing candidate dependencies, selection order or shared deadlines. Distinguish
speeding up native work from overlapping it with inference; the latter is bounded
by both available overlap and contention, particularly on a shared-memory APU.

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

- [X] W001 Add a separate diagnostic manifest/host harness and analysis identity;
  reject training/resume/final-study authority and missing source/weights. New
  summaries must reflect T093 deferral rather than copying stale blocker lists.
- [X] W002 Add coarse spans and1Hz resource collection with GPU mapping, CPU quota
  normalization, missing-sensor handling and bounded cleanup/archive tests.
- [X] W003 Add the fixed-state paired workload and short trace mode; test stable
  seeds/counts/state restoration, no cross-arm cached answers, limits and trace
  bounds, per-arm counts and schedule-weighted reporting before actual HIP execution. Separate mocks/CPU tests from device evidence.
- [X] W004 Push incrementally and provide one copy-pastable Podman/Docker command
  for capability check plus bounded collection/rerun/archive. The command below is implemented. Do not launch it automatically or wait for PR CI.
- [X] W005 Review actual HIP evidence, publish phase/resource/overhead results and
  recommend one measured next step. This does not itself close T094/T098, admit
  transfer-screen training, retry T096 or require deferred T093.
  Completed by the2026-10-23 [corrected rerun review](phase-5-pipeline-profile-rerun-review.json):
  all280 manifest entries and five actual-GPU trace exports verify, all three
  isolation/cleanup audits pass, and CPU scoring completes. Partial portfolio
  work, differing attempts and trace/projection limits are retained. The next
  proposed experiment targets cached-decoder bookkeeping; no optimization is
  implemented or throughput gain claimed by closing this review task.

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

## Implemented command and interpretation

```bash
python3 scripts/profile_bigint_pipeline.py --engine podman \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --output "tmp-validation/008-pipeline-hip-$(date -u +%Y%m%dT%H%M%SZ)"
```

Adjust the two existing input paths if moved; the selected checkpoint, parent
contract, prepared-data identities and every copied input are checked. The host
needs only Python and its normal engine connection. The current GPU image is
built unless `--image` supplies an existing image; a stale image fails the source
check before loading weights. Building is outside measurement caps. A new output
suffix reruns the analysis; existing outputs and the parent are never resumed.
Use `--engine docker` for Docker. Collection requests16 CPU-equivalents and
keeps two PyTorch threads, batch8 decoding and one active native request.

Concrete paired portfolio/scoring ceilings are eight group-arm pairs of240s
of generation plus180s of scoring:3360s within the3600s stage budget. Per-prompt
60s and native/reference limits remain nested inside these caps. The GPU role's
external ceiling is3100s and the scorer's is1600s, reduced by remaining allowance;
cleanup/analysis/archive have remaining room inside90minutes. The external host
watchdog polls every2s, caps each log at8MiB and each role's working output at8GiB.
These disk/log watchdogs are polling bounds, not filesystem quotas. Traces use
40MiB per-file export limits and the128MiB archive trace cap. The archive reserves
core reports and caps selected uncompressed artifacts at255MiB plus its manifest.

Implementation details and explicit coverage limits:

- The only model input is the selected current bigint checkpoint's weights,
  loaded with `weights_only=True`. A diagnostic AdamW is warmed20 disposable
  steps; its moments, the original weights, and RNG state are copied/restored
  identically before each25-update block. The parent optimizer is never resumed.
- Both arms retain separate native clients/caches in each role. Their two idle
  histories coexist in role memory, but requests stay serial; this additional
  diagnostic memory is disclosed and must not be mistaken for production memory
  at a future worker count. Tensor snapshot storage is counted separately; raw
  allocator/cgroup peaks include those analysis copies too. The scorer has no GPU or generation/training inputs.
- The ordinary arm records coarse phase boundaries and cumulative process CPU
  deltas. The observed arm adds1Hz telemetry and nested host spans; neither adds
  token-by-token logs. Short profiler windows preserve the observed batch and
  seed order, cap decoding at256 steps/30s, and report actual kernel-event presence.
- Refinement after implementation review: the longest observed decode is not
  called a long-output fixture if all actual outputs are short. Its coverage flag
  records that gap. An additional calibrated synthetic stress model is deferred
  unless reviewing the actual profile shows it is needed; it would not estimate
  this trained checkpoint's normal runtime. This avoids adding an unrelated
  disposable model to the first whole-pipeline run.
- Schedule-weighted update/validation estimates use the parent's intended cadence
  and maximum updates. Full-control, changing-weight/early-stop, checkpoint/export
  and periodic cache effects remain explicitly unknown. There is no fabricated
  total, normalized100% breakdown or Amdahl claim from that incomplete projection.
  Diagnostic exclusive phase times and resource timelines are reported separately.
- Native IPC includes solver search, compilation, production checks and transport.
  Detailed WASM phases are disabled; unresolved native subcosts stay unresolved.
  Hidden ancestor cgroup quotas or unavailable device counters are labelled gaps.

Share the generated sibling ZIP even for a FAIL/PARTIAL outcome. It contains
`summary.json`, the frozen manifest, `analysis.json`, a standalone hoverable
`timeline.html`, events/samples, workload dispositions, bounded traces and actual
container/cleanup audits. Inputs, weights, synthetic truth, caches and final/OEIS
data are excluded. No actual device result or throughput gain is claimed until
W005 reviews that archive. See tests in `tests/unit/test_pipeline_profile.py`.

Local validation:78 focused tests pass, including25 new profile cases; lint,
formatting, stdlib help/scope rejection, documentation links and Spec Kit gates
pass. [Local receipt](phase-5-pipeline-profile-local.json) is explicitly CPU/mock
evidence; no container or actual HIP execution is claimed.

## First HIP collection review — blocked harness, useful partial evidence

The [207-file verified archive review](phase-5-pipeline-profile-review.json)
records actual HIP500-update/256-validation-row measurements in both arms, with
matching block loss/parameter hashes. The GPU role spent740.8s; the host failed
afterward because Podman reports an empty `HostConfig.Devices`, so the current
audit rejected the GPU role before CPU scoring. Requested device arguments and
successful HIP computation do not by themselves replace actual device-node
attestation. Network/read-only/mount inspection and0.064s cleanup are recorded,
but the full device/isolation audit remains unqualified. All three operator
trace exports also failed and have no trace JSON; the cause is not established.

In the observed321.8s portfolio workload, exclusive decoding takes162.4s and
native request/response148.5s. Native child CPU time is148.1s; aggregate usage
averages about one core despite a16-CPU quota. Sampled GPU busy averages30.8%
during decode versus5.4% during native requests. No cgroup CPU throttling or
OOM events were observed. These measurements justify investigating bounded
native concurrency/overlap; they do not establish a whole-training speedup or
that the native time is entirely WASM. The prior forecast used a different
work/output-length mix. About242s outside container lifetime also needs explicit
launch/startup attribution.

Both arms start560/576 attempts and complete545 generations. Group3 leaves two
correct-conditioning attempts per prompt unstarted at the frozen ceiling.
540 paired attempt records match;20 differ, including six token-prefix hashes
and14 fitting dispositions. Therefore near-equal aggregate timings are not full
parity or an observer speedup claim. All256 template fits per arm are unsupported.
No completeness threshold or original negative T096 assessment changes.

Next work is harness reliability: audit actual device exposure before expensive
work, fix/prove bounded trace export, then recover CPU scoring and short traces
under a fresh linked analysis identity. Preserve this archive and original local
output, including its excluded inputs/truth needed for recovery. Avoid rerunning
the1000 updates or portfolios merely to repair reporting. Do not repeat the
quickstart command unchanged while those known harness failures remain. W005
and T094/T097/T098 stay open; T093 remains deferred.


## Harness correction and approved rerun — 2026-10-22

The user explicitly permits a fresh full run; recovery/resume machinery is not
needed. Keep the reviewed original archive unchanged. R001–R003 below are
implementation tasks, not hardware evidence; W005 remains open until the rerun.

- [X] R001 Reproduce and fix trace-export failure with bounded scheduling and
  partial evidence. Five CPU model updates on the pinned PyTorch2.9.1 reproduce
  the same Kineto failure and leave an exact40MiB `.json.tmp`. Recording one
  steady update within the same five-update replay exports10,445,980 bytes in
  the local reproduction. CPU-only capture explicitly lacks HIP kernels and
  cannot pass device attribution. The sampler's existing budget hook advances
  a scheduled trace only inside explanatory probes: eight active token steps,
  short-prefix prefill included, and a later64-step offset for the long probe.
  Ordinary and observed workload calls get no token profiler hook. Caps stay
  unchanged; a capped/invalid file is retained as `.partial.json` with bytes/hash
  and an explicit reason. Kernel-event presence is still mandatory for COMPLETE.
- [X] R002 Fix rootless Podman device evidence without allowing arbitrary devices.
  Every role records actual accelerator character-device nodes. The host compares
  their paths and major/minor numbers to its selected nodes; CPU roles must see
  none. Nonempty contradictory inspect metadata is rejected. Empty inspect
  metadata alone never grants a pass. Existing mount/network/read-only/root/
  device-request restrictions remain enforced. Each audit and host-role timing
  is retained; host time outside container lifetime is reported separately.
- [X] R003 Add an audited preflight using the same current image/devices/mount
  boundary before the paired workload, regression tests and the quickstart rerun
  command. Preflight requires actual HIP, a tiny kernel trace and one representative
  steady-update trace within a five-disposable-update replay from copied selected
  weights. It cannot mutate or select parent weights. Its420s external ceiling
  allows cold engine startup plus120s of device work. The main GPU role retains
  its3100s maximum, reduced by elapsed allowance; scorer ceilings are reduced by
  the remaining5100s collection allowance. Cleanup/export remain within90minutes;
  image build stays separate. Failure archives retain available role summaries.

The original500 timed updates per arm,32 prompt IDs, seeds/order, batch8, serial
native calls, per-request/60s portfolio ceilings, limits, scoring isolation and
negative T096 remain unchanged. Preflight adds only five disposable diagnostic
updates outside headline timing. A bounded portfolio may still yield PARTIAL;
that work coverage must not be relabelled complete or extended to make a report
pass. Rerun the quickstart with a fresh output suffix and share the ZIP, including
failed/partial cases. No native backend/runtime policy, model architecture,
training admission or worker-pool optimization is changed.

Local correction validation:88 focused regressions pass, including35 profile
cases; the [local fix receipt](phase-5-pipeline-profile-fix-local.json) records the
exact CPU reproduction, checks and explicit pending actual container/HIP gate.

## Host dependency correction — 2026-10-23

- [X] R004 Review the rerun archive and restore the stdlib-only host boundary.
  All three manifest entries in the four-member ZIP verify; no container stage
  started. Importing the device helpers triggered `tracking/__init__.py`'s eager
  legacy run-manager import, which requires YAML. Lazily exporting its two public
  classes preserves their identity and keeps telemetry/report imports independent
  of that dependency. The new full-host regression fails with the reported error
  before this change and passes with site packages disabled afterward. It uses
  fake engine/device execution but real preparation, source hashing, role audits,
  cleanup records, reporting and archive code; it asserts YAML/torch/numpy were
  not imported and parent files are unchanged. Existing run-manager YAML behavior
  is covered separately. No host package installation or device work is required
  for this correction.

The [review receipt](phase-5-pipeline-profile-stdlib-review.json) records100 local
passing tests, including37 profile cases. This does not verify actual containers
or HIP traces; W005 and T094/T097/T098 remain open. T093 is deferred and the
original negative T096 is unchanged. After pulling, rerun the quickstart command
with a fresh output timestamp and leave `--image` unset for the current source.
