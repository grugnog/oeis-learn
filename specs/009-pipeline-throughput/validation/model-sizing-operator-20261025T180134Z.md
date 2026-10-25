# Corrected model sizing: all three profiles qualify for bounded hardware work

Accept the completed finite sizing screen at producer
`56f61564b0b504b9aa5ffc3273bb73974ebaf87e`, source
`sha256:735ede18987edb3d48eeab8c6d9001ec10a5f63c6f0d6581304702acc44d5f15`.
The unchanged `009-model-sizing-20261025T180134Z.zip` is 7,130,107 bytes,
SHA256 `02306e7d77637af302361d064e39fd4bae94e221527f1a4a7d687918f36cbd93`.
All 1,198 listed members plus manifest,1,089 stage-file hashes,26 immutable PASS
receipts and the exact720-entry source closure verify. The invocation ends PASS
in1,411.25s (23.52min), including shared qualification/build/inspection/export.
[Machine-readable review](model-sizing-operator-20261025T180134Z.json) retains
identities, all15 trials, parity boundaries, native clocks, memory categories and
absolute rates. Original attachment bytes are unchanged.

## Qualification and execution correction

170 host tests,692 CPU tests plus31 subtests (723 JUnit cases, one GPU-only
CPU-suite deselection) and20 actual-HIP tests pass; no mandatory skips. All three
CPU references use v2 manifests and the exact execution policy. The installed
image/source prerequisites and18 reference/trial container inspections pass:
read-only roots, unprivileged, one16-CPU quota and120,744,574,976-byte
(112.45GiB) shared memory ceiling per active role. Torch threads remain2/1.

The corrected CPU/HIP fixtures and all15 trial encoders/cached decoders now pass
the original2e-5 absolute/2e-4 relative tolerances and greedy-ID assertions.
All 84 exported parity records are finite, have zero mismatched elements and audit
FP32/highest matmul, TF32 off, deterministic algorithms, fused encoder/MHA paths
off and SDPA math only. Maximum absolute error across these records is
0.00000751019. This confirms the corrected policy works on this actual target;
the [first collection's failed implicit dispatch](model-sizing-operator-20261025T170755Z.md)
remains historical evidence and is not rewritten. Other production execution
policies are not qualified by this scoped result.

Both already-completed producer workflows were inspected once: Foundation review
40218403831 and Container/native acceptance40218403968 succeeded. New publication
CI is not awaited.

## Fixed-work training measurements

Exactly cold, excluded warmup and three calibrations complete for every profile.
Each executes32 actual AdamW updates at unchanged batch8 on the same128 real
training rows,15,360 training tokens and32 teacher-forced validation rows. Actual
parameter counts match the frozen constructor formulas; all weights change during
updates and remain unchanged during generation. Fresh initialization is used for
all three profiles; no parent trained weights are resized or inherited.

| Profile | Actual parameters | Median training tokens/s | Updates/s | Validation rows/s | Largest observed role upper memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| m8 | 8,118,294 | 5,934.19 | 12.36 | 189.09 | 2.45GiB |
| m32 | 31,903,510 | 4,219.27 | 8.79 | 153.41 | 3.36GiB |
| m100 | 104,496,662 | 2,091.76 | 4.36 | 98.80 | 5.89GiB |

Medians use all three registered complete calibrations; cold/warmup remain in the
review without entering aggregates. The104.5M model processes about35.2% of the
8.1M model's training tokens/s and49.6% of the31.9M model's. Those are different
architectures, so this is an absolute capacity/cost tradeoff, not an optimization
speedup or evidence of learned detail. The104.5M training sections last only
7.01–7.45s across calibrations. Rates do not establish thermal steady state,
checkpoint costs, long-sequence capacity or multi-day cost admission.

## Real generation and native work

All 960 planned attempts start:32 distinct prompt IDs per batch8/32 arm, unchanged
per-prompt seeds, original60s group clocks and1024-token maximum. No padding,
replicated capacity rows, forced tokens, retries or favorable subset is introduced.

The8.1M model's320 attempts all terminate with explicit `grammar_dead_end`, zero
decoded programs and67,680 emitted partial tokens across five roles. Each of the
31.9M and104.5M models completes320/320 attempts,4,480 emitted tokens and the same
short14-token constant-return source. All 640 complete programs are independently
submitted to native fitting; all return complete `unsat_in_scope`, zero admitted
assignments and zero independently verified programs. The registered model-sizing
report's hardware eligibility requires the actual update/parity/resource work;
it does not mean useful generated-program throughput has qualified. No learning
quality or curriculum decision is drawn from these seeded fresh-model samples.

Every native job is COMMITTED with exactly one reservation/consumption, zero work
left active/reserved at the boundary, matching public response/evidence hashes and
strict original4s admission-through-completion clocks. Maximum observed dispatch
wait is3.79ms; maximum native admission-through-completion is0.407s. Four-worker
waves are drained; no fixture programs fill missing generation. Exact raw native
responses stay private as specified; compact public evidence does not requalify
fitting semantics or create research admission.

Generation median times cannot rank architectures fairly: m8 emits6,768 tokens
per arm, while m32/m100 emit448. Batch8/32 median generation times are15.20/6.26s,
0.74/0.54s and1.09/0.71s respectively. The shorter larger-model outputs explain
much of their shorter whole-role time. Including engine startup/cleanup, median
trial times are30.07s,14.34s and20.29s. These are observed absolute mixed-work
costs; none establishes full-pipeline adoption or a meaningful programs/min gain.

## Resource interpretation and next investigation

303 bounded one-Hz telemetry samples have no collection errors; watchdogs observe
no breach and cgroups no memory max/OOM event. All roles clean up within0.192s.
For m100, observed cgroup anonymous peaks are about1.22–1.23GiB and file-cache
peaks0.20–0.22GiB; separately observed driver residency contributes to the
conservative5.89GiB role upper value. These are overlapping accounting domains,
not independently additive pools. The bound is sampled, workload-specific and
not the prospective24.61GiB reservation estimate or a worst-case guarantee.

Calibration whole-role CPU-equivalent medians are1.02/1.76/1.70 and sampled GPU
busy means approximately33.4/32.8/47.7%. These include setup, parity, short updates,
validation, early generation and serialized native work. They cannot isolate
training GPU duty or be compared directly with the earlier90–91% forced-replay
overlap result. Low CPU usage alone does not justify raising Torch/native threads.
Fresh processes and a few seconds of updates cannot prove persistent memory
stability, nor that the hardware has reached its best useful throughput.

The practical next engineering candidate is104.5M, following the user's preference
for larger capacity: it is the largest tested, passes all current gates, fits with
ample observed headroom and retains a usable short-update rate. This is a
prospective integration direction, not a claim that it is the largest possible,
best learning model or an already selected production configuration. Keep31.9M as
a measured fallback; do not run another isolated size ladder just to fill RAM.

The next useful investigation should combine persistent model/optimizer state,
actual update/generation/native work, checkpoint/stop/resume, memory categories
and time-windowed component throughput. It must also establish representative
longer-sequence costs and whether a larger *training* batch improves useful
throughput; the current screen fixes training batch8 and does not answer that
question. Freeze any batch/workload changes in a distinct finite engineering
scope, preserve separate denominators, and serialize competing measurement arms
inside one command/archive. Reuse the existing trainer/CheckpointStore and finish
its integration/fault tests before issuing a target handoff. The proposed roughly
two-hour soak remains a proposal requiring that concrete scope, not an executable
or admitted learning run. No new target rerun is requested by this review.

009:M004 and affected-sourceT110 close at56f6156/source735ede…; documentation-only
review changes preserve that runtime closure. Original counts return to78 complete,
18 runtime pending and15 research held/blocked. Original queued-worker/v2,
whole-pipeline/five-pair adoption, portability/operations, complete cost and
scientific gates remain separate.008:T093 stays deferred; the original negative
008:T096 remains7730/14400s. No default, learning strategy, final exposure or
multi-day training authority changes.

Review validation reruns all170 stdlib host cases successfully, verifies task
markers/counts and relative links, and confirms the source snapshot remains
exactly720 entries/source735ede… after documentation edits. Whitespace checks
pass. No new runtime code or dependency is introduced by this review.
