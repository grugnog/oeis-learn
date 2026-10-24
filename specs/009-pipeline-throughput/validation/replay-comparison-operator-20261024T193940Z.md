# Accepted component comparison and hardware priority

`009-replay-comparison-20261024T193940Z.zip` verifies all314 listed members plus
its manifest and matches producer `ac41e4d`, source `c2e5a59d84e35f036e32108f7b45fc7d74ca3d115a1e76a2a58d085a15dfd225`
and all697 closure entries exactly. Archive SHA256:
`fdaa3c3f84f08b5cbddd19858929a4818bd3a0db0ae6e8a3b6672431e45aa600`.
[Machine-readable review](replay-comparison-operator-20261024T193940Z.json).

The registered scope PASSES in1158.45s (19m18s):106 host tests,622 CPU tests and16
actual-HIP tests pass, without runtime failures/errors/skips. The repaired
real-thread fixture passes. The visible-only CPU reference and all ten GPU roles
complete all12 boundaries each, with qualified resources and matching actual
two intraop/one interop threads. Every stage passes; no retry or reused timing.
Both already-completed producer CI workflows40125739381/40125739421 pass.

## Measured isolated intervention

| Exploratory pair | Baseline verified interval | Hoisted verified interval | Reduction |
| --- | ---: | ---: | ---: |
| AB | 29.5459s | 26.4318s | 10.54% |
| BA | 29.5577s | 26.2222s | 11.28% |

Median reduction is10.91%, range10.54–11.28%. Candidate warmup26.1709s is excluded.
The same complete batch1/8/mixed forced replay is verified against the same CPU
reference; eager FP32, weights, device/memory/threads and measurement mode match.
The three fresh baseline calibrations establish60s component/90s internal/120s
external reservations before candidates. Child, baseline and comparison reports
recompute exactly from the immutable receipts. Boundary plans/events/terminal
states and output shapes/dtypes agree. Raw private tensors are excluded, so the
review checks source-bound verification rather than recomputing logits itself.

This supports carrying `hoisted/v1` forward as an **experimental component**.
It is not five-pair production confirmation, generated-program equivalence,
mandatory-stratum nonregression or a whole-pipeline improvement. Both admission
flags remain false, and serial production defaults remain unchanged.

## CPU/GPU/memory observations

The four scored roles use approximately1.007 CPU equivalents during sampled
replay windows, with16 equivalents allowed. Sampled mean GPU busy is31.77/31.63%
for baseline and29.56/29.85% for hoisted. Faster completion despite a lower busy
percentage reinforces that elapsed verified work is the objective. These1Hz
device-level samples cover batch1/8/mixed work together and cannot identify kernel
occupancy, per-batch utilization or pure decoder device time. Memory-busy/bandwidth
is unavailable. Collector CPU time is reported, but wall-time observation overhead
is not calibrated and cannot be inferred from collector CPU alone.

Sampled conservative role memory upper bounds are approximately1.659GiB, against
112.452GiB enforced aggregate ceiling. Machine-wide conservative upper bounds
remain around9GiB. No memory/resource breach is observed. This is ample capacity
for a separately bounded larger-batch investigation, not a reason to allocate
unused memory. The observations suggest testing launch/bookkeeping amortization
and native CPU overlap; they do not prove a single bottleneck.

## User-directed execution priority

The user's latest instruction puts CPU/GPU/memory usage and performance first.
Hold learning-performance investigations and curriculum changes. Hardware work
can use frozen diagnostic weights and bounded disposable-update performance
fixtures without seeking a new learning-success result.

Once a configuration makes effective use of the hardware and passes correctness,
containment, checkpoint/recovery and sustained performance checks, prepare a
separate finite multi-day training decision. That future run is intended to inform
whether to continue organic/stochastic learning or move to curriculum-driven
learning, with pre007 work as context. No multi-day run or curriculum intervention
is launched or admitted by this review. Freeze its recipe, cost, checkpoints,
evaluation cadence and stop/decision rules at that later decision point; do not
weaken existing numerical, provenance or final-data protections.

This closes009:C004 and affected-sourceT110 for this producer only. Original task
counts return to78 complete/18 runtime-evidence pending/15 research held or blocked.
Full generation/native/joint evidence, operations/portability and complete cost
gates remain separate. Existing negatives and the earlier failure archive remain
unchanged. No duplicate run of this accepted comparison is requested.
