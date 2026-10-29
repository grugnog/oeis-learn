# Accepted batch8/16 hardware screen

The uploaded `009-replay-batching-20261024T203537Z.zip` verifies all314 listed
members plus its manifest. It exactly matches producer `d74d30e`, source
`10ea8ca7c2cc459cacdd9ec886e8ae198b9032b8bf725a98600d276c4e6912e4`
and all701 source-closure entries. Archive SHA256:
`b075bcee4348130494859e56110ad69c1c0318d0ae32d54b8e4d377b7786a5a5`.
[Exact machine-readable review](replay-batching-operator-20261024T203537Z.json).

The scope PASSES in1400.15s (23m20s):112 host tests,629 CPU tests and17 actual-HIP
tests pass. All ten GPU roles complete all160 row boundaries each, checking the
same32 rows/32768 supplied tokens. CPU reference completes160 boundaries. Stage
and role seals, row events/shapes/dtypes, calibrated child and comparison report
recompute/verify. All stages, resource/device/cleanup audits pass, with no reused
timing or retry. Producer CI40129761265/40129761192 already passed. Private raw
tensors remain excluded, so the review checks source-bound role verification.

| Pair | Batch8 verified interval | Batch16 verified interval | Reduction | Supplied tokens/s,8→16 |
| --- | ---: | ---: | ---: | ---: |
| AB | 50.5022s | 25.7545s | 49.00% | 648.8→1272.3 |
| BA | 50.3818s | 25.7169s | 48.96% | 650.4→1274.2 |

Median interval reduction48.98%; roughly1.96× supplied-token throughput. The
excluded batch16 warmup is25.8201s. Batch8 calibration yields103s component,
133s internal and163s external reservations before candidates. This is complete
equal-work forced replay with hoisted/v1 in both arms, not generation success,
five-pair production confirmation or a whole-pipeline improvement. Both admission
flags remain false and defaults remain unchanged.

Scored windows consume about1.008–1.013 CPU equivalents out of16. Sampled mean GPU
busy rises from36.00/36.10% at batch8 to44.00/44.31% at batch16. Sampled conservative
role memory upper bounds rise from1.667GiB to1.710GiB, below112.45GiB ceiling.
No memory pressure or resource breach is observed; actual Torch threads remain2/1.
Memory-busy/bandwidth is unavailable; these1Hz device samples cannot prove kernel
occupancy or one specific bottleneck. Telemetry wall overhead is still uncalibrated.

The result supports batch16 as the **experimental hardware baseline** for the next
prospective capacity comparison. It satisfies the original condition to consider
a named batch32 branch: batch16 improves complete-work throughput, and conservative
capacity estimates retain ample headroom. Larger branches still need a new frozen
scope, bounds and parity checks, not automatic expansion of this accepted run.

The user asks to combine several useful experiments to reduce handoffs. The next
hardware scope should share image/preparation/reference and one archive, sequence
timing experiments to avoid contention, and prospectively stop capacity expansion
on incomplete/parity/resource failure or an unsupported gain. Ordinary-versus-
telemetry overhead belongs in that combined scope. Deliberate native/GPU overlap
will remain separately identified rather than contaminating isolated timings.

This closes009:B004 and affected-sourceT110 for this producer only. Original counts
return to78 complete/18 runtime pending/15 research held or blocked. No duplicate
run is requested. Hardware throughput/capacity remains the priority. Learning
quality and curriculum investigations stay held; a separately decided multi-day
training run after hardware qualification will inform organic/stochastic versus
curriculum-driven learning. No such run is launched or admitted here.
