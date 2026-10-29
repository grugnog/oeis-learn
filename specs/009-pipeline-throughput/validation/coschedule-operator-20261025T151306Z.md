# Accepted co-scheduling collection and hardware convergence

The operator archive `009-coschedule-20261025T151306Z.zip` passes its registered
component scope at producer `5014881dba04e1c22da6784e3fcd00ec74e5a365`.
The second uploaded copy has identical bytes; it is one experiment, not a second
replication. [Machine-readable review](coschedule-operator-20261025T151306Z.json)
binds archive SHA-256 `ce32629c46558d71e468bb646a4bb4ee9ebb291c7ae46cf7040f72a6938d909b`,
21,132,463 compressed bytes, 309,244,544 uncompressed bytes and all4,588 listed
members plus the bundle manifest. There are no omissions. Actual bundle verification,
all receipt seals/member hashes, exact713-entry source closure and source
`eb57cb0b2279dbea1430f9a8dd88e2e548c60417aaef36a996563412d40ac02a` match the checkout.

## Complete work and qualification

154 host tests,674 CPU tests plus25 subtests (699 JUnit cases), and19 actual-HIP
tests pass. There are zero mandatory skips; one GPU-only CPU-suite test is
deselected. The real native-cycle and HIP/native coexistence fixtures execute.
All9 outer stages and all20 study slots pass. Every trial completes640 GPU
boundaries against the frozen CPU oracle at the registered FP32 tolerance and
eight complete native4 cycles/144 distinct native jobs. Overall:12,800 GPU boundary
verifications,160 native cycles,2,880 strict job debits and11,520 clock events.
Output tensor hashes are retained; tolerant numerical parity does not assert
bitwise identity across fresh HIP processes.

Every native semantic result agrees with the reference after normalizing only the
fixed cycle-ID prefix. All admission/dispatch/native-completion/owner-observation
events agree with durable job records. Four-second request clocks, eight-second
cycle parents, previous-wave drains and sequential cycle admission verify. Cycle
primary intervals range1.1165–2.9476s; maximum admitted-to-dispatch delay11.81ms.
All fresh GPU processes complete inside270s. The same registered work/resources/
device/actual Torch2/1 threads qualify both schedules. Reference identities, five
baseline receipts, baseline-only400s child and AB/BA decision recompute exactly.
No favorable subset, renewed deadline or replacement sample is used.

## Performance and repeatability

| Pair | Serial seconds | Overlap seconds | Elapsed reduction |
| --- | ---: | ---: | ---: |
| AB | 50.215814 | 39.309418 | 21.7190% |
| BA | 52.212379 | 39.164429 | 24.9901% |

Both gains exceed the prospectively fixed5% exploratory threshold, so overlap is
the experimental schedule. Median paired reduction23.3546%; paired speedups about
1.28–1.33×. These are component-fixture results, not useful generated-program
throughput or the original five-pair production adoption.

All ten separately observed overlap repeats pass. Primary intervals span
39.019378–39.504403s, median39.263571s. Last/first ratio1.003139 (+0.314%); complete
range about1.24% of the median. This is about6.5 minutes of repeated measured work
with fresh GPU processes, clients and caches inside the shared controller. It
does not establish multi-hour persistent model/optimizer stability.

The observed full-trial GPU-busy means span89.56–91.20% (median91.1375%), including
startup/tails; whole-cgroup CPU intervals span1.975–2.001 equivalents out of16.
One-Hz samples miss no scheduled captures. Telemetry overhead is uncalibrated in
this scope; repeats are excluded from scored pairs. GPU/memory bandwidth is not
available and is not inferred from busy percentage. Low average CPU utilization
is consistent with native work finishing during the GPU interval. Raising workers
or manufacturing CPU work is not justified by this result.

## Memory, containment and costs

Actual container inspection enforces one16-CPU quota and120,744,574,976-byte
(112.45GiB) aggregate ceiling. The owning-role watchdog reports qualified accounting,
866 samples, no breach and4,748,537,856-byte (4.42GiB) role upper peak; sampled
machine upper peak is11.33GiB. Parent/child reports cover the same cgroup/driver:
they must not be added. All isolation audits pass and owning-role cleanup takes
0.2153s inside its one5s allowance. Role time865.04s, host-through-cleanup865.41s;
full invocation1,888.51s (31.48min), consistent with the prior planning range.

Cgroup memory increases across repeats: start samples0.886→1.491GiB, while
controller RSS is roughly39–44MiB. Each trial retains evidence and copies frozen
weights/oracles for its fresh GPU child; retained private files and file cache are
plausible contributors. No `memory.stat` breakdown was captured, so this is not
a proven explanation or evidence of persistent-state leak freedom. The next useful
soak must distinguish anonymous, file-cache, GPU residency and bounded retained
artifacts. Do not extrapolate fixture memory linearly into a multi-day trainer.

Completed producer CI was inspected once: Foundation run40170350454 passes; in
Container/native run40170350489 all15 other reported checks succeed and the separate
`podman-build` job is cancelled. There are no reported failed checks to repair.
Cancellation remains cancellation, not successful build evidence. Actual target
Podman qualification passes; original complete portability/operations gates remain
scoped separately. No newly queued PR CI is awaited.

## Disposition and next experiment

Close009:O004 and affected-sourceT110 at this exact producer. Original task counts
are78 complete/18 runtime pending/15 research held or blocked. Original full-generation,
queued/v2 native, five-pair adoption, operations, complete costs and scientific gates
are not closed by component co-scheduling. Existing historical negatives and all
accepted earlier producer evidence remain intact.

We can now converge on a hardware configuration rather than expand isolated size,
worker or overlap sweeps. [Configuration and remaining integration work](hardware-convergence.md)
describe the bridge to one useful real-pipeline/persistent-state qualification.
No repeat of this accepted co-scheduling experiment is requested. No production
default, learning-quality/curriculum change or multi-day training is launched.
