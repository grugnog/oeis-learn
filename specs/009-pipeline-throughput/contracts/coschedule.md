# GPU/native co-scheduling and repeatability v1

This separately registered hardware scope follows accepted GPU128 capacity archive
`d86b1a4499159acc72c78e546868bf6b33d4915d927f6484ec6010dc6b9558c5` and
native4 archive `5449f91f92665a6be42060ab3521f21ecf6363e2888f725a76e2f93de70dc7b2`.
Use `pipeline-coschedule-validation/v1`, scope `engineering-coschedule`, registry
`pipeline-coschedule-registry/v1`, phases1/2/4; work `replay128-native144/v1`,
comparisons `serial-overlap/v1` and `repeat10/v1`. Original scopes remain unchanged.

## Equal component work, one shared envelope

Each trial performs the accepted128-row/131072-supplied-BOS-token/640-boundary
hoisted FP32 eager GPU replay once, plus8 cold native4 cycles of the full18-row
native corpus (144 result rows). The fixed8-cycle count retains complete native
evidence within the existing512MiB mandatory archive allowance (about297MiB
extrapolated native public payload across twenty trials before GPU/report files).
This is a prospective component fixture with128 GPU rows and144 native fits;
its serial/overlap arms have identical work. It does not represent generated-program flow.
Each native cycle prefixes row IDs with its fixed cycle index, so its18 logical
jobs are distinct within the trial; fresh clients/caches and immutable4s request
clocks remain. No hidden truth, parameter/model/training-batch or algorithm change.

One isolated compute container owns one shared at-most16-CPU quota and resolved
90%-physical/ancestor/engine memory ceiling. One fresh supervised GPU child runs
at a time; the parent controller never performs GPU operations. Native4 uses a
bounded daemon thread only in overlap mode. No model/optimizer call occurs there.
Serial order is GPU then native; overlap starts the native group alongside the
GPU child. There are no per-worker copies of the owning-role quota. Reserve at
least16GiB plus12×weight bytes, with existing headroom and10% unreserved capacity;
require at least6 effective CPU equivalents. All descendants share the same
cgroup and mandatory CPU+driver aggregate watchdog. CPU-only reference roles
receive no device nodes; actual HIP/device/source audits remain mandatory.

A frozen CPU full-prefix replay oracle and a complete native4 reference precede
timing. The native cycle reservation is `ceil(max(8,2*reference primary interval))`,
maximum8s: an oversized/unknown reference blocks, never clamps. Each native
cycle's parent and per-request4s clocks include all queue/owner observation;
client shutdown/export must also finish within its cycle reservation.
GPU internal240s/external270s plus8×8s native reservations and66s reporting/cleanup total400s.
No component renews the fixed trial parent. A native failure stops future cycles;
GPU failure/interruption cancels the active native pool and joins within one5s
envelope. Nonquiescent work/resource breaches stop the whole invocation.

## Finite trial and selection roster

Plan all20 trial slots before dispatch. Run five fresh serial trials: cold,
excluded warmup and exactly three complete calibrations. Internal570s/external600s
per baseline. Each must verify640 GPU boundaries and all144 native rows/charges
against the frozen references, qualified shared memory and actual2/1 Torch threads.
The primary interval includes fresh GPU-process startup/work/reaping, native
clients/exports and compact trial export; image/container startup stays separate.

Before candidate timing, seal a child binding current source/manifest, weights,
replay/native references, full fixed work, complete five baseline receipts,
resources/actual threads/device, accepted evidence basis, all later slots and rules.
Use `ceil(max(400,2*max(three complete baseline primary intervals)))`, maximum570s,
with30s external setup/cleanup:430–600s external. Reserve all five comparison
and ten repeatability trials plus60s before dispatch. No clamping or extra samples.
One excluded overlap warmup then exactly two fresh AB/BA pairs compare A=serial,
B=overlap. Primary trials are ordinary apart from mandatory watchdogs; incomplete,
wrong-mode, mismatched work/reference/resource/device/threads or changed children
cannot produce favorable ratios. Select overlap only when both complete gains≥5%;
a complete negative selects serial. An incomplete comparison blocks repeatability.
This two-pair screen grants no five-pair cold/warm production adoption.

Ten named, fresh-GPU-process repeatability trials use that frozen schedule decision.
These alone add aggregate1Hz CPU/process/GPU/memory telemetry; they never enter
scored ratios. Retain all ten, including early failures and never-started work;
no replacement/trimming. This checks bounded shared-controller/role repeatability,
not persistent trainer/model/optimizer stability. Telemetry overhead is uncalibrated
for this scope. Report complete/missing reasons, all component intervals, memory
observations, drift and cleanup without inventing absent bandwidth counters.

## Export and bounds

Every trial and its8 native cycles has a durable plan/terminal ledger. Retain all
GPU boundary and unique native job/clock/debit evidence, child/decision/source
identities and explicit unstarted/partial states. Private weights/oracles/caches
remain excluded. A scope report cannot confer engineering success on an incomplete
trial. All original artifact,8GiB transient/512MiBZIP, stream/trace/report-reserve,
five-second cleanup and global/build budgets remain. A single study role has
12470s internal/12500s external, inside phase4's18000s. Its maximum twenty570s
trial reservations plus reports fit; CPU reference600s is phase1, HIP600s phase2.

One command shares preparation, images, CPU/native/HIP fixtures and one archive.
No isolated native/GPU size screen is repeated. Full generated-program throughput,
original joint adoption, Docker portability, complete research costs, persistent
trainer qualification and learning/multi-day/final-data authority remain separate.
