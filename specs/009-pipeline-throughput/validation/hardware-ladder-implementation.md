Target status: [accepted archive](hardware-ladder-operator-20261024T220811Z.md) at adda1b6. The command below is retained as historical/reproducible documentation; no duplicate run is requested.009:H004 and affected-sourceT110 close for this source.

# Combined hardware experiments: one-command handoff

The [batch8/16 archive](replay-batching-operator-20261024T203537Z.md) is accepted:
629 CPU/17 HIP tests and all roles pass; batch16 reduces equal-work replay interval
48.98% median and nearly doubles supplied-token throughput. The user requests
combining further useful experiments. The next [registered contract](../contracts/hardware-ladder.md)
implements four investigations in one invocation, sharing images, prerequisites,
visible-only preparation, CPU reference and archive. Timings remain serial to
avoid contention; deliberate native/GPU overlap is separate future work.

| Investigation | Frozen behavior |
| --- | --- |
| Observation overhead | Exactly one fresh ordinary then one aggregate1Hz batch16 role; >5% elapsed overhead selects ordinary primary timing |
| Batch32 versus16 | One excluded32 warmup and two fresh AB/BA pairs after complete baseline calibration/capacity admission |
| Batch64 versus16 | Same finite pattern, only if32 supports the predeclared progression rule and capacity fits |
| Batch128 versus16 | Same pattern, only if64 supports progression and capacity fits; no larger branch exists |

## Equal work, limits and stopping

All arms use hoisted/v1, eager FP32 and the same frozen checkpoint/reference.
The32 unique visible input rows are cyclically mapped into128 logical rows with
**physically replicated encoder memories and independent KV caches**. This is a
capacity fixture, not128 distinct prompts. Every arm performs131072 supplied BOS
tokens and verifies640 row boundaries through position1024 against the CPU
full-prefix oracle for each corresponding unique input. No hidden truth, model
update, training-batch/model-size change or learning-quality investigation.

The overhead pair freezes a common primary mode before one batch16 cold role,
excluded warmup and three complete calibrations. Ordinary still retains mandatory
memory/device/containment checks; per-candidate sampled utilization is unavailable
when ordinary fallback is selected, while the separate observed16 role remains.
The single overhead pair is a bounded mode decision, not population confidence or
an estimate of zero watchdog overhead. Role/startup/cleanup costs stay separate.

Each branch seals source/weights/reference/workload, prior accepted evidence,
baseline-only calibrated reservation, selected mode, actual threads/device/memory,
allocation/headroom, five-role roster and thresholds before candidate timing.
Prediction includes at least8GiB plus12×weight bytes or batch-ratio-scaled baseline
memory, whichever is larger, plus headroom, leaving10% of the ceiling unreserved.
Dispatch rechecks current availability and exact weight allocation. Every role
uses fresh weights/caches; no successful exploratory sample becomes confirmation.

Expansion requires both gains versus16 ≥5% and candidate median ≥5% faster than
the preceding eligible size. A complete unsupported gain ends expansion; later
role ledgers say NOT_SELECTED. Failure/partial/unknown capacity blocks dependent
work. Each branch plans all five role ledgers before dispatch; changed children,
resource breach and interruption preserve all terminal/unstarted dispositions.
No adaptive samples/retries or growth beyond128. Experimental selection uses
fastest qualified median, preferring smaller sizes within5%; production defaults
and both adoption/admission flags stay unchanged/false.

## Available verification

**123 host stdlib tests pass.** Full simulated graph tests cover all branches,
ordinary fallback, progression stop, nested failure, changed child before scored
work, exact weight/option rejection, global memory stop/interruption, parent/child
receipt hashes and one verified archive. Preparation reads no hidden truth
and retains32 visible rows while expanding the explicit128-row plan. Partial or
mismatched roles cannot produce favorable-subset timing ratios.

The actual public stdlib CLI exits3 for the missing engine, retains all three
undispatched branch rosters as UNSTARTED and verifies a51-listed-member non-admitting
archive. [Local evidence](hardware-ladder-local-evidence.json) records its exact
source/manifest/archive hashes. Compilation, whitespace and JSON/doc/ledger checks
pass. The007 package binding is refreshed with all68 dispositions and incomplete
foundation acceptance preserved. Existing validation/replay/batching manifest
identities remain unchanged.

New pinned CPU/HIP fixtures construct real serialized128-row reference tensors,
verify physical replication and all batch16/32/64/128 outputs, exercise ordinary
and observed GPU execution, actual2/1 threads, changed reference and finite
deadlines. They use a fresh bounded process; actual HIP is mandatory for GPU
claims. These fixtures are authored/registered, **not executed locally** because
Torch/pytest/engines are unavailable. No host dependencies were installed.

## One target command

After the code work above, run from the repository root:

```bash
git pull --ff-only origin 009-pipeline-throughput
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines selected \
  --manifest configs/foundation/pipeline_hardware_ladder.json \
  --run tmp-validation/008-phase5-diverse32-20261021T160815Z/learnability-run \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-hardware-ladder-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use the same compatible parent/input locations unless moved. Return the single
sibling ZIP, including non-PASS results. No separate build/test/profile/archive
launches or manual decisions between these preregistered experiments. This does
not repeat the accepted32-row batch8/16 scope. Podman-only selection leaves Docker
portability pending.

Planning estimate: roughly40–60 minutes with warm dependency caches if all branches
run and larger batches scale reasonably. This extrapolates the observed25.7s per
32-row batch16 work to the128-row fixture and adds prerequisites; it is not a new
calibration or guaranteed total. The command establishes its actual bounds first.
Reference/baseline role caps are600s external/570s internal. Derived branch child
external bounds are at most340s, with separate30s internal setup/30s outer startup
and cleanup; branch caps1800s include all five roles and report reserves. Phase2
declares9810s, below10800s. Existing14400s-per-engine build,86400s global, archive/
cleanup/disk/memory ceilings remain; no phase borrowing or budget increase.

## Hardware-first disposition

Accepted009:B004 remains complete at its recorded source. T110 reopens for this
new affected source: original counts77 complete/19 runtime pending/15 research
held or blocked. Combined hardware implementation tasks are complete; target
collection/review remains pending. Further native CPU scaling, deliberate overlap,
full-pipeline and sustained operations remain hardware work. Learning quality and
curriculum changes stay held. A separately planned multi-day training run after
hardware qualification will inform organic/stochastic versus curriculum-driven
learning; this command launches none. PR CI is not awaited.
