# Fixed-candidate native fitting diagnostics

The completed `pac-selector-probes-001` run has D 235/256 completed primary
cases, below the frozen 244 gate. Its nine no-selection fitting-timeout cases
contain 72 original attempts: 11 timeouts and 61 completed `unsat_in_scope`
searches. Eight additional cases hit the decoder token cap and four encounter a
grammar dead end; those cases are outside this fit-only diagnostic.

This command needs the existing workstation probe directory. It reads no model
weights, performs no generation/training, and never mounts full-horizon targets
or assessment data in the fitter. Existing probe results and freeze eligibility
remain unchanged, even if all replayed searches complete.

## Run

Check out `012-pac-fit-diagnostics`, then run from the repository root:

```bash
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation fit-diagnostics \
  --probes run-state/generalization/pac-selector-probes-001 \
  --output run-state/generalization/pac-fit-diagnostics-001 \
  --compare-cpus 8 --repetitions 3
```

A fresh output directory is required. Docker is also supported. The command builds
and inspects the current qualification image and runs canonical CPU/native checks,
including the new diagnostic tests. It does not build a GPU image or run HIP.
After those checks it freezes a single diagnostic matrix:

| Order | CPU quota | Concurrent fits | Native worker slots | Container memory |
|---|---:|---:|---:|---:|
| Isolated original-quota baseline | 4 | 1 | 4 | 16 GiB |
| Concurrent original-quota baseline | 4 | 4 | 4 | 16 GiB |
| Separate larger-quota comparison | 8 | 4 | 4 | 16 GiB |

`--compare-cpus` selects the last quota (5–32, subject to the engine's available
CPUs). These explicitly frozen quotas override the global CPU/memory options.
Repetitions are bounded to 1–5. Baseline repetitions finish before resource
comparisons begin. Run on an otherwise idle workstation for interpretable timing.

## What is held fixed

The command verifies the original sealed probe plan, result, scoring file and
recorded generation artifact hashes. It selects all no-selection primary cases
whose recorded reason is `resource_exhausted:timeout` and whose eight generations
completed. It extracts every fit in those portfolios, including the completed
searches as controls. No choice is made using hidden continuation values.

Original source bytes/hashes, per-attempt seeds (the original native modulo-2^32
mapping), and service allowances survive extraction. Allowances are taken from
`fitting.clock.deadline - fitting.clock.admitted`, bounded by the original four
seconds. The unchanged production client continues to allocate its own native
solver and reference/response reserves; no timeout is raised or refreshed.

The original fitter/reference/Rust source and build-input closure must match the
current checkout. A changed core stops the command before any benchmark. New
host/preparation/observer adapters are allowed; this is a new diagnostic source
identity with current CPU/native evidence, not a replacement original qualification.

Each repetition gets a new client, worker processes and cache root. Four recorded
constant-program warmups start the existing worker slots before fit measurements.
The fixed case/attempt order is repeated; concurrent waves assign distinct slots.
This controls warmup/cache effects without reconstructing the original mixed GPU
and solver schedule. It is not an exact replay of original whole-portfolio timing.

## Output and interpretation

Upload `run-state/generalization/pac-fit-diagnostics-001/report.zip` afterward.
It includes the frozen plan, summary, and all repetitions' diagnostic records:

- completion/outcome per original case and attempt; admitted source hashes only;
- original versus replayed service/solver timing and bounded search outcomes;
- observed native transport wall time and worker queue-wait deltas;
- worker restarts, cumulative sampled RSS/thread metrics and explicit unavailable
  measurements;
- cgroup-v2 CPU quota/usage/throttling counters and memory counters before/after
  each measured workload, when the container exposes them;
- native runtime and independent reference identities for every repetition.

Transport wall time includes worker startup/restart, protocol work and native
execution. The native response's `elapsed_seconds` and `timing` describe a
separate inner measurement; do not subtract unrelated clocks or interpret
aggregate queue counters as per-fit CPU time. No GPU utilization is measured in
this CPU-only workload. Resource sampling is outside the measured fit waves.

If isolated fitting repeatedly times out, intrinsic search difficulty or the
frozen search allowance is a stronger explanation than concurrent worker load.
If concurrent four-CPU completion falls and the eight-CPU comparison recovers it,
contention is a plausible explanation. Repeated timings/cgroup throttling provide
supporting evidence, not proof of causality from one run. Completed original
control searches that change to another completed classification, or independent
verification disagreements, fail the diagnostic with preserved evidence.

Every report says `scientific_status: DIAGNOSTIC_ONLY` and
`freeze_eligible: false`. Resolving these searches does not rewrite 235/256 as
244/256 or authorize structural execution. An eventual engineering change needs
its own qualification and a new complete probe validation before a fresh freeze.
Failure bundles preserve the actual exception/traceback and point at original
role output; original probe files remain untouched.

## Local verification boundary

Local counterexamples cover source/seed/allowance tampering, hidden-value
rejection, full eight-attempt extraction, core closure changes, CPU-only disjoint
mounts, resource matrix ordering, retained original bytes and failure authority.
A supervised subprocess fixture exercises the real worker transport with the
observer; synthetic clients exercise unchanged limits/seeds and concurrent replay.
Actual Z3/native candidate results and workstation resource measurements remain
pending operator execution.


## Explicit implementation comparison

The completed baseline reproduced all 11 timeouts and showed no benefit from
eight versus four CPU quota. The separately registered
[native fit fast paths](native-fit-fastpaths.md) benchmark runtime reuse and
modular contradiction proofs against that recorded baseline. The original
same-core command remains strict; the engineering comparison requires its
explicit amendment flag and the original baseline directory.
