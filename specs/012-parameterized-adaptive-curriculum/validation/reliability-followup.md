# Reliability repair and training-only scheduling diagnostic

The follow-up PR stack is merged into #63. Subsequent changes belong on
`012-parameterized-adaptive-curriculum`.

The immediate engineering work addresses the observed grammar dead ends and
fitting timeouts before HPO or another scientific comparison. Historical scores
and plans retain their original identities and completion status.

## Repairs

The typed grammar now excludes operators, assignment destinations and statement
openings whose required children cannot fit within the nesting limit. Bigint
leaves and legal depth-boundary programs remain available. Synthetic and sampled
prefix regressions exercise the distinction between grammar feasibility and the
separate token budget; the 1,024-token limit is unchanged.

The native necessary-condition checker adds modulus 11. When direct residue
enumeration exceeds the assignment cap, it can replace affine parameter forms
at nonlinear boundaries with independent residue variables. This enlarges the
possible assignment space, so only an exhaustive contradiction supplies UNSAT.
It never supplies a positive assignment. Identical forms share variables across
visible indices. Deadline, node, parameter and total-work limits remain enforced;
inconclusive work falls through to the existing solver. Certificate replay
reconstructs the same necessary check from the original symbolic problem.

The historical fitter amendment remains unchanged. The separately named
`native-fit-reliability/v2` records the current core hashes. Current source must
be qualified again; these repairs do not permit historical probe reuse or
retroactive completion of the observed structural result.

The shared CI packaging failure came from an incomplete copied CLI fixture.
It now copies the actual first-party package and still runs under `python -S`
with an empty PATH. This preserves the no-engine/no-third-party packaging check.

## One workstation command

From the repository root, update #63's branch and run:

```bash
git fetch origin &&
git switch 012-parameterized-adaptive-curriculum &&
git pull --ff-only origin 012-parameterized-adaptive-curriculum &&
python3 -S scripts/foundation_run.py --engine podman pac-reliability \
  --campaign run-state/generalization/pac-012-bcd-r1-002 \
  --output run-state/generalization/pac-reliability-001
```

Docker is also supported. A fresh output directory is required. No host toolchain
or additional workstation dependencies are introduced.

The command first builds the current images (including mandatory Rust tests),
requires CPU/native and actual-HIP qualification, then validates both original
final B/D checkpoints on all 256 training probes with all seven controls and
both selectors. This stage consumes no training tokens and follows the existing
244/256 technical gate. Probe accuracy does not select a winning arm.

It then compares the existing drain schedule with the registered matched
schedule on the first 16 training-probe IDs in sorted order. The subset rule and
A–B–B–A order are recorded before probe outcomes. Each arm receives two runs per
schedule, with the same checkpoints, recipe, attempts, seeds and 60-second
portfolio limits. The drain schedule retains four-row batches; the matched
schedule permits the recipe's larger batches and native/decode overlap. Bank
generation is excluded from this scheduling benchmark. Each role gets a fresh
model/client/cache, so model loading is included in workload wall time.

The benchmark is a scheduling counterfactual. Batch numerics and deadline effects
can change proposals or completion; compare attempt-behavior identities and
failure counts alongside time. Shared decode durations are counted once per
batch, overlapping service intervals are merged, and unknown decode timing stays
explicitly unavailable. The benchmark confers no scientific advancement or
automatic schedule adoption, even if the probe technical gate passes.

Both schedules use the existing bounded telemetry recorder. It records collector
overhead, CPU/cgroup counters, process observations and accessible selected-GPU
sysfs sensors, plus GPU allocator peaks. Unavailable sensors remain explicit;
allocator peaks are not GPU utilization. Role startup/cleanup logs and measured
portfolio spans remain separate from model-load and full-role wall time.

## One evidence archive

Share the printed `diagnostics.zip`. It includes a hash/size inventory, intent
and benchmark plan, complete probe scoring reasons, visible prompts, generated
attempts/candidates, selector observations, clocks, benchmark telemetry, and
bounded qualification/control diagnostics. Image-build logs are included so a
native build failure can be diagnosed from the same archive. Model weights,
full-horizon targets, private caches and original corpora are excluded.

Collection uses a closed file inventory and a 512-MiB uncompressed metadata cap.
Missing files and bounded control-copy omissions are recorded. Failure paths
attempt the same collection and preserve the original traceback. A collection
failure cannot print a successful report path. The archive is for workstation
review; only key learnings should be committed to the repo.

## Scientific work after review

First establish whether the repaired probe panel passes its technical gate and
whether larger batches/overlap improve throughput without unacceptable behavior
or completion changes. Use fixed candidate diagnostics for any remaining solver
failures, preserving source, visible inputs, seeds and service allowances.

After that review, separately freeze a prospective B/D/template comparison with
multiple paired training seeds and a cohort that excludes every already observed
structural cohort. Check sparse-family availability before spending training
compute. Assessment remains untouched. A combined learned/template portfolio
needs its own frozen visible-only rule and total budget. HPO remains deferred
until the endpoint is dependable and the single-seed D signal is confirmed.

This command implements the engineering and training-only diagnostic stage. It
does not start research retraining or reuse the already observed cohort as fresh
confirmation data. Those later experiments depend on the diagnostic review.

## Local verification

The CPU regression suite passes 959 checks, with 68 native/hardware-dependent
skips. The affected selector/generalization suite passes 456 checks, with one
installed-native-image skip and five passing subtests. Additional grammar
boundary, timing/accounting, telemetry, orchestration, failure-collection and
real wheel-packaging checks pass. The reported CI packaging failure was
reproduced before repair. Current implementation/core bindings, CLI help,
artifact consistency, compile checks and whitespace pass. Rust compilation and
actual HIP execution remain mandatory workstation gates; CI was not awaited.
