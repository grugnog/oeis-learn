# Native fit engineering comparison

The completed fixed-candidate diagnostic reproduced all 11 SMT timeouts in all
nine repetitions: 61/72 fits completed under isolated four-CPU, concurrent
four-CPU and concurrent eight-CPU configurations. Median workload times were
31.05, 18.55 and 18.57 seconds. CPU throttling, queue waits, memory pressure and
worker restarts do not explain these particular failures. Those results remain
historical diagnostic evidence; D's original probe gate remains 235/256.

## Changes to benchmark

The native body-fit dispatcher previously constructed a fresh `BigintRuntime`
(engine plus compiled immutable BigInt guest) before every call to `fit::fit`.
That work was outside the fitter's inner elapsed timer and occurred even when
no candidate was produced. The worker's installed context now reuses its
immutable compiled runtime, as its execute path already did. It also caches the
immutable guest digest. Candidate modules, stores, instances, clocks, input
values, assignments and fit results are not cached by this change. Original
execution resource checks and fresh per-candidate state remain in effect.

`dispatch_timing` exposes validation, installed load, runtime identity, runtime
setup/reuse and total dispatch timing. This identifies setup overhead separately
from the native fit clock. Reuse reduces actual setup work, so the solver may
receive more of the same unchanged remaining request budget; this is an explicit
engineering counterfactual, not an exact reproduction of old timing.

Before SMT, rank-deficient symbolic problems receive a necessary modular check.
Every admitted integer assignment must also satisfy each output equation modulo
2, 3, 5 and 7. The checker exhausts residue assignments for eligible small
moduli; if none match, it can soundly return UNSAT. A residue witness is never an
integer solution or an admission. Inconclusive checks fall through to the
existing SMT solver, using the same remaining deadline. The checker does not
special-case benchmark programs, family labels or hidden continuation values.

Work is bounded by 4,096 residue assignments per modulus and 2,000,000 node
steps across checks, with cooperative checks of the original search deadline.
The arithmetic DAG contains only exact integer constants, parameters, addition,
subtraction and multiplication. Parameter-independent control was already
resolved by the existing symbolic builder. Magnitude and checked-overflow guards
are relaxed only for this necessary test: inconsistency in the larger arithmetic
set also implies inconsistency in the guarded set. Actual positive assignments
still go through unchanged native production execution and independent admission.

UNSAT certificates explicitly name `residue-exhaustion/v1/mod-P`. Replay rebuilds
the symbolic problem and exhausts the named modulus anew, binding source,
visible prompt, numeric resources and constraints. Forged, inconclusive, expired
or partially exhausted checks cannot become a verified UNSAT result.

## Workstation command

After checking out `012-pac-native-fit-fastpaths`, run from the repository root:

```bash
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation fit-diagnostics \
  --probes run-state/generalization/pac-selector-probes-001 \
  --baseline run-state/generalization/pac-fit-diagnostics-001 \
  --core-policy native-fit-fastpaths/v1 \
  --output run-state/generalization/pac-fit-fastpaths-001 \
  --repetitions 3
```

This is CPU-only. It builds the current qualification image, including mandatory
Rust core tests, then runs the current CPU/native suite before measuring anything.
Existing dependency locks remain unchanged. No host toolchain or checkpoint
upload is needed. Use an otherwise idle workstation, as for the baseline.

The normal same-core diagnostic still rejects core changes. Only this explicitly
named amendment permits the five exact parent/current Rust file hashes recorded
in `src/oeis_learn/generalization/policies/fit-core-amendment.json`. Unlisted or
stale changes fail closed. `--baseline` and `--core-policy` must occur together.
The prior baseline must be the complete original diagnostic with the same probe,
scoring/generation references, recipe, selected cases, repetitions and fixed
four-CPU/four-worker/16-GiB configuration. The full replay packet identity must
match before native benchmarking starts.

The command runs only the optimized four-CPU/four-concurrent-fit configuration;
it compares it with the already recorded equivalent baseline. All 72 original
sources, seeds and per-attempt service allowances remain fixed. Neither the
outer four-second allowance nor the existing native/production/response reserve
policy is raised. Warmup and fresh cache/process rules remain explicit.

Upload `run-state/generalization/pac-fit-fastpaths-001/report.zip`. The ZIP
includes the original versus new timings/completion counts, dispatch/native
phase data, and all five original timeout templates with the nine 20-term
visible prefixes. It contains no hidden continuation terms, original target
programs or model weights. These visible inputs support inspecting all five
queries without another manual packaging step.

The report remains `DIAGNOSTIC_ONLY` / `freeze_eligible: false`. Even if the
unknown fits become completed UNSAT searches, this does not rewrite the old
probe result. Any advancement requires a new full training-probe validation
under the qualified implementation, followed by a separate structural freeze.

## Verification boundary

Python tests cover explicit hash-bound amendments, rejection of other core
changes, baseline/input/packet binding, CPU-only role mounts, retained original
files, and diagnostic authority. Native tests cover modular contradictions,
real-solution counterexamples, inconclusive modular witnesses, work/deadline
exhaustion, certificate replay/tampering, and immutable runtime reuse. The
qualification-image integration test exercises a real native worker, a replayed
modular certificate and independent admission of a positive square solution.

This workspace lacks Rust and a container engine. Local Python checks can run,
but native compilation, Rust tests, real-worker integration and speedup results
must be established by the command's qualification build on the workstation.
No workstation result or HIP evidence is claimed by implementation completion.
