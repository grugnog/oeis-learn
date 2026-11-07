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

## Workstation result reviewed 2026-11-07

The subsequently uploaded `pac-fit-fastpaths-001/report.zip` reports PASS with
zero integrity failures. Archive SHA-256:
`7841a12ebf55f87769288969c7df1eaa754f17606b683fd8af23b27998cf9518`.
Plan identity:
`sha256:3f4fc66ddae5b8085c3d54ca7135931b536c7f3cf9a8fcb1b911958c07b9840a`.
Result identity:
`sha256:2982037104570885dce1f9b15817a08a75dbe50ed7dfa82e086f6a99fa88ad57`.
Both canonical seals and the nested role seal verify. The baseline result bytes
match the earlier uploaded report, and both runs use replay packet
`sha256:e898be57bb26fb05ca2b33d776c14872d28b4473ddc5718daefcef554b30c37e`.
All five current Rust hashes match the registered amendment. All 216 paired
attempts retain case, position, source hash, seed and four-second allowance.

| Measure | Original four-CPU/concurrency-four | Optimized four-CPU/concurrency-four |
|---|---:|---:|
| Workload seconds, three repeats | 18.533, 18.564, 18.545 | 0.130, 0.119, 0.119 |
| Median workload seconds | 18.545 | 0.119 |
| Completed fits per repeat | 61/72 | 72/72 |
| Fully completed case searches per repeat | 0/9 | 9/9 |
| Median measured CPU seconds per repeat | 31.716 | 0.154 |
| Sampled cgroup peak memory, MiB | 204.1–209.2 | 85.6–87.1 |
| Median individual fit wall time, all repeats | 156.4 ms | 2.44 ms |

The warmed, fixed-candidate workload is about 156 times faster. Warmup is
excluded from these workload timings: each optimized worker's recorded warmup
takes approximately 173–181 ms. This is not an end-to-end campaign speedup or
an estimate for arbitrary future solver queries. Both changes were applied
together, so this experiment does not separately estimate each one's effect.
Cgroup memory peaks are cumulative samples, not a sum of worker RSS.

Every repeat has the same phase counts: 42 affine UNSAT, 29 modular UNSAT and
one SMT UNSAT. The 61 previously completed controls keep their classifications;
the 11 timeouts become completed modular contradictions. There are no admitted
candidates in any of the nine case searches. All 216 measured requests reuse
the compiled runtime. No optimized repeat records CPU throttling or a worker
restart; aggregate queue wait including warmups is below 0.6 ms per repeat.

| Former timeout template, source SHA-256 prefix | Attempts per repeat | Contradiction modulus |
|---|---:|---|
| `a6c794d0a175` — shifted product plus index and offset | 7 | 2 for five attempts; 3 for two |
| `fb05771a4f41` — affine in index, nonlinear parameters | 1 | 3 |
| `44c0f08f7825` — polynomial with parameter products | 1 | 3 |
| `eda5aa5728ae` — nested polynomial with parameter products | 1 | 5 |
| `b1e7f8cbd21e` — polynomial with parameter products | 1 | 3 |

An independent Python integer-AST interpreter additionally enumerated every
parameter residue assignment at the reported modulus against all 20 visible
terms for each of the 11 former timeouts. None has a residue witness. The
enumerations contain 8–625 assignments each; arithmetic uses exact Python
integers and preserves the source's index-dependent loop counts. This check
uses the exported source/prefix rather than trusting the fitter's UNSAT flag.
It supports rejection of these candidates, not correctness of a target program
or a general proof that modular checks will settle every future timeout.

The host command requires the qualification-image build and CPU/native checks
before emitting this benchmark. The uploaded ZIP contains the benchmark and
plan, not the underlying test transcripts; native test success is inferred from
that guarded command path rather than independently reviewed test receipts.
This CPU-only result supplies no HIP utilization or training-throughput evidence.

The original full probe result remains immutable and INCOMPLETE. Completing
these nine searches could address the numerical gap from 235 to the 244/256
gate, but this replay neither regenerates the candidate pools nor scores all
controls. The next step is a new full training-only B/D probe validation with
the existing checkpoints, current CPU/HIP qualification and unchanged selector:

```bash
cd /home/owen/workspace/oies-learn
python3 -S scripts/foundation_run.py --engine podman pac-selector-validation probes \
  --campaign run-state/generalization/pac-012-bcd-r1-002 \
  --output run-state/generalization/pac-selector-probes-002
```

Run on `012-pac-native-fit-fastpaths`, with a fresh output directory, then share
`run-state/generalization/pac-selector-probes-002/report.zip`. This performs no
retraining. Review its actual completion and paired outcomes before separately
freezing any fresh structural comparison. This diagnostic alone does not
justify changing the selector, adopting D, combining arms or starting HPO.
