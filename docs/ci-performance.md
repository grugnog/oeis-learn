# Build and test performance

Baseline: successful PR #48 head `541b58246ff13b234966f10e4edad8956ac4b19d`.
Measurements come from workflow logs for runs
[39052414322](https://github.com/grugnog/oeis-learn/actions/runs/39052414322) and
[39052414352](https://github.com/grugnog/oeis-learn/actions/runs/39052414352).
Times below run from a step's start to the next step's start, rounded to seconds;
they include small amounts of step overhead and reflect one runner sample.

| Work | Observed time |
| --- | --- |
| Docker build context | 5.55 MB, about 0.1 seconds to transfer |
| Cold qualification image builds | 2m28s to 3m32s |
| Rust release compilation within those builds | 1m14s to 1m56s |
| CPU dependency setup | 36s |
| Main CPU regression suite | 2m49s (934 passed, 60 skipped) |
| CPU input diagnostics, after regressions | 1m51s |
| CPU quickstart/replay, after input diagnostics | 2m02s |
| Installed-wheel acceptance and accompanying gate/test groups | 10m22s |
| Public Docker/Podman CLI validation | 8m44s to 8m54s |

The build context is already small on a clean runner. Missing exclusions for
`.role-venv`, wheel/evidence outputs, and local tooling caches can still make
local or repeated builds much larger and invalidate `COPY` layers unnecessarily.

The CPU checks originally serialized three independent workloads. Input
diagnostics and quickstart/replay use distinct output directories and do not
consume the regression suite's outputs; they can run as separate jobs while
retaining all commands and artifacts. This improves feedback from that branch
of CI; the longer wheel and portable-validation jobs can still dominate total
PR latency.

Persistent download/build caches can reduce repeated dependency setup and build
work. Caches are accelerators only: install/build/test commands still execute,
there is no cached passing-test result, and Podman still builds with Podman.
GitHub cache visibility restricts reuse between PR refs; without a default-branch
cache-warming run, expect the main benefit on subsequent commits to the same PR.
No additional default-branch builds are introduced in this small change.

The entire repository is copied before native compilation, so Python/test edits
can still invalidate the native build layer. Separating native compilation from
Python packaging may help later, but requires checking the source-distribution,
ABI, and native build-identity contracts. It is not part of this quick pass.

Per-test duration output will make the next successful run more useful for
choosing further test splits. The 10m22s installed-wheel step includes ordered
gate/evidence production and rebuilt-wheel checks; it should not be split
without preserving those dependencies. No speedup estimate here is a measured
result of the new configuration.
