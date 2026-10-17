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
of CI; portable-validation jobs can still dominate total PR latency. Training/replay integration previously run only by the wheel job
also runs as an independent CPU acceptance matrix entry.

The container jobs now depend on one shared qualification-image build per
workflow run. Same-repository PRs push that image to GHCR; Docker, Podman and
native qualification pull the resulting **digest**, not a moving tag. The loader
also checks that the image's source-revision label matches the tested checkout.
The jobs still execute their tests; there is no cached passing-test result.

Fork and Dependabot PRs do not publish or need registry write credentials. Their
producer exports the freshly built image as a short-lived workflow artifact,
which both engines load. A missing artifact, failed pull, or revision mismatch
fails the job rather than silently using an older image. Consumer jobs have only
package-read permission and remove registry login credentials after pulling.

A separate `podman-build` job retains native Podman build coverage when the
Dockerfile, ignore file, shared installation scripts, Python build/dependency
manifests, or native Rust crate change. Ordinary Python/test-only PRs no longer
compile the same image again with Podman. This PR changes `.dockerignore`, so it
still runs that separate build check; the regular Podman test job pulls the
shared image.

Merged changes publish a qualification image and refresh its registry build
cache through `.github/workflows/container-images.yml`. PR builds import that
cache as well as their PR-scoped GitHub Actions cache, but cannot overwrite the
shared default-branch cache. This gives new PRs a warm dependency-layer cache;
Python/native source changes still rebuild the affected layers. The first run
or a cache miss still performs the full build. Pip download caches remain keyed
by Python version and dependency manifests. Superseded PR runs are cancelled.

The entire repository is copied before native compilation, so Python/test edits
can still invalidate the native build layer. Separating native compilation from
Python packaging may help later, but requires checking the source-distribution,
ABI, and native build-identity contracts. It is not part of this quick pass.

Routine CI now keeps the Docker/Podman end-to-end validation and native G7–G9
qualification, with a small installed-image smoke check in the native job. That
check runs outside the checkout with Python isolated mode, checks packaged
resources and source identity, executes the Rust backend with no fallback, and
checks CLI help. The normal image build still builds and installs the Rust wheel.

The full distribution audit is available through **Actions → Manual packaging
and release audit → Run workflow** (`.github/workflows/foundation-release.yml`).
It retains the clean-environment installation, source-distribution rebuild,
resource equivalence, installed CLI acceptance, and G10 archive/replay evidence.
This repository is used through containers, so those release checks no longer
run on every PR. Removing that job from routine CI does not qualify a release or
make G10 pass; the manual audit must run when that evidence is needed. The other
regression tests formerly repeated in that job remain covered by the CPU,
training-integration, native-qualification, and portable-engine jobs.

Per-test duration output will make the next successful run more useful for
choosing further test splits. No end-to-end speedup is claimed before measuring
the new layout: image transfer and the existing 8–9 minute public CLI validation
can still dominate, even after duplicate builds are removed.

## Published images

The registry is `ghcr.io/grugnog/oeis-learn`. Each published build receives a
unique `ci-<target>-<run-id>-<attempt>` tag; its exact digest is recorded in the
producer job summary. Default-branch publishing also updates `qualification-main`
or `gpu-main`. Qualification publishing runs automatically after relevant merges.
The much larger ROCm GPU image is manual: **Actions → Publish container images
and shared build cache → Run workflow → target: gpu** on `main`. It is an image
build, not evidence of actual GPU execution or successful training.

For a local ROCm run after that publication, pull with your chosen engine and
use the reported digest with the existing portable CLI, for example:

```sh
podman pull ghcr.io/grugnog/oeis-learn:gpu-main
podman image inspect ghcr.io/grugnog/oeis-learn:gpu-main --format '{{index .RepoDigests 0}}'
# Use the printed ghcr.io/...@sha256:... reference as --image in your usual command.
python3 scripts/foundation_run.py --engine podman --image ghcr.io/grugnog/oeis-learn@sha256:REPLACE_WITH_REPORTED_DIGEST --help
```

Docker supports the same pull/inspect commands. GHCR packages initially use
GitHub's private visibility default; local private pulls require an existing
registry login. CI uses the repository's built-in `GITHUB_TOKEN`, with package
write permission only on the producer. If an existing package denies publishing,
check that it is linked to this repository and permits its Actions workflows.
No personal access token or host account integration is added to the application.
