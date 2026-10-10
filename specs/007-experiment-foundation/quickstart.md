# Spec 007 operational quickstart

These commands describe implemented interfaces. Use explicit new output directories
and the root Dockerfile; local diagnostics do not authorize workstation training,
release, performance adoption or a LODA comparison. Exact current evidence and
remaining acceptance are in [validation/handoff.md](validation/handoff.md).

## 1. Project and document checks

```bash
uv sync --locked
uv run oeis-learn foundation --help
export SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation
bash .specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-tasks --include-tasks
uv run python specs/007-experiment-foundation/validate_artifacts.py
```

With the official Spec Kit CLI installed, run `specify version`, `specify check`
and `specify integration status`. The checked-in managed integration is Pi; do
not overwrite it to perform this review. Read-only semantic analysis compares
spec/plan/tasks, the adopted constitution and all 68 source findings. Structural
validation is not a runtime gate.

## 2. Execute the bounded CPU slices

```bash
OMP_NUM_THREADS=2 uv run python scripts/verify_phase11.py --output runs/007-cpu-quickstart
uv run python -m oeis_learn.evaluation.handoff --archive runs/007-cpu-quickstart/handoff
```

The executable driver records exact commands, stdout/stderr and per-slice status
in `quickstart.json`. It uses the small indexed source in `tests/fixtures/foundation`,
`generic_sampler.yaml` and the full configured three-update `wat_smoke.yaml` model:
cohort, admitted pool, CPU pause after update one, resume through update three,
inspection, development evaluation, stopping lock/final evaluation, census,
measurement dry-run and actual CPU diagnostic, then immutable handoff. It does
not use miniature substitute weights. Linux `/proc` boot identity and process
containment are required; interrupted prerequisites remain explicit failures.
Small-host resource floors are recorded diagnostic deviations.

The individual continuation interfaces are:

```bash
uv run oeis-learn foundation train --config configs/foundation/wat_smoke.yaml --pool runs/007-cpu-quickstart/pool --run-dir runs/007-another-smoke --device cpu --diagnostic --diagnostic-small-host --stop-after-update 1 --json
uv run oeis-learn foundation resume --run-dir runs/007-another-smoke --json
uv run oeis-learn foundation inspect --run-dir runs/007-another-smoke --json
```

Use the actual manifest path returned by train/resume for evaluation, not a
weights-only file. Finalization binds checkpoint/protocol/cohort/stopping record;
final evaluation accepts only that lock. Inspection is read-only. Existing run
settings, source hashes and budget cannot be overridden on resume. Recovery and
migration rules are in [docs/foundation.md](../../docs/foundation.md).

## 3. Native, qualified interfaces and clean wheel

Build from the repository root and use a fresh evidence directory:

```bash
docker build --target qualification -t oeis-007-qualification .
mkdir -p runs/007-native
chmod 777 runs/007-native
docker run --rm -v "$PWD/runs/007-native:/evidence" oeis-007-qualification bash -c '
  set -euo pipefail
  trap "chmod -R a+rX /evidence" EXIT
  python -m oeis_learn.evaluation.repair_gate --output /evidence
  python -m oeis_learn.evaluation.learning_gate --output /evidence/learning
  PYTHONPATH=/workspace/src:/workspace python scripts/verify_phase8.py --evidence /evidence
  oeis-learn foundation qualify-discovery --native --output /evidence/discovery --json
  PYTHONPATH=/workspace/src:/workspace python scripts/verify_phase9.py --evidence /evidence
  export OEIS_WHEEL_WORK_ROOT=/evidence OEIS_NATIVE_WHEELS_DIR=/tmp/oeis-wheels
  export OEIS_REQUIRE_NATIVE=1 OEIS_QUALIFICATION_EVIDENCE=/evidence
  bash scripts/foundation/test_wheel.sh
'
```

This executes real Rust/native parity and G7/G9; G8 CPU software can pass while
hardware stays PENDING. The verification drivers exercise actual qualified
learning updates/resume/search and generated-archive discovery/fresh certificate
checking. Public names are `learning-train`, `learning-search`, `qualify-repairs`,
`qualify-learning`, `qualify-discovery`, `discover` and `check-certificate`; there
is no catch-all `foundation qualify` or baseline `--qualified-config` flag.
Gate artifacts must replay in their matching code/runtime/image. The installed
wheel test runs outside checkout with `PYTHONPATH` unset and requires six cases
without skips when native qualification is required.

For actual host Docker roles after building this CPU image:

```bash
export OEIS_ROLE_TEST_IMAGE=$(docker image inspect oeis-007-qualification --format '{{.Id}}')
OMP_NUM_THREADS=2 uv run python -m pytest -q tests/unit/test_foundation_roles.py --junitxml=runs/007-native/roles.xml
```

[foundation.yml](../../.github/workflows/foundation.yml) executes the clean-wheel,
phase 10/11 integration and role suites, then `scripts/archive_phase11.py` in the
matching qualification runtime. Its copied G7–G10 dependencies are freshly
validated before publication. G0–G6, real-source/live decisions and G8 hardware
remain pending; a successful archive is not a passing release.

## 4. Workstation and real source acceptance

Use the existing managed Ryzen environment:

```bash
lab-gpu image ensure
lab-gpu exec -- python -m pytest -q tests/integration/test_foundation_gpu.py tests/integration/test_foundation_learning_gpu.py -ra
```

Actual HIP, full-model updates and allocator evidence are mandatory; unavailable
GPU support cannot fall back to CPU. The host-controlled smoke additionally
requires an explicit resolved ROCm `OEIS_GPU_ROLE_IMAGE` and host Docker access.
Run that acceptance in the existing GPU/container environment; missing host
capabilities keep it skipped. `foundation coordinated-smoke --help` and
`qualify-baseline --help` expose the exact run-bound containment and G0–G6 inputs.

A real release requires supplied indexed term files and authoritative metadata,
not the diagnostic source. First create an `indexed-source/v1` manifest with
hashed `terms`/`metadata` references and select `prefix20_total100_affine_v2` in
an explicit cohort config. See [evaluation.md](contracts/evaluation.md) and
`evaluation/release_source.py` for the closed input schema. Then:

```bash
uv run oeis-learn foundation prepare-release --manifest /path/to/indexed/manifest.json --config /path/to/release-cohort.yaml --output runs/007-release --json
```

Supply an actual generic archive, its expanded holdout profile and snapshot to
the source audit. Missing metadata or an empty structural training archive cannot
qualify. This work can run on CPU once the real source files are supplied.

## 5. Bounded live measurements and decisions

After baseline and G7/G8 acceptance, prepare in the matching image/runtime:

```bash
oeis-learn foundation prepare-measurement --archive /path/to/generic-archive --cohort /path/to/release-cohort --holdouts /path/to/expanded-holdouts.json --gate-report /path/to/g7.json --learning-gate-report /path/to/g8.json --baseline-report /path/to/baseline.json --baseline-checkpoint /path/to/checkpoint-manifest.json --image sha256:RESOLVED_IMAGE_ID --output /path/to/measurement-bundle --json
```

`RESOLVED_IMAGE_ID` must be a real immutable image identity, not a placeholder or
tag. On the Docker host, run `foundation measure --manifest
/path/to/measurement-bundle/manifest-input.json --output runs/007-live --dry-run
--json`, then use a different fresh output without `--dry-run` for execution.
The two-hour engineering cap and single six-hour diagnostic/paired pilot are
charged across setup, updates, generation, recovery and evaluations. No final
truth or imported comparative scores enter measurement inputs. Results preserve
not_triggered, baseline_retained, unavailable, inconclusive or failed outcomes;
no automatic winner or LODA experiment follows document validation.

## 6. Immutable handoff and replay

`foundation handoff --manifest /path/to/handoff-inputs.json --output
/path/to/new-archive --json` consumes this closed shape:

```json
{
  "version": "handoff-inputs/v1",
  "files": [{"path": "g7.json", "sha256": "sha256:ACTUAL_FILE_HASH"}],
  "readiness": {"g7": "g7.json"}
}
```

Paths are relative to the input manifest directory. Enumerate all required
JUnit reports, native evidence, profiles, checkpoint blobs/manifests/sidecars and
durable journals; unlisted dependencies cannot pass. Optional readiness selections
are `checkpoint`, `baseline_bundle`, `g7`, `g8`, `g9`, `wheel_junit`,
`phase10_junit`, `role_junit`, `dispositions`, `measurement_root` (a directory)
and `release_source` (cohort/archive directories plus inventoried holdouts/snapshot).
The collector rejects unsafe paths, symlinks, duplicates, changed bytes and quota
excess; it stages, validates, fsyncs and publishes once. Identical retries replay
the archive; different inputs cannot overwrite it. Replay with
`python -m oeis_learn.evaluation.handoff --archive DIR` in the pinned qualification
runtime. Archives preserve exact source bytes; artifacts containing original
absolute run paths retain those identities and are not rewritten for migration.
Only native LODA runtime/comparison/conditional transcoding is deferred to 008.
Pending non-LODA acceptance stays in 007.
