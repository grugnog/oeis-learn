# 007 validation and implementation quickstart

**Current status**: This branch supplies the complete design and Spec Kit integration. The `foundation` application commands, Docker files and test modules below are implementation targets; they are **not implemented or executed by this documentation change**. See [validation/report.md](validation/report.md) for checks actually run. Do not confuse document validation with the runtime gates below.

## Validate the documents now

Run from the repository root. A checkout's `.specify/feature.json` is intentionally local/ignored; the environment variable below selects this feature without depending on a previous session.

```bash
python -m venv .venv-speckit
.venv-speckit/bin/pip install 'git+https://github.com/github/spec-kit.git@fe1d00e3ccaf495880aaf90fb0e17679e82f065b' 'jsonschema==4.26.0'
export SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation
.venv-speckit/bin/specify version
.venv-speckit/bin/specify integration status
.venv-speckit/bin/specify check
bash .specify/scripts/bash/setup-plan.sh --json
bash .specify/scripts/bash/setup-tasks.sh --json
bash .specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-tasks --include-tasks
.venv-speckit/bin/python specs/007-experiment-foundation/validate_artifacts.py
git diff --check
```

`setup-plan.sh` preserves an existing plan; `setup-tasks.sh` resolves the active template and lists design artifacts. Neither generates a finished plan/task list without the corresponding skill workflow. `specify check` inventories tools; optional missing agents/toolchains do not establish missing runtime capability in the future Docker image. The project-specific validator checks coverage, structure, local document links and schema fixtures; it is not an upstream Spec Kit command or a semantic proof. Review with `$speckit-analyze` after document changes, and report governance adoption separately.

The project installs upstream skills under `.agents/skills/`. New agent sessions can discover `$speckit-specify`, `$speckit-plan`, `$speckit-tasks` and `$speckit-analyze`. This turn followed their installed instructions directly. Copilot's prior managed skill set is replaced on this branch by the validated Codex integration; the original main branch retains the old setup. To switch integrations later, use the official `specify integration` commands and verify status; do not edit manifests by hand.

## Implementation order and prerequisites

Follow [tasks.md](tasks.md). T001 is a standalone Docker/device feasibility probe; T002 records actual evidence and maintainer adoption of the [constitution proposal](constitution-rfc.md). All later strict implementation promotion requires that adoption. No multi-day language comparison is part of007 acceptance. Expanded acceptance includes a bounded two-hour engineering session and at most six hours for a diagnostic plus one conditional paired learning comparison.

On Ryzen, prerequisites are the user's existing Python and Docker, permission to run Docker, sufficient free disk, and existing AMDGPU/KFD/render-node support. No host compiler, ROCm package installation or NixOS reconfiguration is prescribed. A failed GPU gate reports exact missing capabilities for a separate decision. Image download/build time is measured separately from the ten-minute execution probe.

## After T001: standalone hardware evidence (G0)

The planned launcher contract is `run_container.sh ROLE -- COMMAND...`; roles `preflight`, `test`, `controller`, `learner`, `evaluator` select fixed bounded mounts/devices. It runs the digest-locked image produced by `build_image.sh`; default networking is disabled during execution. Paths below are relative to the checkout inside the image, while `runs/007-preflight` is a bounded host output mount.

```bash
bash scripts/foundation/build_image.sh --output runs/007-preflight/runtime.lock.json
bash scripts/foundation/run_container.sh preflight -- \
  python -m oeis_learn.cli.foundation_preflight --hardware-only \
  --config docker/foundation/preflight.yaml --output runs/007-preflight --json
```

Expected: explicit HIP/GPU architecture and device identity; at least three changed-parameter updates with finite loss/gradients on the actual encoder/decoder; synchronized timings/memory; real image/lock hashes. Failure: nonzero exit with `hardware_not_ready` or the specific numerical/resource cause. CPU-only diagnostics do not pass this gate. Attach evidence to the RFC adoption record; do not mark adoption on the basis of these commands merely being documented.

## After implementation: focused software acceptance (G1–G4, G6)

```bash
bash scripts/foundation/run_container.sh test -- \
  python -m pytest tests/contract/test_foundation_artifacts.py \
  tests/unit/test_multi_limb_arithmetic.py \
  tests/contract/test_macro_instruction_lowering.py \
  tests/unit/test_foundation_reference.py \
  tests/unit/test_foundation_codec.py \
  tests/unit/test_foundation_cohort.py \
  tests/unit/test_foundation_admission.py \
  tests/integration/test_foundation_execution.py \
  tests/integration/test_foundation_evaluation.py \
  tests/integration/test_foundation_resume.py \
  tests/integration/test_foundation_proof_boundary.py -q
bash scripts/foundation/run_container.sh controller -- \
  oeis-learn foundation conformance --profile configs/foundation/wat_profile.yaml \
  --output runs/007-acceptance/conformance --json
```

Expected: exact negative multiplication/full literals/carries; bad solver assignment rejected; correct arity/status and fresh state; zero unexplained differential disagreements; no unknown/truncated tokens; no teacher/imported/legacy admission; hidden/metadata perturbations do not alter candidates/selection; missing checkpoints cannot invoke references; finalization cannot be bypassed; full checkpoint/RNG/cursor/budget recovery; proof labels cannot enter results.

The proposed tests must be added by their tasks before these commands are meaningful. Run targeted regressions first and the existing affected suites after integration; do not repeatedly expand tests without a specific remaining risk.

## Small end-to-end fixture (G2–G5)

Implementation provides `tests/fixtures/foundation/source/`, a source-only cohort fixture with no program solutions, and its small-count `cohort.yaml`. These are isolated tests, never part of the real final benchmark. Freeze the cohort before creating the generic pool.

```bash
bash scripts/foundation/run_container.sh evaluator -- \
  oeis-learn foundation freeze-cohort --source tests/fixtures/foundation/source \
  --config tests/fixtures/foundation/cohort.yaml --output runs/007-smoke/cohort --json
bash scripts/foundation/run_container.sh controller -- \
  oeis-learn foundation build-pool --config configs/foundation/generic_sampler.yaml \
  --cohort runs/007-smoke/cohort --output runs/007-smoke/pool --json
bash scripts/foundation/run_container.sh learner -- \
  oeis-learn foundation train --config configs/foundation/wat_smoke.yaml \
  --pool runs/007-smoke/pool --run-dir runs/007-smoke/training --json
bash scripts/foundation/run_container.sh test -- \
  python -m pytest tests/integration/test_foundation_resources.py -q
bash scripts/foundation/run_container.sh learner -- \
  python -m pytest tests/integration/test_foundation_gpu.py -q
```

For the GPU-test command, the learner role mounts only the GPU test module and its bounded generated-program fixtures, never the private cohort/truth files. Ordinary test-role commands have no GPU devices.

Expected: 64 distinct independently verified generic outputs within the declared fixture budget, rejection/yield accounting, three finite GPU updates, complete external-hash checkpoint manifests and bounded resource evidence. If the fixture pool cannot be built, report that shortfall and inspect generator/operator yields; do not inject a family teacher to pass. No synthesis success-rate threshold is required from three updates.

The CLI's JSON result supplies the actual completed checkpoint manifest path. Use that emitted path as `CHECKPOINT_MANIFEST`; do not guess a checkpoint filename. A real evaluation also requires a frozen protocol from the implementation fixture, not a copied legacy 20+100 protocol.

```bash
bash scripts/foundation/run_container.sh evaluator -- \
  oeis-learn foundation evaluate --checkpoint "$CHECKPOINT_MANIFEST" \
  --cohort runs/007-smoke/cohort --split development \
  --protocol tests/fixtures/foundation/evaluation.json \
  --output runs/007-smoke/development --json
```

Expected: actual checkpoint identity, sealed prefix-only candidates, separate prefix/any/top-one metrics, complete fixed denominator and finite-evidence status. A random/smoke model may solve zero targets and still demonstrate correct infrastructure.

## Recovery and negative cases

Use the deterministic CPU integration fixture for interruption equivalence. It compares uninterrupted execution with an update-boundary checkpoint/resume, injects a corrupt newest generation, verifies fallback to the previous valid one, and checks that the ledger never refunds consumed budget. GPU tests measure finite recovery and identity, not cross-platform bitwise equality.

Try wrong hashes, unknown/inactive options, a weights-only checkpoint, a missing pool/checkpoint, a truth field injected into a visible prompt, repeated finalization with a changed candidate list, a full-match record with fewer than 100 values, and a `PROVEN` status. Each must fail before qualification. Resource tests hang/kill workers and exhaust queue/cache/disk allowances in temporary directories; they never deliberately exhaust the real host.

## Expanded repair and release acceptance (G7–G10)

After implementing the relevant tasks, run the owning-path regressions first, then qualify extensions. The initial formula checker is a G7 requirement; richer program-class checkers are G9.

```bash
bash scripts/foundation/run_container.sh test -- \
  python -m pytest tests/unit/test_foundation_grounding.py \
  tests/unit/test_foundation_symbolic.py \
  tests/integration/test_foundation_native_optimizer.py \
  tests/unit/test_foundation_cache.py \
  tests/unit/test_foundation_policy.py \
  tests/unit/test_foundation_archive_curriculum.py \
  tests/unit/test_foundation_integer_features.py \
  tests/unit/test_foundation_structural_splits.py \
  tests/integration/test_foundation_qualified_resume.py \
  tests/unit/test_foundation_extended_wat.py \
  tests/integration/test_foundation_discovery.py \
  tests/unit/test_foundation_source_policy.py -q
bash scripts/foundation/test_wheel.sh
bash scripts/foundation/run_container.sh controller -- \
  oeis-learn foundation qualify --config configs/foundation/qualified.yaml \
  --output runs/007-acceptance/qualified --json
```

Expected: repaired original APIs, no false UNSAT/proof labels, actual native parity rather than skipped tests, logit/support equivalence, exact input and split eligibility, complete active-state resume, atomic stream retries, replayed certificate mutations, installed package resources and all source-policy denominators. CandidateResult remains finite evidence; separate checked certificates expose domain/assumptions.

The implementation creates a frozen measurement manifest from `configs/foundation/experiments.yaml`, recording actual permitted fixture/checkpoint/source hashes. Use the path returned by that preparation as `MEASUREMENT_MANIFEST`; do not invent a manifest or reuse a fixture identity as real evidence.

```bash
bash scripts/foundation/run_container.sh controller -- \
  oeis-learn foundation measure --manifest "$MEASUREMENT_MANIFEST" \
  --output runs/007-measurement --dry-run --json
```

Inspect the emitted schedule/limits, then execute the same command without `--dry-run` after all required gates pass. The manifest runs one engineering session and a30-minute diagnostic; it may select one55-minute-per-arm comparison, up to three paired seeds within six total training hours. Missing trigger evidence is inconclusive, not permission for another run. No final truth is mounted. See [experiment contract](contracts/experiments.md) for effect/stop rules and measured-versus-planned distinctions.

The native CI job builds its extension inside Docker and verifies test execution counts; CPU wheel CI and actual GPU evidence are separate gates. Do not mark full007 complete from the earlier baseline smoke alone.

## Handoff

Store all G0–G10 gate reports, runtime locks, immutable profiles, cohort/pool hashes and the complete smoke run manifest. Mark hardware/qualification gates passed only from actual evidence. At that point, draft/finalize 008's native LODA adapter and paired experiment protocol using these interfaces. The 007 quickstart does not launch either multi-day arm or authorize host changes.
