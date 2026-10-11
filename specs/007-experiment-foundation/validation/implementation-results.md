# Implementation results

Phase 10 implementation is in [PR #11](https://github.com/grugnog/oeis-learn/pull/11). [The review report](phase-10-review.md) records the completed integration, tests and explicit acceptance limits. [The disposition map](phase-10-dispositions.json) preserves all 68 original findings and binds the phase 10 package implementation hash.

The remaining historical-wrapper, telemetry, wider-trace and live-measurement software paths have been implemented. They include source-file replay, shared measured updates, charged stateful retries, provenance-linked tracing, isolated Docker roles, strict GPU smoke, frozen engineering comparisons, a single conditional pilot and replayed G0–G10 readiness. Diagnostic hypotheses now pass the same archive admission and original holdouts as self-training. Encoder triggers probe the actual frozen baseline's input features; legacy feature aliases cannot spuriously trigger a comparison for the repaired baseline. Repeated installed-wheel verification allocates fresh run directories while preserving earlier checkpoints. Historical resume selects epochs numerically. Timed-out workers are reclaimed before replacement without regenerating requests; container and client reclamation share one two-second allowance.

Phase 10 executable code is `ba89b9694128775cd136b890f77a249b650a70ae`. Phase 10 code validation:

- [CPU/native review CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372626): **712 passed, 56 explicit capability skips**; compile/lint, Spec Kit structural checks and native qualification passed.
- [Installed-wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38088372595): **6 installed-wheel cases passed with no skips**, **69 phase 10 regressions passed**, and **8 actual Docker role tests passed**. Repeated train/resume/search verification preserves independent checkpoint directories.
- G7 and G9 passed; G8 software passed. G8 hardware and G10 live/source acceptance remain pending. These results do not qualify GPU learning or performance.

Local focused validation recorded **50 passed, 1 Docker capability skip**, then **16 passed, 1 skip** for final repair regressions and **22 passed** for source/readiness tests. Local source-distribution and wheel builds preserved all **388** qualification source/resource bindings. The earlier restricted-host broad run and its environmental limitations are recorded in the review report. CPU fixtures cannot establish GPU performance or learning benefit.

T093 and T095–T100/T102 remain half-checked for actual-source, trained-model or GPU acceptance. T094 packaging and T101 reporting remain checked. The actual release corpus is unavailable here and is a data dependency; its audit runs on CPU once supplied. The independently named bounded capacity pilot uses FF2048 to reach the 25M class; the existing qualified FF1536 constructor remains supported. Forward/backward interface fixtures execute for both encoder representations and positional variants even when their comparison is not triggered.

## Phase 11 cross-cutting review

[The phase 11 review](phase-11-review.md) records repaired operational documentation,
original-input/task reconciliation and atomic bounded handoff/replay. See the
[handoff](handoff.md) and [current 68-input dispositions](phase-11-dispositions.json).
Earlier validation above remains historical; full 007 qualification stays pending.

Local focused regression command:

```sh
OMP_NUM_THREADS=2 .review-venv/bin/python -m pytest -q -ra tests/unit/test_foundation_handoff.py tests/contract/test_foundation_qualified_cli.py --junitxml=phase11-evidence/focused-final.xml
.review-venv/bin/ruff check --select F,E9 src/oeis_learn/evaluation/handoff.py src/oeis_learn/evaluation/review_dispositions.py scripts/verify_phase11.py scripts/archive_phase11.py tests/unit/test_foundation_handoff.py
```

Result: **24 passed, 3 explicit native/gate capability skips**; lint passes.
The locked CPU environment uses Python 3.12.14 and torch 2.9.1+cpu; dependencies
are exported from `uv.lock` with torch pruned and installed with hashes/no-deps.
Docker, Cargo and `lab-gpu` are unavailable locally. No host compiler/ROCm
installation or production boot/deadline relaxation was performed.

Actual CPU quickstart command: `OMP_NUM_THREADS=2 .review-venv/bin/python
scripts/verify_phase11.py --output phase11-evidence/quickstart`. The workspace lacks
`/proc/sys/kernel/random/boot_id`: cohort/measurement/handoff execute; the initial census driver also rejected its malformed terms,
pool rejects and dependent training/evaluation remain explicit failures. The initial bounded archive preserved these failures and pending G0–G10.
The driver now preserves indexed census records and uses the final lock output
path; Linux CI executes the corrected full quickstart and fresh replay. Missing GPU/source/trained-model acceptance
keeps T104/T105 and earlier acceptance tasks half-checked.

Additional local validation:

- Broad command: `OMP_NUM_THREADS=2 .review-venv/bin/python -m pytest -q -ra tests/unit/test_foundation_*.py tests/contract/test_foundation_*.py tests/integration/test_foundation_*.py --junitxml=phase11-evidence/broad.xml`: **699 passed, 56 capability skips, 22 failures, 12 setup errors**. Nineteen failures and all twelve setup errors directly name the unavailable Linux boot ID; three CLI assertions reflect the corresponding invalid-input exits. These are not marked passing. Current Linux CI must supply acceptance.
- `python -m build --sdist --wheel --no-isolation --outdir phase11-evidence/dist` and an extracted-sdist wheel rebuild succeed. The wheel verifies **395** source/resource bindings, including exact original-input and task-checklist governance bytes. Native compilation stays in Docker CI.
- Official upstream Spec Kit at `fe1d00e3ccaf495880aaf90fb0e17679e82f065b`, CLI 1.0.7: `specify version`, `specify check`, and `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation specify integration status` succeed. Integration is Pi, with zero modified/missing/invalid/unchecked managed files; local coding-agent CLIs are unavailable and not installed for this review.
- `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation bash .specify/scripts/bash/check-prerequisites.sh --json --require-spec --require-tasks --include-tasks` passes. `python specs/007-experiment-foundation/validate_artifacts.py` passes: 52 requirements, 108 tasks, 68 original findings, 100% structural requirement/task coverage, 82 checked links and 14 negative schema cases. Neither command executes runtime gates.
- Read-only semantic review compares constitution/spec/plan/contracts/tasks/source findings; no new constitution conflict or unmapped requirement/task. Findings and remaining runtime acceptance are summarized in the review. Requirements/design checklists remain untouched.

The first Linux run at `9f92f4cb88f5d97ccb7152b3eb48f48d0e119eec`
passed **733 CPU tests / 56 capability skips** and native qualification, then
exposed two quickstart argument errors: a non-lock final output path and indexed
integer text incorrectly supplied as normalized census integers. These driver
errors are repaired; failures stay visible in that run's archive. No service
validation guard was relaxed. Clean local installed-wheel acceptance on the
corrected final-path driver recorded **3 passed / 3 native/gate skips**; packaged
reconciliation independently verified all **68** original inputs/tasks outside
checkout with Python isolated mode.

The first wheel/native CI passed all **6 installed-wheel cases**, **90 integration
regressions** and **8 actual host Docker role tests**, but its sampler-inclusive
handoff exceeded the fixed file cap. The collector now selects the owning reports
and their declared dependencies. A 4,100-unrelated-trial regression proves these
cannot consume the gate archive, while missing/escaping dependencies still reject.
No quota or gate was weakened. Latest focused suite: **27 passed / 3 capability
skips**; lint passes. The full CI run below validates this closure.

At `252e6372d557625d515a91b31d7847338bfcbb38`,
[review CI](https://github.com/grugnog/oeis-learn/actions/runs/38095379995)
passed **736 CPU tests / 56 capability skips**, lint/structural checks, the complete
CPU quickstart and native G7/G8-software/G9 qualification.
[Wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38095379931)
passed **6 installed-wheel cases**, **93 integration regressions**, **8 actual
Docker role tests**, and copied-dependency archive replay. The gate archive has
**30 files / 41,960,608 bytes**, with G7/G9 PASS, G8 software PASS/hardware pending,
and G10 wheel/integration/68-input checks PASS; source/measurements remain pending.

A final source-distribution check found that owning tests need the phase 11
disposition JSON. `MANIFEST.in` now includes it. The extracted source distribution
executes all 24 handoff tests plus the existing current-disposition binding test:
**25 passed**, using the locked local CPU environment with the extracted source
and scripts first on `sys.path`. Direct and rebuilt wheels verify **395**
source/resource bindings. This packaging-only change receives fresh native/wheel
CI below; Python service and driver bytes remain those validated at `252e637`.

Final packaging code is `e8b64943078484f53f7de16ec6fd1aea22910962`.
[Review CI](https://github.com/grugnog/oeis-learn/actions/runs/38096255204)
passed **736 CPU tests / 56 explicit capability skips**, code-quality and Spec Kit
checks, all **11 actual CPU quickstart slices** plus immutable replay, and native
G7/G8-software/G9 qualification. The rebuilt wheel from the reconciled source
distribution independently verifies all **395** exact source/resource bindings.

The CPU diagnostic archive contains **3,125 files / 207,117,355 bytes**.
Its input manifest is
`sha256:6471743bd4d8fc002f6e5feeedf2595ee8f090b2e85e2b315707e3d97bfd3faf`;
its report is
`sha256:9eda29e812caa8bbb1577565dab882fc09ecf27c2251a8b756684c879fa92264`.
The published CPU results artifact
[11685674791](https://github.com/grugnog/oeis-learn/actions/runs/38096255204/artifacts/11685674791)
has digest
`sha256:079f3074b178f51f5a1b50dbca43274c30dab0bf7f66d51dc9494d6524264c4e`,
verified after download. It includes the report and slice logs, with large
checkpoint blobs retained only inside the bounded CI replay. Four measured
fixture prompts leave a 124-prompt shortfall; all six conditional decisions are
inconclusive, `qualified=false`, and every release gate in that diagnostic archive
is pending. These are executed diagnostics, not adopted experiment outcomes.

[Installed-wheel/native/Docker CI](https://github.com/grugnog/oeis-learn/actions/runs/38096255200)
on the same final packaging code passed **6 installed-wheel cases with no skips**,
**93 integration regressions**, **8 actual Docker role tests**, and copied gate
archive publication/replay. The archive contains **30 files / 41,960,097 bytes**;
its input manifest is
`sha256:bd4083ab47c3fbaebb59ed610cde136e933f1d6acdc2e876ed6d13ca475c86e6`
and its report is
`sha256:5a36dbcf9a70d526b127561a824ab977914894a87b9262e9baa28df120a8000a`.
G7/G9 PASS; G8 software PASS/hardware pending; G10's six wheel cases, 101 combined
integration/Docker cases and 68-input reconciliation PASS. G0–G6 baseline/device
binding and G10 source/qualified measurement acceptance remain pending.

The complete selected archive, actual native wheel and resolved image are in
[artifact 11686525939](https://github.com/grugnog/oeis-learn/actions/runs/38096255200/artifacts/11686525939),
whose digest is
`sha256:8770f4eb2e44a4edac247eff1fb3e01417f94ade795e8b125eee45d869fbac0c`.
After download, ZIP integrity, the artifact digest and every one of the 30
copied dependency hashes/file counts/byte counts independently verified.
The final evidence update changes documentation and restores the original
Markdown file mode only; qualification source bytes and task markers remain
identical to this tested code. No pending acceptance is promoted to a pass.

## Reconstructed VM GPU progression runner (2026-10-11)

The previous transient checkout and unpushed local commit `b522f6b` were no longer
available. GitHub had neither that commit nor its intended branch. The runner
and integration repairs were reconstructed from the recorded session history;
the prior version's test results are not transferred to these new bytes.

The disposable `scripts/gpu_progress.py` controller and worker module inventory
all earlier unfinished tasks before phase-13/14 owning checks. They cover CPU
input work, real HIP smoke/checkpoint/constructor checks, all 15 accelerator
cases, host containment, preflight and baseline, actual historical artifacts,
GPU-produced overflow strata/telemetry, the immutable two/six-hour engineering
and conditional pilot schedule, native/installed-package/readiness handoff and
one final diagnostic lifecycle. Task and checklist markers remain unchanged.

Receipts bind source, settings, recursively referenced inputs, dependency
identities and retained outputs. Verified successes skip on rerun; temporary
dependency failure preserves their receipts. Remote locks/watchdogs and saved
lease-scoped reclamation protect against duplicates and abandoned Docker roles.
Measurement issuance is non-reissuable. The final stage has a persistent one-hour
default wall allowance, maximum twelve hours, including transport/cleanup grace;
its separate three-update training ledger remains capped at ten minutes. The
last stage pauses/resumes isolated training, evaluates a small original-partition
cohort, freezes stopping/finalization, evaluates sealed final truth and writes
planning metrics. Three updates are lifecycle diagnostics, not convergence.

Reconstructed code-path repairs include foundation CLI argument forwarding,
closed preflight run provenance, coordinated pause/resume configuration binding,
caller-interruption reclamation of a detached learner/controller session, role
stdout/stderr retention, Docker controller clients in the GPU image and image
cache input closure. The guide records the required source/weights/host inputs
and remaining sustained historical launcher acceptance.

Fresh validation:

- `OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/unit/test_foundation_gpu_progress.py tests/unit/test_foundation_run_provenance.py tests/unit/test_foundation_preflight_identity.py tests/unit/test_foundation_legacy_wrappers.py tests/unit/test_foundation_roles.py tests/unit/test_foundation_config.py tests/contract/test_foundation_artifacts.py tests/unit/test_foundation_handoff.py tests/unit/test_foundation_task_inventory.py tests/unit/test_foundation_phase10_integration.py`: **231 passed, one explicit Docker capability skip**.
- The four `tests/integration/test_foundation_legacy_artifacts.py` cases executed
  against a generated tiny CPU v2 checkpoint/configuration: **4 passed**. This
  verifies the test harness, not actual historical parent/GPU acceptance.
- Black on all five new Python files and Ruff F/E9 on all changed Python files
  pass. `git diff --check` passes.
- Offline direct wheel/source-distribution builds and an extracted-source wheel
  rebuild pass. All **181** package files match byte-for-byte; the source
  distribution includes the controller, worker, guide, settings and tests.
  Generated compiler/package metadata is excluded from source-cache identity;
  a regression verifies real source edits still invalidate it.
- `python scripts/gpu_progress.py --plan` and Spec Kit prerequisites pass.
  The artifact validator passes: 52 requirements, 121 tasks, 68 original findings,
  83 local links and 14 negative schema cases. No extension hooks are registered;
  both existing requirements/design checklists have zero unchecked items.

This host lacks `lab-gpu`, Docker, native compilation and the required actual
source/historical artifacts; Linux boot identity is unavailable here. Live
hardware, host containment, source, sustained launcher, engineering/pilot and
final training/evaluation acceptance remain pending. No missing acceptance is
promoted to passing or replaced with fixture evidence.
