# Phase 5 CI correction: current review implementation binding

Date: 2026-10-23. Source reviewed: `ef240bf8e5ad04df93555fa43097565250d9eed5` on PR #59.

Three failed jobs share one cause:

- [CPU regressions](https://github.com/grugnog/oeis-learn/actions/runs/39958634987/job/120430690083): one failure, 957 passes, 68 explicit skips.
- [Podman portable CLI](https://github.com/grugnog/oeis-learn/actions/runs/39958635040/job/120432480701): its nested CPU regressions reject the same stale binding.
- [Docker portable CLI](https://github.com/grugnog/oeis-learn/actions/runs/39958635040/job/120432480851): its nested CPU regressions reject the same stale binding.

`test_review_dispositions_bind_current_package_implementation` correctly rejects the current-dispositions report's older Python implementation hash. The package now hashes to `sha256:9a5aedc3f2dffa52d6f66f528e799ebfcd0f9a8ea5662109e7c78ff433d394bf`; the report still recorded `sha256:a99cb9b79050dea0fe11d9092c3b9e0a132937e2865c63c859713c79b80d6d22` from the earlier diagnostic/fitting source.

Recomputed the current binding using `experiment_runner.implementation_identity()` and updated its source-description label for the engine-policy/pipeline-profile code. All 68 item dispositions, original-input/task hashes, historical references and pending acceptance gates are unchanged. This refresh is traceability only, not new runtime qualification. The identity assertion remains strict; no tests are removed or skipped and no runtime code changes.

Local validation reproduces the exact CI assertion before the correction. Afterward:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m pytest -q \
  tests/unit/test_foundation_phase10_integration.py \
  tests/unit/test_foundation_handoff.py
python specs/007-experiment-foundation/validate_artifacts.py
git diff --check
```

The two test modules pass all 42 cases with zero failures/skips. The artifact validator passes its document/schema checks, including 68 review inputs and 84 local links. Whitespace passes. A structural comparison confirms only the current report's `code_revision` and `implementation_sha256` fields changed.

The original CI runs' image builds, native qualification and other acceptance jobs passed; these results retain their original source scope. New Docker/Podman CI is still required after this correction. Spec 009 remains untouched and its rebase is deferred until Spec 008 CI is green. CI is not polled or awaited during this fix.
