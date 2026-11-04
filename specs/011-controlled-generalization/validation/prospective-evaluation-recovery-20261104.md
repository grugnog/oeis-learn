# Prospective evaluation recovery audit — 2026-11-04

## Evidence and conclusion

Reviewed `recovery(5).zip`, SHA256
`404c6b900d76583ad790a60e8fbe1952d4c61adc15fc74f9168d977848721a84`.
Canonical bundle verification passed inventory/hash checks:523 payload members,
3,300,142 expanded bytes. This establishes archive integrity, not experiment success.
Original run: `run-state/generalization/observable-eval-20261104T193226Z`.

| Recorded evidence | Interpretation |
| --- | --- |
| Acceptance preflight, qualify, prepare, retained-eval, refresh, parameter-targets, replay-resume and failure-packaging: PASS | Current-source qualification fixtures completed |
| Acceptance bundle verification process: COMPLETE, exit0 | Qualification verification completed; generic diagnostics omitted its COMPLETE result rather than recording a failure |
| Qualification CPU suite243.16s; HIP suite117.81s, both COMPLETE/exit0 | Actual qualification execution recorded |
| Fresh preparation process FAIL/exit1 after39.02s | Not a preparation timeout |
| Parameter progress:1,681 draws/admitted/native rows, phase blocked | Surviving candidates passed native admission but required quotas were not filled |
| Peak18,736,071,108 bytes;456,460,005,376 bytes free | No recorded disk-cap failure |
| Workflow FAIL, research_tokens0, UNMEASURED | Neither research generator/scorer nor learner launched |

All original exclusion inputs were visited: selection/assessment parameter and
structural cells (128 rows each), static and eleven refresh shards (32,768 rows
each). Parameter native time was9.30s. The original error was
`native cohort shortfall or disagreement`; the recorded progress narrows this to
quota shortage. The original quota ledger is absent from the recovery archive,
so deficient families and deficits remain unknown. Structural preparation was
not reached. No model-quality conclusion follows from this run.

## Confirmed reporting defects and fixes (T175)

The exporter prefixed canonical proof paths with `qualification/`. The real
archive verifier rejects `qualification/build/Dockerfile` and corresponding
Cargo.lock paths: pinned extensionless build inputs are allowed at their
canonical paths, optionally under one `payloads/` envelope. The fix uses that
existing envelope and preserves qualification-relative references. No archive
safety rule is relaxed.

Export failure then overwrote the first error with `export: ValueError`.
Packaging now retains the original error and appends bounded later failures.
Preparation stage status/time are recorded even on failure. Both full export and
fallback diagnostics now discover `prepare/cohort/cells`, projecting only bounded
quota/progress metadata. Fresh preparation also writes per-family aggregate
availability/exclusion counts before native execution; it exports no target terms,
source programs, assignments or protected task IDs. Those new counts cannot be
reconstructed retroactively from the uploaded archive.

The new `diagnose` operation reads original evidence into a fresh, disjoint
destination. It cannot launch qualification, containers, generation or training.
It preserves original failures and never marks the experiment successful.

Validation:166 affected tests passed. Counterexamples exercise the real archive
packer with pinned build paths, original-error preservation, byte-for-byte
unchanged original run files with hardware calls forbidden, and an exhausted
family without loosening exclusions. Stdlib-only CLI help and diff checks pass.
Production source remains
`sha256:d44d83d2a8a88344893a49ccd1b80a3d8b3246a8c86953adf858105cf86dc57d`.
No source rebuild, new qualification, HIP execution or CI wait was performed.
Numeric seed, reservoir, quotas, exclusions and inference budgets are unchanged.

## Next action (T176): retrieve the original quota ledger

Do not repeat the full evaluation yet. Keep the failed run and its qualification.
Run this block from the workstation, then share the printed archive:

```bash
(
  set -euo pipefail
  cd /home/owen/workspace/oies-learn
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 -S specs/011-controlled-generalization/validation/prospective_evaluation.py diagnose \
    --run run-state/generalization/observable-eval-20261104T193226Z \
    --output "run-state/generalization/observable-eval-diagnostics-$(date -u +%Y%m%dT%H%M%SZ)"
)
```

This should take seconds to minutes for local evidence copying and hashing,
depending on filesystem throughput. It prints `DIAGNOSTIC_EXPORTED`, original
status `FAIL`, hardware `NOT_RUN`, and a new `report.zip` path. If original
qualification/ledger files have been removed, report that error; do not substitute
another run. Once shortfalls are known, assess whether reservoir expansion could
actually supply the missing families before freezing any design amendment.
The compatible original qualification can be consumed through `--qualified` on a
subsequent run, subject to unchanged source/image/resource/receipt checks.
