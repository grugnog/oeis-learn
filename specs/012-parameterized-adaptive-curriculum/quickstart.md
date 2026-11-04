# Validation guide and later operator handoff

Status: specification and plan only. The proposed PAC wrapper and recipe are not
implemented. Do not reuse the old failed 256-task prospective command as a 012 run.
No HIP command is ready in this document; phase D will publish and validate it.

## Checks available now (no engine or HIP work)

From the repository root:

```sh
SPECIFY_FEATURE_DIRECTORY=specs/012-parameterized-adaptive-curriculum \
  bash .specify/scripts/bash/check-prerequisites.sh --json --require-spec
python3 -S scripts/foundation_run.py --help
python3 -S scripts/generalization_inputs.py --help
git diff --check
```

Expected: the 012 spec/plan and design artifacts resolve; both existing host entrypoints
show help without third-party host dependencies. These checks validate planning
and entrypoints, not PAC behavior or hardware readiness. `tasks.md` is intentionally
absent until the tasks workflow.

## Required original workstation inputs

Use explicit roots or a bounded inventory rather than broad automatic search:

- `run-state/generalization/fresh-p1-v2-20261103T035534Z`: original prepared corpus,
  inventories and all protected exclusion inputs.
- `run-state/generalization/fresh-p1-hourly-20261104T093555Z`: original full pilot
  receipts, initializer/checkpoint state and results.
- `run-state/generalization/observable-eval-20261104T193226Z/acceptance/report/qualified`:
  historical producer authority, reusable only if the required closure still matches.
- `run-state/generalization/uniform-import`: provenance/recipe inputs while the
  original chain references them.

Keep referenced immutable images and critical public reports. A public ZIP cannot
replace private weights/optimizer/RNG or protected exclusion data. Input inspection
must name exactly what is missing; do not ask the operator to relist every run.
No new prior chat/file upload is required for task planning.

## Local implementation validation before hardware

The implementation must add tests alongside the existing related suites, then run
the affected non-HIP unit/contract/integration selection in the pinned environment.
Do not prescribe a nonexistent test file as a runnable command today. Record the
actual validated command and outcomes here during phase D. Required scenarios:

1. Reject changed structural draw IDs, missing original exclusions, a single quota
   deficit, unknown PAC fields and an old P3 disguised as adaptive. No GPU launches.
2. Ground paired targets with independent expectations, preserve control fallback
   and exact original rows; reject probe/gradient or held-out-role leakage.
3. Exercise hand-calculated improving/stuck/noisy families, probability caps,
   simultaneous due reviews, actual token-charged review and censored diagnostics.
4. Run installed real-tensor CPU fresh/pause/resume paths and inject failure before
   and after debit, update, feedback, review and atomic snapshot publication.
   Preserve exact clean replay and separately debit discarded crash work.
5. Construct Docker and Podman commands for every new role; verify the shared
   environment, read-only inputs, no socket, private disk scratch and bound source.
6. Validate the entire nested deadline/storage graph with delayed cold start,
   production-sized metadata, save/copy/hash/exit/cleanup and reporting. A larger
   outer timeout must not hide a smaller inner/schema maximum.
7. Use the real archive packer/verifier for success, failure, interruption, missing
   nested evidence, stale PASS and export failure; preserve first and later errors.
8. Resume a fixture campaign and demonstrate that compatible completed stages are
   not repeated while incompatible/unverified outputs cannot be reused.

## One later operator invocation

The implemented handoff must provide one copy-pastable block with explicit branch,
engine, original roots, output and frozen plan identity. It composes:

1. Read-only input/control audit and full CPU/native post-exclusion quota admission.
2. Required compatible qualification and representative cold/warm production-size
   update/save/load calibration, under relaxed finite limits.
3. Tiny installed HIP parameterized/adaptive/review and full-state resume acceptance.
4. Retained structural baselines and new B/C training/evaluation only when the sealed
   plan already authorizes that research scope; optional fresh A only if predeclared.
5. Deterministic case analysis, independent final verification and shareable evidence.

The wrapper must package failures automatically and name the minimal retry boundary.
A compatible resume reuses completed work. Read-only `recover` never launches a
learner. Planned subcommands are in [execution.md](contracts/execution.md); their
actual syntax must be verified against installed help before replacing this guide
with a run command.

## Runtime expectations

Original literal training took 147.4/159.5 minutes per 25M-token arm, as recorded in
the [011 closeout](../011-controlled-generalization/validation/merge-closeout.md).
These are context, not parameterized/adaptive throughput measurements. B/C may
change examples per token, update count and diagnostic overhead. Give measured ETA
ranges after calibration and retain substantial timeout headroom. Initial planning
floors are 12 hours per learner and 24 hours for preparation; larger projections
raise them. The full campaign ceiling sums its selected stages plus cleanup and
at least one hour of finalization; it is not restricted by an inherited six-hour
or fixed global ceiling. Scientific per-task attempts/time remain unchanged.

Expected final output separates software PASS/FAIL, scientific interpretability,
matched/unmatched A, parameter endpoint DEFERRED, completed/remaining work, costs,
censoring, cases, archive hash and original producer references. A negative learning
result can accompany software PASS. No automatic extra seed or winner selection.
