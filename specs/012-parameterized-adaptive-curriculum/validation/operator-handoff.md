# T067 — Recorded acceptance scope and resource freeze

Recorded 2026-11-05 from the operator's supplied `report(10).zip`, corresponding to
the `pac-012-acceptance-004` invocation in this conversation. This records the
already-frozen run; it does not retroactively create or alter a scientific plan.

## Reproducible public evidence

The operator-supplied public archive `report(10).zip` is 453,255 bytes,
SHA-256 `ff468438aafc28c086b3a835953d4968c68b682c7a03c11e771baac4a05f40c1`.
Its 66 members expand to 1,432,998 bytes. The accompanying
[machine-readable record](evidence/acceptance-004-record.json) preserves original
plan, source/image/helper, scope, stage ceilings, inventory and verifier identities.
It contains no private weights, optimizer state or protected truth.

The complete ZIP stays with the operator; automatic approval review rejected its
GitHub upload because raw logs/source metadata may be sensitive. The PR contains
bounded findings and content hashes, not the ZIP, logs or full source inventory.
To reproduce verification, obtain the exact original archive from the operator and
run this read-only check from the repository root (adjust the archive path):

```sh
PYTHONPATH=src python3 -S - <<'PY'
from oeis_learn.pipeline_validation.generalization_archive import verify_archive
print(verify_archive('/path/to/report.zip'))
PY
```

The original source inventory was compared byte-for-byte with the checkout before
closeout changes. Identity:
`sha256:a5140e5d2be137411d6ae82bf6b6e28ec37ad27a8d71d3a283a4ce1ed814ac5c`.
The plan is
`sha256:7f0cd8e86a10e450998eaf26fab1b0704cee882d7fa0001ce5858e14bcf351c8`.
The archive independently verifies integrity PASS, software PASS and feature
qualified, with scientific status UNMEASURED.

## Authorized scope and originals

The exported scope is **acceptance only** (`research=false`), seed 20261016,
declared future B/C arms and a 25,000,000-token research allowance per arm. That
allowance is not a claim of completed research training. No fresh A or D was
selected; parameter evaluation is DEFERRED and assessment NOT_EXECUTED. Original
control matching must not be inferred from the absence of an A learner. The
source-bound assembler uses REFERENCE_ONLY when no fresh A is declared.

The supplied operator command used these workspace-relative roots:

- `run-state/generalization/fresh-p1-v2-20261103T035534Z`: original prepared corpus.
- `run-state/generalization/fresh-p1-hourly-20261104T093555Z`: pilot and lineage.
- Referenced uniform-import and qualification lineage remain on the workstation.

`native-admission.json` inventories all sixteen required original cells: four
128-row selection/assessment parameter/structural cells, one 32,768-row static
cell, and eleven 32,768-row refresh cells. These remain exclusions even when
parameter evaluation and assessment execution are deferred. Exact protected
artifact references are intentionally omitted from the public projection; the
original admission identity commits them on the workstation. The ZIP is not a
replacement for those originals or a resumable tensor checkpoint.

## Frozen operational graph

| Serial stage | Requested/effective outer ceiling (seconds) |
| --- | ---: |
| Admission | 118,860 |
| Qualification | 126,060 |
| Calibration | 50,460 |
| Acceptance | 68,460 |
| Protected final reporting/recovery | 3,600 |
| Campaign | 367,440 |

Both requested and effective stage lists are present and identical. The campaign
ceiling equals the four stage ceilings plus finalization. Detailed private plan
and peak-free-space measurements are not exported; qualification records four
CPUs, one learner, two intra-op threads, one inter-op/native thread, and a 16 GiB
role memory limit. The stage graph is finite; elapsed time is separate from work.

Initial calibration gaps are explicit in the source-owned envelope. The actual
production calibration now supplies update/diagnostic/save/load samples. Generation
and preparation forecasts still carry `calibration_gaps=['forecast']`; their
preserved ceilings are not measured ETAs. B/C each extrapolate about 11.64 hours
of updates plus diagnostics, with a 66.47-hour learner ceiling; these are limited
sample-based planning figures, not completed arm times or worst-case guarantees.
Operational revisions cannot change the scientific population, policy, arms or
per-task budgets, or silently extend a running research allowance.

No acceptance rerun, image rebuild, private input mutation or research launch is
required to record this already successful session. Preserve the old failed runs.
