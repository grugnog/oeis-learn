# Saved snapshots import successfully; CPU fixture blocks calibration

The attached030746Z calibration archive has170members (169inventoried payloads),
zero omissions and zero payload hash/size mismatches. Its exact source is
`sha256:47c47e090ae462891dfbf3f1b2cca66feea694d064adfd4053e1e646db2a9bfb`.
The invocation takes1422.083s. [Machine evidence](pilot-calibration-operator-20261027T030746Z.json)
retains its failure, test counts, stage states and full public import proof.

The actual private prepared corpus and all three diagnostic snapshots pass their
read-only source/data/recipe/checkpoint/weight/cleanup/image bindings. No new
reference generation, corpus selection, native readmission or training occurs.
The primary engine/images,24actual HIP tests,123retirement cases and1115workflow
cases/9skips pass. This preserves the24HIP and actual import evidence at their
exact producer; it does not qualify future changed source by assumption.

CPU qualification reports819cases with one failure: the new host failure-routing
test explicitly creates its temporary directory under the installed source root,
`/opt/oeis-build`. That root is read-only in the qualification container, so
`tempfile.TemporaryDirectory(dir=root)` raises EROFS. This is a test-fixture
path defect, not a learner/model/snapshot/native-resource failure. Policy compare
and the calibration session are blocked by the failed CPU prerequisite. Zero
calibration evaluations start, cost remains UNKNOWN and long-run admission stays
blocked. No scientific negative or measured cost is inferred from unstarted work.

## Repair and validation

Only `tests/unit/test_pipeline_pilot_calibration.py` changes in the runtime/source
closure: use the platform temporary directory for a synthetic repository, create
its minimal synthetic governance inputs there, and assert that the missing-private
failure still emits successful structured calibration/final reports and a verified
archive. The real installed checkout is never a fixture output directory.
Production Python, models, data, clocks, budgets, selection, scoring, identities
and numerical policies are unchanged. The new source is
`sha256:32e77120159d9ccb1203654dc7fcbc4967252623b30b7c2e7b7193a6c7a27fdf`.

A local read-only counterexample denies source-tree directory creation: the old
fixture reproduces EROFS and the fixed fixture passes. All9calibration unit cases
pass, including failure reporting and archive verification. Full239-test host
verification is recorded in the machine evidence after completion. Actual
installed CPU qualification of this fixed fixture still belongs to the rerun.
No hypothetical permission workaround or host write access is required.

Use the unchanged [single calibration command](pilot-calibration-implementation.md#run-once-from-the-existing-checkout)
after pulling the repair, with a fresh timestamped output. Its `--inputs` remains
the complete213151Z producer directory, not this failed030746Z calibration or its
ZIP. Retain both output directories. No new download, preparation, training,
hardware soak or manually assembled commands are needed. Calibration itself was
unstarted; there is no completed calibration timing to repeat. Current affected
qualification remains inside the same finite DAG. No CI wait or long run launch.
