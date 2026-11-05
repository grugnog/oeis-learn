# Focused T072 HIP crash/recovery test

The operator authorized the missing device crash test and permits skipping ZIP
upload. The original workstation summary now verifies actual HIP PASS; see
[t072-result.md](t072-result.md). T072 is complete. The procedure below is retained
for reproducibility, not an instruction to repeat the successful output directory.

From the repository root, with the usual Podman/ROCm access:

```sh
git pull --ff-only origin 012-parameterized-adaptive-curriculum
python3 -S specs/012-parameterized-adaptive-curriculum/validation/t072.py \
  --engine podman \
  --output run-state/generalization/pac-012-t072-001
```

This stdlib helper serially invokes the canonical portable `validate` endpoint
with the dedicated `pac-crash-hip` suite. It builds the current pinned GPU image
(cached layers may be reused), checks HIP/native availability and no-Lt startup,
holds the workspace GPU lock and runs exactly one crash/recovery test. It does not
repeat original-data admission, the entire qualification suite, calibration, full
acceptance or research training. Do not supply an old image as current evidence.
Use a fresh output suffix if the chosen directory already exists; old runs remain
untouched. Do not run it alongside another learner in the same workspace.

## Actual test boundaries

- Prepare a small mechanical supported/fallback fixture with actual native checks.
- Run the installed learner on HIP with reduced acceptance cadence, real target
  loss, reward/grade observations, gradients, AdamW and hourly persistence.
- After nineteen completed/synchronized optimizer calls, the subprocess writes a
  bounded marker and executes `os._exit(86)`. No Python learner finalization runs.
  Its complete checkpoint is still initial, while newer rewards/scaler/graded
  anchors exist in the dead process. This is actual process death, not a raised
  exception, orderly pause or machine power failure.
- Launch a separate recovery subprocess. Before its first recovered gradient,
  compare the full saved policy/observer and host/device torch RNG, allowing only
  crash cost and segment fields to change. Verify restored old tensor hashes and
  zero initial policy clocks, rather than importing newer unsaved state.
- Charge positive lost tokens from the original unclosed journal and the actual
  measured crashed-process duration. Finish sixteen real recovered updates, verify
  exact charged/useful/discarded accounting within the cap and a new segment.

The HIP path uses the standard resolved bootstrap recipe; CPU wiring explicitly
uses tiny real models and cannot qualify HIP. Sparse semantic generation alone is
disabled because this test targets crash accounting, not structural accuracy;
actual loss/reward/grade, updates and persistence are retained. The stage ceiling
is 7,200 seconds, with each subprocess bounded at 2,100 seconds and inner learner
work at 1,800 seconds. Those ceilings are not an ETA.

## Small result, no ZIP

On success the helper validates a closed `pac-crash-proof/v1` record against actual
source, HIP/RNG/no-Lt policy, exit/restore counters, original JUnit/receipt/file
hashes, retained token/time charges and the immutable image. CPU, missing, skipped,
tampered or incomplete proof fails closed. It prints the sealed summary and saves:

`run-state/generalization/pac-012-t072-001/t072-summary.json`

Paste the printed JSON or attach that one small file. **No ZIP is produced or
uploaded.** Private fixture tensors/checkpoints and subprocess stdout/stderr remain
under the stage's `.runtime/pytest` directory on failure; the normal validation
output names their log location. Failed outputs remain available for diagnosis.
The public result contains hashes/counters, not private states or raw source text.

## Source and claim boundary

Adding the focused test/suite/proof validator changes the current source snapshot
and adding this helper changes its separate helper hash. The original -004 archive
remains PASS only for its original source/images; the historical closeout/audit
record is not silently rebound. The new result is a supplementary, source/image-
bound crash diagnostic, with `qualified=false` at the validation boundary, not a
replacement full qualification or research authorization. Core learner/persistence/
curriculum algorithms remain unchanged. Review actual source/image/backend and
compatibility before reconciling the missing T072 sub-proof or any later research
scope. No task marker or full feature PASS is minted from a local CPU test.
