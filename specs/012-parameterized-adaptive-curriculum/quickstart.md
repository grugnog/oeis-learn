# 012 operator handoff

The R0 rehearsal is implemented. Use the exact current-source commands in
[the R0/R1 handoff](validation/r0-implementation.md): first the 256-update B/C/D
rehearsal and report, then resume the same frozen plan into the long campaign.
Actual HIP R0 now verifies PASS for the exact source/images in the operator's
`pac-012-bcd-r0-00` directory. T083 is complete. Use the
[exact R1 resume command](validation/r0-implementation.md#verified-actual-hip-r0-and-exact-r1-resume-2026-11-06)
with that existing spelling and unchanged runtime/helpers/images to reuse all five
completed gates. No rebuild or repeated R0 is needed for that compatible plan.
Historical acceptance/repair commands below are retained for provenance.

The operator's `pac-012-acceptance-004` run passed admission, CPU/HIP qualification,
production calibration and reduced HIP acceptance. Independent archive verification
reproduces software PASS and scientific UNMEASURED. **That historical acceptance needs no retroactive repair.** Current R0/R1
uses its own source/image qualification and acceptance gates.

[Phase E closeout](validation/closeout.md) records seven completed evidence tasks,
including the verified [actual HIP crash result](validation/t072-result.md). Only
conditional T073 research analysis remains deferred.
[Actual device evidence](validation/hip-acceptance.md) and the bounded
[machine record](validation/evidence/acceptance-004-record.json) retain exact original
source/image/plan identities. The complete uploaded ZIP is not published to GitHub.
No runtime or orchestration-helper byte changed during this documentation closeout.

The original 011 failed 256-task prospective command remains superseded, not a
012 run. The helper's default scope is acceptance; research requires its own
explicit frozen choice after all applicable foundation proof is complete.

## Completed focused T072 proof

The operator ran the dedicated HIP test and supplied its small JSON summary.
[The result](validation/t072-result.md) verifies old model/policy restoration,
910 discarded tokens retained, sixteen recovered updates and no refunded crash
time. **T072 is complete; no rerun or ZIP upload is needed.** The
[focused procedure](validation/t072-run.md) remains available for reproduction.

The focused test has its own source/image/helper closure. Original -004 PASS
remains evidence for its original producer; the supplemental diagnostic does not
replace full qualification for a new research plan. No research was launched.

## Required workstation inputs

Run from the existing repository on the normal workstation, with Python 3's standard
library, a usable **Podman** connection, the established AMD/ROCm device access and
enough free disk for the complete live graph (initially at least 128 GiB for B/C;
64 GiB more for each additional fresh A or D). Dependencies and native code are
built through the pinned Dockerfile/locks. No host package/driver/service change is
required. Docker is also supported by choosing `--engine docker` before freezing.

Keep these original roots and all files referenced by their sealed inventories:

- `run-state/generalization/fresh-p1-v2-20261103T035534Z`: original prepared corpus,
  all sixteen exclusion inputs and original producer receipt.
- `run-state/generalization/fresh-p1-hourly-20261104T093555Z`: original full pilot,
  recipe/descriptors, retained checkpoints, weights and producer lineage.
- Referenced `run-state/generalization/uniform-import` provenance and the original
  qualification lineage where the pilot references them.

The historical qualified root
`run-state/generalization/observable-eval-20261104T193226Z/acceptance/report/qualified`
is context, not current qualification for changed production code. The helper builds
and qualifies the current immutable image pair. A public ZIP cannot replace private
weights, optimizer/RNG state or protected exclusion inputs. Missing originals are a
specific preflight failure; the helper neither invents provenance nor silently
substitutes new draws. Use a fresh output name if the chosen output already exists.

## Historical repair notes (preceding the successful -004 run)

### Original admission failure

The original `pac-012-acceptance` directory remains failed and immutable. The
operator-authorized [scope amendment](validation/structural-amendment.md) freezes
new quotas under `pac-structural-scope/v2`, using the identical 2,288 draws and all
sixteen original exclusions. The current 128-task population has 23 families,
including 68 certified tasks across 15 families. It is a disclosed feasible subset,
not a completed result for the old 27/18-family scope. Training data and learner
policies are unchanged. The printed/sealed plan identifies the new scope.

Pull the repair and run with the fresh `-004` output below. Do not resume the failed
old plan into the new scientific population. Current images rebuild automatically
with cached layers where compatible; old weights/data stay untouched. The operator's
second run completed native admission for its frozen source/images.
The qualification projection repair below changes that closure, so current
workstation/native/HIP acceptance still must run.

To recover richer diagnostics from the original failed directory without device
work or mutation, after pulling the repair run:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-acceptance/campaign \
  --output run-state/generalization/pac-012-admission-recovered-001
```

This uses original role/control files whose hashes match the original inventory.
The report preserves the innermost known cause, exact opaque quota deficits where
originally recorded, and wrapper/cleanup/export errors. Known cause messages are
safe fixed vocabulary; arbitrary raw exception text and host paths remain private.
Missing or unverified detail is explicitly omitted. Recovery does not rerun admission.

### Second run's qualification failure

The `pac-012-acceptance-002` archive records admission PASS (1,155.7 seconds),
followed by qualification FAIL before its tests. The isolated validator omitted
`private/structural-targets.json` even though the admission receipt referenced it.
The repaired projection verifies and copies this protected dependency into the
custodian; learner/generator views remain unchanged. Known missing-artifact errors
now have a readable safe message, including during recovery of old error hashes.

Use the fresh `-004` invocation below. Changed production source requires a new
sealed plan and rebuilt images. It repeats admission under that new closure; the
successful `-002` receipt cannot be silently rebound to different source/images.
Keep the entire old directory. For improved diagnostics alone, without rerunning
admission or modifying old results:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-acceptance-002/campaign \
  --output run-state/generalization/pac-012-qualification-recovered-001
```

### Third run's calibration failure

The `pac-012-acceptance-003` archive records admission PASS (1,065.0 seconds) and
CPU/HIP qualification PASS (715.4 seconds; 1,496 CPU and 27 HIP tests). Calibration
then failed with `KeyError('source')` before its first native timing completed.
Learner rows intentionally omit source text; calibration now decodes the paired
literal target tokens that preparation already verified. Learner projections and
training policy remain unchanged. A complete CPU handler regression now exercises
64 updates, feedback, save/load and all calibration samples, using explicit device
and native-service adapters; it is not production performance evidence.

Use the fresh `-004` invocation below. Production source changed, requiring a new
sealed plan and rebuilt image pair, including repeated admission and qualification.
The old successful receipts remain evidence for their original source/images;
keep the entire `-003` directory. To obtain readable diagnostics alone:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-acceptance-003/campaign \
  --output run-state/generalization/pac-012-calibration-recovered-001
```

## Recorded successful operator block

The operator executed this acceptance-only invocation and supplied its verified
report. It is retained for provenance, not an instruction to overwrite or repeat
`-004`. Any separately authorized new scope must use a fresh output path. The
implementer performed read-only verification, not another device run.

```sh
git fetch origin
git switch 012-parameterized-adaptive-curriculum
git pull --ff-only origin 012-parameterized-adaptive-curriculum
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --seed 20261016 \
  --output run-state/generalization/pac-012-acceptance-004
```

The helper audits originals and capacity, builds the pinned qualification/GPU
images, writes and prints the sealed plan and its identity, then serially executes
full CPU/native structural and training-probe admission, current qualification,
representative production-shape calibration and small installed HIP acceptance.
Acceptance reaches actual changed finite parameters, supported/fallback targets,
nondegenerate feedback, acquired then graded review, both complete-state resume
boundaries and a freshly loaded final-checkpoint forward evaluation. Diagnostic
weights cannot initialize research. Failures suppress dependent work and still
produce a verified normal or minimal archive.

The initial plan is
`run-state/generalization/pac-012-acceptance-004/inputs/plan.json`; its `identity`,
scientific scope, source/images, helper hash and finite stage graph are printed
before admission. Calibration may raise and separately seal **operational** limits
before research. It cannot change scientific choices, add arms or extend a running
research allowance. The revision is `campaign/operational-freeze.json`.

The normal shareable result is
`run-state/generalization/pac-012-acceptance-004/campaign/report.zip`. A failure during
input inspection or image building instead writes `bootstrap-failure/report.zip`
under the selected output. If rich proof collection fails, the verified minimal
archive retains the original failure plus the export error and names omissions.
If even minimal export fails, the helper prints the last-resort `workflow.json`.
Private originals and prior failure records remain unchanged.

## Scientific scope and optional research

Acceptance performs full feasibility admission but no 25M-token research learner.
For an explicitly authorized research campaign, choose a **new output root** and
add `--research` to the same command before plan creation. The frozen screen then
runs retained structural baselines and serial fresh B/C learners, each capped at
**25,000,000 non-padding target tokens**, followed by final structural scoring and
case analysis. B uses fixed allocation with parameterized targets; C adds the
published family bandit and charged per-example review over the identical finite
32-family/32,768-row concrete pool and training-only probes. Retained A is labeled
REFERENCE_ONLY unless complete compatibility proof exists; no causal B–A claim is
made from a missing historical match. C–B measures allocation plus review.

Optional `--fresh-control` adds a matched fresh literal A; `--bandit-only` adds D
for allocation without review. Choose these only with the research scope **before**
freezing; they add their full learner/evaluation/storage reservations. There is no
automatic extra seed, ablation, winner adoption or assessment execution. The
parameter endpoint stays DEFERRED. Structural evaluation preserves 128 tasks/23
families (68 certified/15 families primary, 26 ambiguous/3 families, 34 unresolved/5 families), all original
exclusions, 20 visible/100 scoring terms, eight attempts/60-second portfolios and
seven controls. Interpretation needs at least 65/68 certified completions per
compared arm; a single seed remains exploratory.

## Compatible resume and read-only recovery

For an interrupted campaign with unchanged scientific source/images/plan, use a
fresh sibling output. The helper verifies original artifact bytes, process and
cleanup receipts, reuses compatible successful stages and resumes a failed learner
only from complete compatible model/optimizer/RNG/data/adaptive state. It never
restores a newer policy journal onto older weights or refunds discarded work.
An incompatible full checkpoint blocks instead of silently restarting a learner.

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py resume \
  --plan run-state/generalization/pac-012-acceptance-004/inputs/plan.json \
  --previous run-state/generalization/pac-012-acceptance-004/campaign \
  --output run-state/generalization/pac-012-resumed-001
```

A production-source/image/backend change requires a newly frozen plan and affected
qualification. A temporary helper-only change has its own requested/observed hash
and does not invalidate an otherwise identical learner producer. Never edit the
old sealed plan or clear failed outputs to obtain a PASS.

Recovery alone packages existing evidence; it runs no training or device work and
does not rewrite the old outcome:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-acceptance-004/campaign \
  --output run-state/generalization/pac-012-recovered-001
```

It prints the exact verified archive path, including a `-minimal.zip` suffix if
only minimal evidence could be verified.

## Measured forecasts versus generous ceilings

The actual production-shape calibration now projects 6.72 hours of updates plus
4.92 hours of diagnostics per B/C arm (11.64 hours total), with a 66.47-hour learner
ceiling. These limited-sample planning figures exclude preparation and structural
evaluation, and are not measured completed-arm times or worst-case guarantees.
Historical literal runs took about 147–160 minutes per 25M-token arm; that is
context, not this adaptive experiment's ETA.
Calibration records actual short/long-family updates, cold/warm native execution,
early/later row access, production-size save/load and full reward/grade/common-panel
costs, with sample counts and uncertainty. Inspect the exported
`calibration/calibration.json`: `update_forecast_seconds` plus
`diagnostics.forecast_seconds` is the sample-based work estimate for each arm;
`learner.seconds` is the larger, margined **ceiling**, not its ETA. Raw samples and
worst observations disclose the finite measured range; they are not a worst-case
guarantee. Preparation, qualification, generation, startup and export are additional
stages, with actual elapsed costs recorded separately.

Initial floors are 24 hours for preparation and 12 hours per new learner, raised
when the complete forecast requires. The seven-control generation graph has its
own generous ceiling; the campaign sums every selected stage plus at least one
hour of final reporting/recovery. Missing measurements do not mean zero. The
helper prints explicit unavailable progress initially and bounded supervision
signals during work; elapsed heartbeats are never called completed work. Scientific
60-second portfolios and five-second semantic probes retain their meaning.

## Local verification boundary

The implementation ledger records the exact combined CPU/contract/integration
command and outcomes. It includes real CPU tensors/clean and crash recovery,
Docker/Podman argv/mount/deadline checks, stdlib-only successful public verification,
checkpoint reads and analysis, and real success/negative/minimal archive fixtures.
Synthetic device labels exercise archive wiring only. The supplied -004 archive
now provides actual engine/native/quota/HIP acceptance and short production
calibration evidence. The supplemental actual HIP crash-debit proof now closes T072; T073 research
is deferred. Longer-run performance and scientific outcomes remain unmeasured.
Historical 011 T174 and T177–T181 remain unchanged; CI was not awaited.
