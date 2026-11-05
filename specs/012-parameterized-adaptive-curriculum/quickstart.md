# 012 operator handoff

Phases A–D are implemented and locally validated. Actual original-data/native,
container and HIP evidence remains Phase E. Do not reuse the failed 011 256-task
prospective command as a 012 run. The temporary helper composes canonical installed
operations; the default command below performs software acceptance and exports its
evidence. It does not start research training or expose assessment truth.

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

## Repair after the original admission failure

The original `pac-012-acceptance` directory remains failed and immutable. The
operator-authorized [scope amendment](validation/structural-amendment.md) freezes
new quotas under `pac-structural-scope/v2`, using the identical 2,288 draws and all
sixteen original exclusions. The current 128-task population has 23 families,
including 68 certified tasks across 15 families. It is a disclosed feasible subset,
not a completed result for the old 27/18-family scope. Training data and learner
policies are unchanged. The printed/sealed plan identifies the new scope.

Pull the repair and run with the fresh `-002` output below. Do not resume the failed
old plan into the new scientific population. Current images rebuild automatically
with cached layers where compatible; old weights/data stay untouched. A capacity
proof from the uploaded counters is implemented and tested, but actual new native
agreement and workstation/HIP acceptance still must run.

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

## One operator block

This is the Phase-E invocation, provided for handoff; it was **not run on HIP**
during implementation. Its syntax and routing were checked with actual `--help`,
`python -S` and the fixture campaign.

```sh
git fetch origin
git switch 012-parameterized-adaptive-curriculum
git pull --ff-only origin 012-parameterized-adaptive-curriculum
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --seed 20261016 \
  --output run-state/generalization/pac-012-acceptance-002
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
`run-state/generalization/pac-012-acceptance-002/inputs/plan.json`; its `identity`,
scientific scope, source/images, helper hash and finite stage graph are printed
before admission. Calibration may raise and separately seal **operational** limits
before research. It cannot change scientific choices, add arms or extend a running
research allowance. The revision is `campaign/operational-freeze.json`.

The normal shareable result is
`run-state/generalization/pac-012-acceptance-002/campaign/report.zip`. A failure during
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
  --plan run-state/generalization/pac-012-acceptance-002/inputs/plan.json \
  --previous run-state/generalization/pac-012-acceptance-002/campaign \
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
  --previous run-state/generalization/pac-012-acceptance-002/campaign \
  --output run-state/generalization/pac-012-recovered-001
```

It prints the exact verified archive path, including a `-minimal.zip` suffix if
only minimal evidence could be verified.

## Measured forecasts versus generous ceilings

There is no measured 012 workstation ETA yet. Historical literal runs took about
147–160 minutes per 25M-token arm; that is context, not an adaptive forecast.
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
Synthetic device labels exercise archive wiring only. Actual engine/native/HIP
execution, original full quotas and production performance remain T067–T074;
historical 011 T174 and T177–T181 remain unchanged. No CI wait is needed for this
handoff.
