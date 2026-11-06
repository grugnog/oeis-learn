# R0 implementation and workstation handoff

Implemented 2026-11-05. Local CPU/contract checks are software evidence only.
Actual HIP R0 task T083 remains open. No device or full research run was executed
by the implementer. Existing -004/T072 evidence retains its historical scope.

## Disk reservation stop after the repair (2026-11-06)

The next supplied report (`report(20261106-040037).zip`) independently verifies
integrity PASS, software FAIL and scientific INCOMPLETE. Its source is exactly
the repaired `6d6153e` scope, `sha256:74820dd420936bf1c3cdd9c817440d7d675821d59ceb4aaa092a9052f6d3f20b`.
Admission passed in 1,082.34 seconds; qualification stopped after 0.306 seconds,
before validation/tests/HIP/R0. Its stdout is empty. The 46-byte stderr hash
`sha256:7c78449a36258309757cf1203463947ba2ad9fb126bfe625a2190e38f7ca825a`
exactly matches the fixed source line `ERROR: insufficient PAC reserved disk storage`.
This identifies the original cause without guessing private exception text.

For this frozen B/C/D+R0 scope the gate requires **256 GiB currently free** on the
workspace filesystem. The initial check was before image builds/admission; a
successful initial check does not establish that the later check still passes.
The old report recorded no available-byte measurement, so its historical shortfall
and actual physical peak-capacity needs cannot be reconstructed from this ZIP.
No reservation is lowered and no original files, images or checkpoints are pruned.

Check current space from the workstation repository root before another run:

```sh
python3 -S - <<'PY'
import shutil
gib = 1 << 30
required = 256 * gib
available = shutil.disk_usage('.').free
print(f'Available: {available} bytes ({available / gib:.2f} GiB)')
print(f'Frozen gate: {required / gib:.2f} GiB')
shortfall = max(0, required - available)
print(f'Current shortfall: {shortfall} bytes ({shortfall / gib:.2f} GiB)')
PY
```

This command reads the current filesystem only; it does not launch containers or
change any run. Do not rerun admission merely to obtain this measurement.

The follow-up now writes `execution/storage-check.json` and a canonical BLOCKED
terminal before validation when the unchanged gate fails. Its closed sealed
`pac-free-storage/v1` record contains required, available and shortfall bytes.
`pac-stage-failures/v3` includes verified storage checks in normal/minimal/recovery
reports. Exact known early stderr is recognized even when the historical run has
no application result; arbitrary raw stderr remains private. The old FAIL and
missing historical measurement are preserved. New admission role jobs also bind
their outer supervisor controls/logs; old unbound logs remain explicit omissions.

To enrich the old failure after pulling, use the read-only recovery command below
with the actual failed campaign root (for the previously published `-003` command,
substitute `-003` for `-002` and choose a fresh diagnostics output). Rebuilding is
not required for recovery. A future device run needs sufficient current space and
a fresh source-bound plan because this reporting/terminal repair changes installed
code. No new run is requested until the current space check is resolved. T083 and
research remain pending; the prior `-003` launch block is not a blind retry request.

Local validation: 98 affected tests plus five subtests, 38 conformance/governance
tests and 16 dependency-free tests pass (the 16 are included in the 98). Three
native qualification checks remain explicitly skipped. No actual HIP or research
was executed and CI was not awaited.

## Current repair and retry (2026-11-06)

The supplied `pac-012-bcd-r0-002` report independently verifies integrity PASS,
software FAIL and scientific INCOMPLETE. Its source fingerprint exactly matches
commit `512256b054b7dfe4c9dc1f1244f2fe876f6dc619`:
`sha256:1b0898b312070c85ff284291c5122b84498c8f9c7478bd4cf872e65aa2dcf8a1`.
Admission passed. CPU qualification then failed six R0 ownership regressions;
HIP qualification was BLOCKED, and calibration/acceptance/R0 did not start.
Both image inspections passed. This was not deadline exhaustion or a failed HIP
learner. The JUnit suite reports 1,565 executions, while its XML has 1,488 explicit
testcase elements; both scopes are preserved rather than conflated.

One failure reproduced `FileExistsError`: the canonical launcher creates the
output mount before dispatch, but R0 tried to create it exclusively again. Five
others imported the not-yet-implemented `claim_output` helper. T084 had published
the red counterexamples; T085 now implements the repair. R0 accepts the existing
mount, claims fresh private work exclusively, preserves launcher scratch and
rejects stale work or terminal records. Duplicate dispatch cannot overwrite an
original result. All five real CPU learner invocations and the complete in-block
resume proof now pass; this is not current-source HIP qualification.

The misleading public failure was a reporting defect too: `QUALIFICATION_FAILED`
had no exception text, and its private-detail hash was the hash of an empty JSON
string. The collector skipped pipeline receipts and JUnit files, and the log
projector discarded QUALIFICATION lines. `pac-stage-failures/v2` now carries
verified substage outcomes, suite and explicit-case counts, up to 64 failed-case
diagnostics and explicit truncation/omissions. Test names are checked against
public source; parameter IDs and unknown names are opaque hashes. Fixed messages
identify the missing R0 claim and existing output, while arbitrary exception text,
raw traces, protected payloads and original host paths remain private. Normal,
minimal and read-only recovery exports share the same collector. New wrapper
receipts bind outer supervisor controls/logs; unbound historical logs are explicitly
omitted rather than treated as original proof. The original nested receipts/XML
remain usable if they match the failed command's inventory.

After pulling the fixes, repackage the existing failure without running an engine,
changing original results or rebuilding anything:

```sh
git switch 012-parameterized-adaptive-curriculum
git pull --ff-only origin 012-parameterized-adaptive-curriculum
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-bcd-r0-002/campaign \
  --output run-state/generalization/pac-012-bcd-r0-002-diagnostics-001
```

To execute repaired R0, use a fresh plan/output. Production source changed, so the
old failed plan cannot be resumed onto the repair; admission, image/source
qualification and subsequent gates must bind the new closure. Preserve all old
outputs. The original frozen scientific choices and generous limits are unchanged.

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --research --bandit-only --rehearsal --stop-after-rehearsal \
  --seed 20261016 \
  --output run-state/generalization/pac-012-bcd-r0-003
```

Use a fresh sibling name if either proposed output already exists. The expected
successful R0 stop and independent verifier below are unchanged; substitute
`pac-012-bcd-r0-003` for `-001`. Actual T083 HIP evidence and R1 remain pending.
Do not continue R1 until repaired R0 has a verified diagnostic PASS.

Repair validation: 29 rehearsal tests; 80 reporting/archive/wrapper tests plus
five subtests; 38 conformance/governance tests passed. Three actual-native
qualification checks were explicitly skipped because their native build/original
G7/G8/G9 evidence is unavailable here. Eleven dependency-free reporting/ownership
tests also pass under `python3 -S` (included in the 80 above). Local development
used Python 3.12.14, PyTorch 2.9.1+cpu, pytest 9.1.1, Wasmtime 48.0.0 and NumPy 2.3.5;
this is not a pinned-image/native/HIP qualification claim. The original uploaded
archive remains independently verifiable with its original negative result.
The implementation fingerprint was refreshed without changing the 68 historical
dispositions. No actual HIP run, research run or CI wait was performed.

## Implemented scope

The temporary `validation/run.py` composes the canonical `pac-stage` operation
`pac-rehearse` after native admission, current-source qualification, production
calibration and reduced acceptance. It reuses the existing installed learner,
checkpoint store, full-state proof and loaded-checkpoint forward handlers.
There is no second training engine, fitter or adaptive policy.

B/C/D each stop at exactly 256 committed updates using the production recipe,
batch32, full original 32-family/32,768-row pool and production 8/32 clocks and real loss/recall probes. All
start from the frozen seed20261016 and identical verified random tensor content.
Diagnostic configurations have a distinct study/compatibility identity and cannot
initialize or resume a 25M-token research learner. Their individual disposition
remains PARTIAL; the separate diagnostic proof can PASS.

C also pauses after update129 (one completed update inside a bandit block), then
resumes to256 and is compared with uninterrupted C. Thus R0 performs **1,024 actual
optimizer calls** in total: B256, C256, C129+127, D256. The extra C trace is proof
work, not additional research exposure. Finite changed weights, optimizer/RNG/data/
policy/observer equality, channels/debits, actual device RNG and final loaded forwards
are checked. Wall-time observer coordinates are excluded only by the existing
qualified deterministic comparison convention.

Expected primary trace counts: B256 fixed; C192 bandit and64 review/fallback
opportunities; D256 bandit. C completes24 rewards. D intentionally stops after
gradient256 and before the last post-probe, so it has31 completed rewards and a
pending final observation. This is recorded honestly rather than fabricating a
32nd reward. Timed semantic generation is explicitly UNMEASURED in R0 to keep the full-state
comparison independent of five-second inference censoring. Its closed diagnostic
config flag is rejected for research and reduced-cadence acceptance; production
calibration and R1 retain semantic generation. R0 does not prove production spaced retention at256 updates; initial
acquisition/incidental practice is distinct from graded due review. Existing
reduced-cadence acceptance separately covers acquired-anchor grades and recovery.

Full learner/proof budgets are enclosed bottom-up even for the short stop schedule;
calibration can raise them before work. No six/twelve-hour truncation is added.
The initial B/C/D+R0 disk reservation is256GiB free; actual model sizes additionally
reserve16 diagnostic snapshot generations alongside the research peak. Originals
are preserved and GPU roles remain serial.

## Run R0, stopping before research

Use the existing workstation repository and a fresh output name. This freezes the
entire selected R1 campaign before any new structural outcomes, then stops after R0.

```sh
git switch 012-parameterized-adaptive-curriculum
git pull --ff-only origin 012-parameterized-adaptive-curriculum
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py run \
  --engine podman \
  --prepared run-state/generalization/fresh-p1-v2-20261103T035534Z \
  --pilot run-state/generalization/fresh-p1-hourly-20261104T093555Z \
  --research --bandit-only --rehearsal --stop-after-rehearsal \
  --seed 20261016 \
  --output run-state/generalization/pac-012-bcd-r0-001
```

Expected successful stop: command exit0, `rehearsal_status=PASS`, workflow/export
PARTIAL, stop reason `REHEARSAL_COMPLETE_RESEARCH_PENDING`; scientific campaign
status INCOMPLETE because R1 has not started. This is not a completed research PASS.
The bounded `rehearsal/rehearsal.json` payload itself reports scientific UNMEASURED.

R0's actual rehearsal work is provisionally **30–60 minutes**, including the extra
C comparison trace, plus image builds, admission, qualification, calibration and
acceptance. Those stages add time and depend on current-source measurements. Old
samples imply about15 minutes of updates before real loss/recall observation,
model load/checkpoint/proof and export overhead. Timed semantic portfolios are
explicitly omitted from R0, so this is a generous planning range rather than a
claim of measured semantic work. This is not a measured runtime guarantee or ceiling.

Independently inspect the report without an engine or GPU:

```sh
PYTHONPATH=src python3 -S - <<'PY'
import json
from oeis_learn.pipeline_validation.generalization_archive import verify_archive
v=verify_archive('run-state/generalization/pac-012-bcd-r0-001/campaign/report.zip')
print(json.dumps(v,indent=2))
assert v['integrity']=='PASS' and v.get('rehearsal_status')=='PASS'
PY
```

## Continue R1 using the same frozen plan and completed gates

After inspecting R0, keep the same source checkout/images. Do not edit the sealed
plan, pull new code between R0/R1 or use the diagnostic weights as initializers.

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py resume \
  --plan run-state/generalization/pac-012-bcd-r0-001/inputs/plan.json \
  --previous run-state/generalization/pac-012-bcd-r0-001/campaign \
  --output run-state/generalization/pac-012-bcd-r1-001
```

This verifies and reuses compatible admission/qualification/calibration/acceptance/
R0 artifacts, then evaluates the retained A reference and runs fresh B/C/D serially
at25M charged target tokens each, with final structural evaluation and analysis.
The reference does not establish a matched representation effect. The one-seed
comparison is exploratory; no extra A/seed, winner adoption or assessment is added.

Plan roughly **35 hours of learner work**, plus retained-reference/final evaluation
and reporting; reserve a multi-day uninterrupted window. Recalibrated ETAs take
precedence. The old approximately66-hour per-learner ceilings are operational
margins, not expected training times. The R1 report is at
`run-state/generalization/pac-012-bcd-r1-001/report.zip` (resume outputs have no
extra `campaign/` component).

For one uninterrupted invocation instead, omit `--stop-after-rehearsal` from the
first command and choose a fresh output; it continues only after R0 passes.

## Interrupted R0/R1 and read-only recovery

To finish an interrupted R0 without starting research:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py resume \
  --plan run-state/generalization/pac-012-bcd-r0-001/inputs/plan.json \
  --previous run-state/generalization/pac-012-bcd-r0-001/campaign \
  --output run-state/generalization/pac-012-bcd-r0-resumed-001 \
  --stop-after-rehearsal
```

An unfinished R0 is one bounded diagnostic stage and is rerun; already sealed gates
are reused. The in-stage C pause/resume is a deliberate full-state proof, not an
external process-death recovery claim. Once resumed R0 passes, use its completed
output as `--previous` for R1, retaining the original plan path.

To resume an interrupted full campaign from its actual complete model state:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py resume \
  --plan run-state/generalization/pac-012-bcd-r0-001/inputs/plan.json \
  --previous run-state/generalization/pac-012-bcd-r1-001 \
  --output run-state/generalization/pac-012-bcd-r1-resumed-001
```

Spent/discarded research tokens and time remain charged. The smallest compatible
unfinished dependency is resumed; successful stages are not automatically repeated.
Changed source/image/scientific scope requires a newly reviewed plan/qualification.

To package existing R0 evidence only, without training or modifying original results:

```sh
python3 -S specs/012-parameterized-adaptive-curriculum/validation/run.py recover \
  --previous run-state/generalization/pac-012-bcd-r0-001/campaign \
  --output run-state/generalization/pac-012-bcd-r0-recovered-001
```

For R1 recovery, substitute its actual output as `--previous` and a fresh recovery
output. Failures preserve first/secondary errors and produce a verified rich or
minimal archive; the helper prints the exact archive path. Missing or invalid R0
proof cannot exit as a successful intentional stop.

## Contracts and validation boundary

Rehearsal-enabled plans use `pac-serial-plan/v2` and reports use
`pac-public-evidence/v2`; historical v1 interpretation is unchanged. Both full and
stopped reports require the frozen R0 stage. The stdlib offline verifier reconciles
closed diagnostic counts, bound installed tensor proofs, source/environment/seed,
all prerequisite proof closures and terminal states. Research status is still
independently reconstructed from the original final analysis/outcomes. No raw logs,
private weights/state, original host paths or protected truth enter the public proof.

Local checks and exact outcomes are recorded with T079–T082 in `../tasks.md`.
The CPU handler fixture uses explicit small-data/model/device adapters while running
real forward/backward/optimizer updates, checkpoints and in-block restore; it does
not establish current-source HIP, production-pool or performance readiness. Actual
T083 evidence must come from the workstation command above. No CI wait is required.
