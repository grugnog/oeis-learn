# Continue after the successful panel revision

The workstation's `panel-revision.zip` is a successful outcome-blind data repair.
It supplies revised materials and passes all ten panel guards, with byte-identical
training inputs. It is not a frozen campaign or permission to start learning.

The host continuation in `progressive_panel_campaign.py` binds these materials
into the canonical compiler, checks that scientific settings/budgets and learning
compatibilities are unchanged, and reuses the existing training-only evaluation,
trajectory and exact-input calibration evidence. It then invokes canonical
Podman-backed preparation replay, freeze and installed campaign acceptance.
Only after all succeed does it automatically start the research campaign.

## Run without changing the qualified checkout

Keep the workstation at `2faa8e43fb772b7ad5d4232fe99ff93b3d96b81d`. Do not pull,
switch branches or rebuild. Fetching the helper into `run-state` leaves the
qualified source closure unchanged. The existing hash-bound
`run-state/discovery-tools/repair-round016.py` is also required; retain it.

```bash
cd ~/workspace/oies-learn &&
git fetch origin 013-progressive-mathematical-discovery &&
mkdir -p run-state/discovery-tools &&
git show 1c1d24182923:src/oeis_learn/pipeline_validation/progressive_panel_campaign.py > run-state/discovery-tools/continue-round016-v2.py &&
python3 -S run-state/discovery-tools/continue-round016-v2.py
```

The helper reads Podman, memory, CPU and device settings from the original
round016 session. No extra Python dependencies are needed on the host. Original
ownership, source, image bindings and completed artifact hashes are verified.
It does not build images, repeat qualification/rehearsal, repeat calibration or
timing measurements, generate panels, change the scientific design, or expose
assessment scores. The outstanding campaign-bound acceptance includes its
required installed CPU/native and HIP checks; it cannot be replaced by the old
rehearsal because the actual research campaign must be bound.

The canonical replay still takes time: the prior successful prepare was about
50 minutes, and freeze must validate the revised panels and the full calibration
publication. Neither step is a fresh 1,296-case calibration run. Exact remaining
runtime is not yet measured; the invocation continues into research without
requiring another command after preparation.

## Recovery and evidence

New metadata is under `prepare-research/panel-campaign`. The original failed
freeze and traceback are preserved at `panel-campaign/original-failed-freeze`;
prior prepare/freeze journal entries remain in history. The revised frozen
campaign uses the standard `prepare-research/frozen` path so the installed
run/resume machinery consumes it normally.

Metadata, replay and freeze share the unused original preparation allowance,
subtracting the previous successful repair, failed freeze, audit and panel-repair
elapsed time. This avoids another independent two-hour freeze timeout without
resetting a reservation or increasing scientific task caps. Acceptance retains
its original 7,800-second envelope, including startup/cleanup headroom.

Reinvoke the same saved Python command to reuse completed preparation and use
canonical research resume. A completed invocation and its report/verification
are reused. Failed or unknown preparation executions are retained for explicit
reconciliation; no automatic new measurement or budget reset is permitted.

Diagnostics are packaged on failure as well as success:

- `run-state/discovery/013-research-round-016/share/prepare-research.zip`
- `run-state/discovery/013-research-round-016/share/run-research.zip`

The latter appears once research starts. Weights, raw data and private panel
truth remain local. Canonical acceptance, run/report and independent verification
retain their existing execution authority; local tests do not constitute HIP
validation of the revised campaign. T220 remains pending that workstation run.


## Local validation

- 114 unit/construction tests passed across the four supported research designs.
  The changed-panel reconstruction preserves all 352 tasks, 1,296 exact-input
  cases and original scientific budgets; requests and study bindings validate.
- 19 canonical prepare/replay/freeze integration tests passed with explicit
  simulated device boundaries, including original reservation preservation.
- 11 continuation tests passed against the actual pinned round016 code. They
  exercise the real journal, allowance and workflow dispatch, completed-step
  reuse, paused research resume, descendant blocking and ZIP collection. Only
  external execution/measurement producers are doubled; those doubles confer
  no hardware authority.

The complete rebind also ran under system Python 3.13 `-S` against the actual
pinned package and original reservation helper. All 352 closed requests and
study bindings passed without importing Torch, NumPy, Optuna or jsonschema.
The qualified source digest remained
`sha256:e3592c86ab55c8aad83763cbaae686eaca895bfe58b560e1b64b422231b17ce5`.


## V2: recover the study-derived seed guard

`prepare-research(20261116-152014).zip` records successful reconstruction in
275 seconds, followed by a 5.3-second metadata rejection. No new preparation,
freeze, acceptance or research command ran. The original and revised construction
both contain 352 tasks and 1,296 calibration cases with identical registered
design and resource ceilings. All 16 screening-profile seeds differ because
these seeds are derived from the study identity, which binds the repaired panels.
The original test changed only diversity membership and missed that dependency.

V2 verifies the exact canonical formula for both sides, permits only those
prospective screening-seed changes, and keeps registered fixed seeds and other
settings unchanged. The installed study validator requires this derivation;
keeping obsolete screening seeds would be invalid. Completed measurements remain
subject to exact input compatibility and canonical replay. The public
`panel-campaign/calibration-reuse/task-rebind.json` reports every seed change and
any unexpected field mismatch, without panel truth.

Keep the old `continue-round016.py` file: its original hash is part of the
retained enrollment. Save V2 under the separate filename in the command above.
V2 reuses the completed roster, feasibility and construction, and the original
remaining allowance; it retries only the known original metadata failure.
The original failed freeze is archived from its retained journal history even
though its current row is dependency-blocked. Successful preparation still rolls
straight into canonical freeze, campaign acceptance and research. No tests,
qualification or calibration measurements are launched by this metadata repair;
the outstanding campaign-bound acceptance still runs its mandatory checks.


V2 validation: 67 material/recovery and canonical reuse integration tests passed,
plus 19 focused checks against the pinned package. The expanded full compiler
regression passed with development, diversity and tuning-panel changes: 352 tasks,
1,296 exact-input cases, and 256 correctly derived seed bindings across 16 trials.
Canonical requests/studies validate; fixed/confirmation seed and budget tampering
are rejected. The actual uploaded FAILED/BLOCKED journal was exercised under
Python 3.13 `-S`, including preservation of the failed freeze from journal history.
No private workstation measurement was regenerated during local validation.
