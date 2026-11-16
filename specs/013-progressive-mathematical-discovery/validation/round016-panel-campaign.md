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
git show origin/013-progressive-mathematical-discovery:src/oeis_learn/pipeline_validation/progressive_panel_campaign.py > run-state/discovery-tools/continue-round016.py &&
python3 -S run-state/discovery-tools/continue-round016.py
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
