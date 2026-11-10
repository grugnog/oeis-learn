# Consolidated workstation handoff

The entrypoint is `python3 -S scripts/discovery_hip_handoff.py`. The host uses only
Python 3.11+'s standard library and the selected container engine. Device work uses the
existing root Dockerfile, immutable inspected image IDs, foundation launcher,
GPU lease and canonical `discovery` commands. No host venv is needed.

## First command: tiny rehearsal

From the repository root, on the Ryzen AI Max+ 395 workstation:

```sh
git switch 013-progressive-mathematical-discovery &&
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman \
  --session run-state/discovery/013-hip-001 \
  rehearse \
  --reference-campaign run-state/generalization/pac-012-bcd-r1-002
```

This builds/reuses the pinned CPU qualification and ROCm GPU image targets, audits
retained inputs, runs base qualification, and constructs a sealed diagnostic
campaign. Missing historical research inputs do not prevent the diagnostic
fixture run. The selected model is the registered width-128, four-layer profile,
seed 13, dropout 0.1, FP32. Q covers both four-/20-input profiles and microbatches
7/16. Preparation/freeze and the complete registered installed acceptance follow:
26 CPU/native scenarios and seven actual-HIP scenarios, plus the independent
assessment contract tests. These exercise the real train/restore, source,
component, report and verification paths with tiny fixtures; they confer no
scientific improvement or research-cohort qualification.

Defaults are one serial GPU owner, 64 GiB container memory, eight CPUs, and a
24 GiB host free-memory floor. The container memory cap is not a claim that all
128 GiB of shared system memory is available as GPU memory. Native services and
host overhead remain relevant. No precision, driver, BIOS or GPU-memory setting
is automatically changed. If several render nodes exist, use the same explicit
`--render-node /dev/dri/renderD...` on every phase after selecting the correct
node. Initial image builds are independently bounded to one hour each; base
qualification has its existing 6,600-second aggregate cap, tiny Q 1,800 seconds,
and post-freeze acceptance 3,600 seconds. These are ceilings, not predicted
runtimes. No long research training starts from `rehearse`.

The shareable result, on success or failure, is:

```text
run-state/discovery/013-hip-001/share/rehearse.zip
```

Return that ZIP for review before starting the scientific campaign. The support
archive is explicitly nonqualifying: original producer receipts establish actual
results. Task statuses T082/T088–T111 remain unchanged until their original
physical/scientific evidence is reviewed.

## Following phases

| Phase | Concrete prerequisites | Work performed |
| --- | --- | --- |
| `prepare-research` | Successful rehearsal; an original research blueprint with qualification, preparation, exact-input calibration and acceptance plans | Selected research Q, complete bounded exact-input calibration, preparation, freeze, campaign-bound acceptance |
| `run-research` | Successful frozen research acceptance | Canonical run/resume, report, independent bundle verification |
| `assess` | Same campaign acceptance plus an original locked selection supplied with `--selection` | Separately authorized final assessment, report, verification |
| `collect` | Existing session | Repackage retained evidence without launching work |

Use the same engine, session and resource arguments across phases. A research
blueprint may be named `research_blueprint` in the original `retained-inputs.json`
inventory or supplied through `--blueprint` with a workspace-relative path.
The script does not invent a retained checkpoint, exposure history, fresh cohort,
held-out truth, scientific task graph or measured timings. Missing concrete inputs
produce `prepare-research/deficits.json` and a support ZIP. The long campaign must
use the real selected architecture, paired seeds and preregistered budgets; the
tiny recipe is not silently promoted to the scientific configuration.

For example, once that original blueprint has been resolved in the inventory:

```sh
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py --engine podman \
  --session run-state/discovery/013-hip-001 prepare-research
```

Then, only after reviewing preparation/acceptance and finite feasibility:

```sh
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py --engine podman \
  --session run-state/discovery/013-hip-001 run-research
```

Repeat the last command for a canonical continuation if the previous invocation
is partial or paused. It resumes the original run and accounting; it does not
allocate fresh scientific budget. A complete invocation is not rerun. Optional
branches still require their original registered eligibility. There is no
unconditional loop that forces every optional research task to execute.

Sustained T091 evidence requires a concrete, prospectively validated two-task
protocol, including actual duration, interruption point and recovery. The runner
records 30-second host telemetry but does not inject an arbitrary kill into an
unmodified scientific schedule or claim that heartbeats prove sustained training.
Its sustained status remains `PENDING_ACTUAL_PROTOCOL_RECONCILIATION` until those
original observations exist. Resolve that protocol against the selected eligible
research path after the first hardware results, within the original allowance;
do not add an unbudgeted soak or extend a failed task. The existing
`preflight.sustained` validator remains the protocol gate.

## Failure, recovery and evidence

Independent CPU work runs even if the GPU image fails. Independent Q, calibration
and device acceptance cases use separate processes and preserve per-case results.
After a failed device case, a fresh HIP health probe must succeed before later
GPU cases run. Dependent work is blocked. Every case retains a finite share of
the original aggregate deadline; a timeout never resets the allowance.

The journal is atomically persisted before and after each stage. A support ZIP is
atomically replaced after each stage and in final cleanup. It includes original
command lines, image/source identities, stdout/stderr, JUnit, first exceptions,
per-case outcomes, inspection/OOM/cleanup records, original public campaign
archives, runtime facts and memory/disk/thermal observations. Missing sensors
remain explicit. Collected metadata is capped at 512 MiB, with bounded log
prefixes/tails and an omissions manifest. Private truth, checkpoints and model
tensors remain local; original public archives are retained intact when they fit.

Ctrl-C/SIGTERM/SIGHUP trigger cleanup and final packaging. SIGKILL, power loss,
or a full disk cannot guarantee a new ZIP; previous atomic archives and original
stage files remain. To collect without retrying work, even after a source update:

```sh
python3 -S scripts/discovery_hip_handoff.py --engine podman \
  --session run-state/discovery/013-hip-001 collect
```

This produces `share/collect.zip`. Normal exit codes are 0 for completed wrapper
stages, 1 for failures, 2 for blocked prerequisites, and 130 for interruption.
Consult original scientific dispositions even when the wrapper exits zero.

Changed source/resource scope blocks execution before reuse. Completed stages
check their original output hashes; interrupted/failed stages are retained rather
than silently retried with fresh budgets. After a code fix, preserve the old ZIP
and use a new session name for requalification. A dry preview is available with
`--dry-run` before the phase; it performs no build, device launch or file writes.

## Research input registration after rehearsal 007

The original B/D manifest is committed at
`validation/research-input-resolution.json`. A separate `register-inputs` phase
verifies those actual workstation files, all checkpoint sidecars and ledger
bindings; imports all supplied original cells/probes/observed fresh-002 exposure
into a private SQLite index in the CPU qualification container; and inventories
existing indexed OEIS source manifests. Independent checks continue after a
missing original or source. It neither loads tensors nor starts GPU work.

```sh
git switch 013-progressive-mathematical-discovery &&
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman \
  --session run-state/discovery/013-research-inputs-001 \
  register-inputs \
  --reference-campaign run-state/generalization/pac-012-bcd-r1-002 \
  --input-manifest specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json
```

Share `run-state/discovery/013-research-inputs-001/share/register-inputs.zip`,
including when the command exits 2. The shared manifest does not identify the
immutable indexed OEIS snapshot for fresh transfer panels. The first invocation
therefore records `ORIGINAL_SOURCE_SELECTION_REQUIRED` while completing the
independent original-data checks/import. The ZIP lists existing candidates under
`data`, `downloads` and `run-state`, with a finite scan limit and explicit
truncation. No source is downloaded or selected automatically.

Once an exact existing source is selected, repeat the same command with
`--oeis-snapshot WORKSPACE_RELATIVE_MANIFEST_PATH`. Completed imports are reused
only after their original hashes still match; the source choice cannot replace a
completed projection. Projection preserves exact offsets, validates original
term/metadata hashes, selects a fixed outcome-blind reservoir, excludes formulas,
and retains missing/corrupt members without substituting later candidates. Its
600-second/1-GiB source-read and 128-MiB candidate-output ceilings are explicit.
Exposure import is limited to two hours and an 8-GiB private index. The CPU image
build has a separate 3,650-second ceiling. Private terms, index and tensors stay
local; summaries, exact references, failure details and container inspection are
packaged on failure as well as success.

**Registration is not research preparation.** Even exit 0 means only
`INPUTS_REGISTERED`, with `research_ready=false`. The emitted fixed-reference
capability/diversity proposal remains non-executable. Automatic fresh panel/support
construction, a concrete scientific task graph, measured sensitivity/feasibility,
and its connection to exact calibration/freeze are still outstanding. The existing
`prepare-research --blueprint` route requires those concrete original inputs; the
registration proposal must not be passed as that blueprint. T089 remains open.

These software changes alter the source closure. Preserve rehearsal 007 as the
original successful diagnostic result; it cannot qualify this new closure or the
retained research architecture. Registration requires no repeat HIP run. Finish
the remaining research construction software before the next hardware handoff.
