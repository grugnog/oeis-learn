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
  --session run-state/discovery/013-research-inputs-002 \
  register-inputs \
  --reference-campaign run-state/generalization/pac-012-bcd-r1-002 \
  --input-manifest specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json
```

Share `run-state/discovery/013-research-inputs-002/share/register-inputs.zip`,
including when the command exits 2. The shared manifest does not identify the
immutable indexed OEIS snapshot for fresh transfer panels. The first invocation
therefore records `ORIGINAL_SOURCE_SELECTION_REQUIRED` while completing the
independent original-data checks/import. The ZIP lists existing candidates under
`data`, `downloads`, `tmp-oeis-bulk` and `run-state`, with separate finite
scan limits and explicit truncation. It checks manifests before large directory
inventories, skips raw member directories, and retains original bulk cache
fetch-plan/result references for diagnosing an unavailable indexed snapshot. No source is downloaded or selected automatically.

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

## Full OEIS corpus preparation

The 2026-11-11 scope amendment prepares every entry at a pinned revision for
future use. It does not assign all entries to training or waive the reserved
assessment/confirmation groups. The original bounded `fresh.bulk` workflow is
unchanged. This is a separate, resumable CPU data operation:

```bash
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/foundation_run.py \
  --engine podman --memory-gib 12 --cpus 4 \
  data fetch --config configs/foundation/oeis_corpus.json
```

The host needs only system Python and Podman. It builds/reuses the qualification
image through the root Dockerfile, grants network access only to the CPU corpus
worker and mounts the dedicated cache plus read-only config/stripped inputs.
There is no GPU/device/socket mount. No new host packages or DuckDB dependency
are required. The existing DuckDB is preserved; its inferred tags and missing
source offsets cannot substitute for authoritative metadata.

The first invocation pins the upstream OEIS commit in
`data/oeis-corpus-001/snapshot.json`. It inventories every sequence, including
short/finite/table entries. A bare repository retains original entries and
supporting-file pointers, without materializing a second b-file working copy.
All current b-files are fetched by one native queue per attempt:

```text
git -C <bare-repository> -c lfs.fetchrecentalways=false \
  -c lfs.concurrenttransfers=4 -c lfs.transfer.maxretries=8 \
  -c lfs.transfer.maxretrydelay=30 -c lfs.skipdownloaderrors=false \
  lfs fetch --include=files/**/b*.txt --exclude= origin <pinned-commit>
```

The argument list is passed directly, without shell wildcard expansion. There
are no Python batches or per-sequence HTTP calls. `--all` is deliberately absent
because it includes historical objects. Native transfer retries are supplemented
by up to three whole-command attempts, ten seconds apart, within one twelve-hour
invocation ceiling. Existing LFS objects survive retries; completed entries are
not reparsed when source fingerprints and published projection hashes match.
Short upstream size declarations are never trusted over content hashes; every
new b-file is hash-verified while its complete contents are parsed. Ordinary Git
b-files are also retained and parsed. Malformed data retains its disposition.

Defaults reserve at most 256 GiB of declared and actual source objects, 16 GiB for derived
projections/inventory, and 32 GiB of free disk headroom. These are limits, **not
measured requirements or a download-size estimate**. The full pointer census
records actual declared bytes before LFS transfer; disk/inode checks can pause
acquisition. If a limit is insufficient, edit the checked-in config locally to
raise the appropriate reservation and rerun the same command; retain the cache
and pinned revision. Each invocation has separate logs/cost records. Do not
change the selected revision in an existing cache; use a new cache for a new
snapshot. Rerunning the command resumes data work and does not start training.

The retained `data/oeis_cache/stripped.gz` is optional. Its hash and compatibility
with each pinned entry are recorded. Matching values are reused only over the
range verified against the pinned main entry; mismatches remain explicit, and
unverified extra cached values cannot manufacture a 100-term record. Names,
formulas and imported implementations never become target values/programs.
Raw full b-files stay in the native LFS store. `inventory.sqlite` retains every
entry's metadata, status, raw object location, availability counts and a bounded
120-term indexed projection, including short/finite/table entries. The public
per-member report excludes the actual terms. On complete preparation,
`data/oeis-corpus-001/indexed/manifest.json` exposes the eligible 100-term subset
through the existing indexed-source contract; eligibility is not membership.

Failures continue independent entry processing where resources permit. ZIP
collection runs on success, failure and handled interruption. Reinvocation
reconstructs diagnostics after a killed host and reclaims any leftover owned
container before starting new work. Share:

```text
data/oeis-corpus-001/share/corpus.zip
```

The ZIP includes exact commands, source/image identities, environment, per-member
outcomes, first errors/tracebacks, native command logs, container exit/OOM state
and cleanup records. Raw corpus, local SQLite data and weights stay local.
Any support ZIP size omission is explicit with local path/hash; it is never a
qualification artifact. Keep the cache and all original diagnostics.

### Recovery from the 2026-11-11 partial corpus

The uploaded report inventoried 401,255 entries, processed 401,250 and found
192,088 eligible under the 100-term rules. Its first native LFS fetch failed on
upstream network errors; the retry succeeded. Verification then rejected 1,345
objects solely because actual size exceeded the pointer declaration (1,341
auxiliary files and four primary b-files). A fifth primary entry, A363936,
contained conflicting main-entry/b-file values. All 41 ZIP file hashes verified.

Run the **same data-fetch command above after pulling this repair**, keeping
`data/oeis-corpus-001` and the original stripped cache. The CPU image is rebuilt
through the existing cached Dockerfile layers. The pinned Git revision, imported
cache and completed projections are reused. No new download is requested when
the selected LFS objects already exist. Completed entries still undergo cheap
projection-hash checks, but only blocked/changed entries are reparsed. Expect a
local scan, not a zero-work resume. Progress prints size-preflight, entry-scan
reuse counts and object-verification milestones.

Policy `sha256-content-bounded-actual-size/v2` accepts a size discrepancy **only**
after verifying the entire bounded object against its pinned hash. Both actual
and declared bytes remain in the object report. Aggregate actual bytes must fit
the configured reservation; same-hash size discrepancies are counted as warnings.
Hash mismatches still block completion and now include the observed hash and
size for diagnosis. Do not replace a pin or delete a cache to hide a mismatch.

A conflict between independently hash-verified source values is `EXCLUDED`,
with mismatch count/first source index, and cannot enter the indexed manifest.
It is not resolved by preferring the main entry or b-file. `COMPLETE` means all
entries are accounted for and all objects verified, including explicit ineligible
dispositions; it is not a claim that every sequence is usable for training. An
incomplete object-verification pass cannot publish a new complete manifest.

This is a production **CPU data-preparation** repair, so its source identity
changes. It does not invoke HIP/rehearsal or mutate the saved rehearsal artifacts;
those retain their original scope. Existing current-source qualification checks
still apply before later research execution. After recovery, share the same
`data/oeis-corpus-001/share/corpus.zip` path, including any remaining deficits.

### Reusing registration002 exposure data

Code changes require a new discovery session; they do not require rebuilding the
reviewed original data index. After corpus preparation completes, the registration
command can explicitly reference the completed original producer:

```bash
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman --session run-state/discovery/013-research-inputs-003 \
  register-inputs \
  --reference-campaign run-state/generalization/pac-012-bcd-r1-002 \
  --input-manifest specs/013-progressive-mathematical-discovery/validation/research-input-resolution.json \
  --oeis-snapshot data/oeis-corpus-001/indexed/manifest.json \
  --reuse-exposure-session run-state/discovery/013-research-inputs-002
```

This re-audits the original inputs and hashes the existing private index, checks
the reviewed producer/receipt/origin accounting, and references it in place.
It skips the 84-minute import and its image build, retaining original import and
preparation costs. Changed/partial/unreviewed producers are rejected. Keep the
original session: it owns the reused index. Current-source hardware qualification,
fresh panel/support construction, exact calibration and research freeze are still
required; this command does not complete those pending tasks.

## Independent OEIS memorisation diagnostics

Use the separately identified [OEIS observer](../../tools/oeis_diagnostics/README.md)
when the real OEIS training pool/cycle is ready. Freeze its training-group sample
before search, then analyse the original yield publication. It reports concrete
and parameterised lengths, literal information, direct-index exceptions and
terms 101–200 agreement without changing training/admission policies. It uses a
retained CPU image with no build/GPU/network/experiment-write access; these
observer-only changes leave the production qualification fingerprint unchanged.
The current corpus acquisition needs no restart and no exposure re-import.
The optional small CPU `self-test` and existing Docker CI exercise the real native
admission/archive regressions; they are not a repeated HIP rehearsal.


## Registered-data preparation and automatic research round

`prepare-research --registration-session run-state/discovery/013-research-inputs-003`
can now verify and reuse the completed registration in a **new preparation
session**. It retains the original corpus projection, exposure database and costs;
it does not fetch the corpus, repeat exposure import or request another tiny HIP
rehearsal. An optional `--rehearsal-session` records the actual original session
and scope differences, without treating old hardware evidence as current.

The installed CPU census v2 reserves 8,192 training draws and 32,768 evaluation
draws, with native/reference verification and exact original-exposure exclusions.
It reserves related groups together before quotas; missing strata/groups remain
explicit. Private support and panel candidates stay on the workstation. The
custodian projection publishes canonical cohorts, separate probes, finite support
shards and a fixed screening roster from those original reservations. The public
summary reports support supply at 25M charged tokens and the prospectively declared
eight-update-aligned cadence rule; a missing group or infeasible cadence remains
a deficit. It never
silently substitutes static data, reduces panel quotas or resets failed work.

Ceilings are one hour for CPU image build, six hours for native census including
finalization, and two hours for independently collected synthetic sensitivity.
These are resource ceilings, not an ETA. Case outcomes and original failures are
persisted, failed independent cases do not stop remaining cases, and incomplete
sensitivity scenarios have no estimated success/error rates. The whole run writes
`share/prepare-research.zip` on failure too. Completed stages are hash-verified and
reused; interrupted/failed stages are retained instead of receiving a fresh budget.

The registration route now constructs and validates the complete selected
capability/diversity/tuning round: 376 canonical roles, twelve learning-setting
trials, four sizing trials, three screening looks, and up to four fresh paired
confirmation positions. It obtains current-source qualification for twelve
model/dropout profiles, measures fourteen training-only evaluation cases, retains
all 1,296 exact calibration combinations, finalizes measured storage, freezes the
campaign and obtains campaign-bound acceptance. It does not repeat the tiny
rehearsal or expose final assessment.

From the repository root, use a new preparation session:

```bash
git pull --ff-only origin 013-progressive-mathematical-discovery &&
python3 -S scripts/discovery_hip_handoff.py \
  --engine podman \
  --session run-state/discovery/013-research-round-001 \
  prepare-research \
  --registration-session run-state/discovery/013-research-inputs-003 \
  --run-after-prepare
```

The command proceeds directly to canonical research run/resume, reporting and
independent verification only after preparation reports READY and the original
acceptance receipt is rechecked. Missing support, failed device cases or an
infeasible design stop dependent work with diagnostics. Reinvoking the same
command verifies and reuses completed stages; it does not grant a fresh budget
to failed/interrupted preparation. Keep the original inputs-002 and inputs-003
sessions because they own referenced artifacts. Both preparation and research
ZIPs remain under the new session's `share/` directory, including on failures.

Preparation is not expected to be a short 30-minute check. Evaluation timing has
a four-hour ceiling and exact calibration has a twelve-hour ceiling, in addition
to the native census, sensitivity, construction and device qualification. These
are finite bounds, not predicted durations. The outer reservation is published
as `prepare-research/roster/research-reservation.json` before device preparation
and rechecked against every literal task ceiling at freeze.

| Envelope | Selected maximum | Spec maximum |
|---|---:|---:|
| Training tasks, including all screening children and shared finalization | 176 h | 176 h for these stages |
| Mechanical preparation and host allowance | 23.95 h | 24 h |
| Images, qualification, freeze and acceptance | 8.09 h | 12 h |
| Evaluation/analysis, including training-only timing | 67.46 h | 72 h |
| Packaging and verification | 4 h | 4 h |
| New acquisition | 0 h | 24 h |

The selected sum is approximately 279.5 hours, not an ETA or a requirement to
consume it. Prior registration/import costs remain separately recorded in the
original registered-data receipt and journals. Screening evaluations and the
thirty-minute shared reserve are included once inside each two-hour trial.
The complete optional 672-hour envelope is not allocated to this subset.

The aggregate review corrected an over-allocation in the initial operational
ceilings. Non-screening generation now has a prospective seventy-minute task
ceiling, including at least thirty minutes of finalization and its startup;
language audit has one hour. No panel member, trial, look, model choice, attempt,
per-sequence portfolio cap, fitting cap or token target was removed. Training-only
measurements must demonstrate feasibility with the original safety margin before
any research panel is scored. An infeasible result requests a new prospective
design; the wrapper never silently reduces the experiment.

Scaling, OEIS, components and the source pilot retain their separate conditional
eligibility and finite-freeze requirements. This first round does not reserve
100M/400M continuations, claim sustained HIP evidence or automatically open final
assessment. Its results determine which later branches can be properly frozen.

Exact-input calibration plans may prospectively select
`checkpoint_retention: verified-calibration-inventory/v1`. That policy retains
two live temporary checkpoint generations for the current serial case instead
of retaining every case's tensors. Only an original passing, verified calibration
can retire its temporary snapshots. Complete file hashes and byte counts are
published before cleanup, with a separate completion receipt; original producers,
input packages and costs remain intact. Failed cases and scientific training
checkpoints are not eligible. Legacy plans retain all calibration snapshots.
Storage reservations include every retained input and per-case inventory.

Calibration publishes immutable per-case start/terminal records and a final
roster. Bad input hashes or malformed receipts cannot make the batch pass;
independent cases continue, while a failed GPU health probe or expired deadline
leaves explicit blocked cases. The support ZIP includes these records and both
retirement inventories, without exporting tensors. CPU and simulated-boundary
tests do not establish actual HIP qualification or resource feasibility.
