# Temporary VM GPU progression runner

Run from the checkout **inside the VM built from the root Dockerfile**. CPU
commands run locally; GPU stages use `lab-gpu exec -- python …`. This disposable
controller does not install packages at runtime or change task checkboxes.

**Docker access is not needed in the VM.** The current lab broker supports
standalone GPU diagnostics. Its protocol cannot launch the isolated foundation
roles needed for the baseline, measurement and complete end-to-end run. Those
stages now report `BLOCKED` before issuing work or spending a lifetime budget.

## How the VM reaches the GPU

Reviewed against [grugnog/lab at ebb54858](https://github.com/grugnog/lab/tree/ebb54858b1ed2f80dd6fdb4b752d96cfdd9852de),
including its [broker implementation](https://github.com/grugnog/lab/blob/ebb54858b1ed2f80dd6fdb4b752d96cfdd9852de/packages/lab-gpu/lab-gpu.py)
and [setup/recovery guide](https://github.com/grugnog/lab/blob/ebb54858b1ed2f80dd6fdb4b752d96cfdd9852de/docs/lab/gpu-runner.md).

| Location | Responsibility |
| --- | --- |
| Host [start.sh](../start.sh) | Copies the lab client from `/etc/nixos/packages/lab-gpu`, builds the `smolvm` Dockerfile target with Podman, and mounts the checkout at `/workspace` in SmolVM. Run this script on the host. |
| VM [Smolfile](../Smolfile) | Sets `gpu=false`, the restricted host address/account and `/home/agent/.ssh/lab-gpu` identity. Python and OpenSSH run the adjacent `lab-gpu`/`lab-gpu.py` client files. |
| Host SSH broker | Accepts only `ensure` and `exec` requests carrying argv and a relative working directory. Forced-command SSH provides no host shell, forwarding, runtime image override or arbitrary environment overrides. |
| Host rootless Podman worker | Snapshots `gpu-project.json` inputs, builds/caches the `gpu` target and returns its immutable content ID. Runtime mounts the entire project workspace and private cache, passes GPU devices, and supplies fixed confinement. |
| Host supervisor | Owns the GPU lock, stops/restores inference, enforces the host timeout and removes its container on exit/disconnect. Stale host leases require administrator recovery. |

No container-management socket or SSH identity enters the GPU image. Installing
a Docker CLI there would not grant the missing host API. The project key cannot
be used as a `docker -H ssh://…` endpoint. The existing OEIS role launcher needs
separate input/output mounts, CPU workers without GPU devices, and private
generation/evaluation views. The broker's whole-workspace container cannot
establish those guarantees. Integrating an accepted host-side role launcher is
separate work; the validation script does not weaken the role gates.

The lab project's checked-in policy uses the module's **3600-second exec
timeout**. The host enforces this independently of script deadlines. A two-hour
learning allowance or an eight-hour engineering/pilot allowance does not grant
that time through the broker. A host policy change and appropriate launcher
integration are prerequisites for longer runs. Timeout/transport errors remain
failures; they are not evidence of a passing hardware gate.

```sh
python scripts/gpu_progress.py --plan
# No settings or Docker access needed for these stages:
python scripts/gpu_progress.py --until gpu-environment
python scripts/gpu_progress.py --until gpu-smoke
python scripts/gpu_progress.py --until preflight
# To also inspect source, legacy and currently unsupported role stages:
mkdir -p tmp-gpu-inputs
cp scripts/gpu_progress.example.json tmp-gpu-inputs/settings.json
# Fill the required settings below, then:
python scripts/gpu_progress.py --config tmp-gpu-inputs/settings.json
```

Start with `--until gpu-environment` for image/HIP checks, `--until gpu-smoke`
for short optimizer/checkpoint checks, or `--until
gpu-learning` for all 15 accelerator cases. Rerun the identical command to
continue. Successfully verified stages are skipped. There is no force option,
gate override or CPU fallback.

## Required inputs

Paths in settings are relative to the repository root or absolute shared paths.
Keep local settings and data in top-level `tmp-*` directories. Missing inputs
block their consumers; independent checks continue.

| Setting | Required evidence |
| --- | --- |
| `image_id` | Usually null: automatically resolved from `lab-gpu image ensure` stdout, normalized to `sha256:…`, and recorded in the image receipt. An optional immutable pin must match the broker's actual result. A tag or registry manifest digest is insufficient. |
| `preflight_config` | Usually null. The runner generates a configuration from the pinned root Dockerfile and exact `uv.lock`. Supply an existing real configuration if the accepted build needs it. |
| `source_manifest` | Authoritative `indexed-source/v1` manifest, with hashed b-files and offset/keyword/finite/table metadata. Stripped sample data cannot qualify a release. |
| `cohort_config` | Frozen `prefix20_total100_affine_v2` source configuration, with explicit seed/development/final counts. |
| `archive` | Existing strict admitted archive, partitioned against every member's reserved release prefixes. Engineering additionally requires 128 distinct generic training prompts and a nonempty structural development complement. |
| `holdouts` | Exact expanded holdout profile for that archive, accepted by `load_holdouts`. |
| `legacy_cases` | JSON array of actual trained parent checkpoint/configuration pairs, described below. |
| `final_seconds` | Reserved final lifecycle deadline: default 3600, allowed integer 600–43200. The final stage remains blocked until host role integration exists; it does not issue a new budget. |

`docker_host` is no longer supported. Existing files with `docker_host: null`
migrate automatically; a non-null endpoint fails with the broker capability
explanation. The runner does not provision host drivers or host access.

Example historical cases, with paths relative to the JSON file:

```json
[
  {
    "checkpoint": "../../runs/my-historical-run/checkpoints/model_epoch_050.v2.pt",
    "config": "../../runs/my-historical-run/config.yaml"
  }
]
```

Default checks load actual v2 parent bytes and configuration, require a trained
epoch, compare CPU/GPU logits, execute a finite shared SFT update, check
weights-only reload parity, perform owning vocabulary surgery and model-based
evaluation, and preserve `legacy_unverified`. The GPU stage also executes the
owning warmstart gate. These are bounded shared compatibility paths. Sustained
historical launcher acceptance remains pending. v2 artifacts do not restore
optimizer/RNG state; reload parity is not labelled full resume. Unsupported
formats fail rather than being silently converted.

A case can additionally supply `tests`, mapping all four keys `train`, `resume`,
`warmstart`, `evaluate` to nonempty lists of owning pytest node IDs. Each receives
`OEIS_LEGACY_PARENT`, `OEIS_LEGACY_CONFIG`, `OEIS_LEGACY_OUTPUT` and
`OEIS_LEGACY_DEVICE`. Those suites must execute actual bounded launcher behavior;
mock tests do not establish sustained T117 acceptance. Skips and errors reject.
The default runner does not promote sustained historical acceptance.

## Ordered checks and earlier partial work

The first inventory maps **every earlier unfinished task**, including T047,
T113 and T114. An unmapped new partial task is an error. Early CPU regression
checks cover all foundation modules; explicitly deferred Docker, device and
actual-artifact cases have later owning stages. The Docker cases remain blocked
under this broker. Unknown skips fail. Image/HIP checks run before those
regressions; release/legacy inputs are not needed through `--until preflight`.

| Stage | Scope |
| --- | --- |
| `inventory` | Partial-task map, packages, Linux boot identity, disk, native runtime, `/workspace` checkout, and lab client/SSH/key availability. |
| `image`, `gpu-environment` | Broker image ensure/content ID, actual HIP/native identity and matrix multiplication, broker/VM clock agreement. |
| `cpu-regressions` | Earlier configuration/cache/mask/admission/resume/resource/wrapper/source/decision regressions and the unavailable-GPU negative. |
| `gpu-smoke` | Three actual HIP optimizer updates, checkpoint/prefix-only generation and effective constructor/backward behavior. |
| `gpu-learning` | Owning G8 software and all 12 FP32 full/prefill and three FP16/GradScaler GPU cases. Skips reject. |
| `preflight` | Standalone three-update 20-term/1024-token feasibility probe bound to the broker image; establishes GPU feasibility, not private role isolation. |
| `legacy-cpu`, `legacy-gpu` | Actual historical artifacts and bounded train/reload/warmstart/evaluation paths. |
| `phase13-inputs` | First 256 generic draws and rejections, available archive strata, independent 4096-bit traces, checked/unchecked CPU timings, available checkout artifacts. |
| `release-source` | T113 exact horizon/metadata/group audit and strict structurally partitioned nonempty archive. |
| `host-containment`, `baseline` | **Blocked:** owning Docker resource/isolation tests and privately mounted admitted-pool training/checkpoint bundle require host launcher integration. |
| `overflow-gpu` | **Blocked:** actual greedy model draws and candidate validation need the baseline and separately isolated CPU workers. The owning verified stratum is `verified_solutions`. |
| `measurement` | **Blocked:** fixed T119/conditional T120 schedule needs private roles and sufficient host time allowance. Its two/six-hour caps remain unchanged. |
| `release-handoff` | **Blocked:** requires the actual measurement, installed qualification and bounded immutable handoff replay. |
| `end-to-end` | **Blocked:** complete private training, pause/resume, development/final evaluation and planning report require the same missing host role integration. |

Engineering and conditional pilot form one **non-reissuable** stage because the
owning service has one immutable issuance schedule. Their separate two/six-hour
caps and decision rules remain required. Baseline-retained, not-triggered or
inconclusive decisions may be legitimate. Interrupted work cannot silently
receive a fresh allowance.

## Final lifecycle and planning evidence

The preserved, currently blocked final implementation uses strict FP32
batch-four **three-update** training: pause after
update one, resume through update three. Its independent training ledger retains
its ten-minute cap. The outer deadline includes preparation, training,
checkpointing, evaluation and cleanup. It defaults to one hour and cannot exceed
twelve hours. Offline/interrupted time remains conservatively charged on rerun;
clock disagreement blocks the run.

Before training it freezes at most two original development groups and two
original final groups, preserving all affine-group members and original
partitions. This is a diagnostic projection. Generation uses two attempts and a
128-token cap. Zero solved groups is a measured outcome. Final truth does not
alter the stopping decision.

After launcher integration, `end-to-end/planning-report.json` is intended to
retain loss, gradient, token, phase, resource
and sensor events; parameter count; checkpoint size; pause/resume identity;
training and lifetime budgets; development/final scores; measured phase throughput.
Use T119 ten-minute measurements to plan steady-state scaling. Three updates
measure lifecycle cost; they cannot establish convergence.

## Logs, reruns and recovery

Default output is `tmp-gpu-progress/`. Custom state directories must be top-level
`tmp-*` directories in the checkout. Each step preserves `transport.log`, GPU
`worker.log`, actual reports/JUnit and `step-result.json` for issued stages.
Root files include `receipts.json` and `summary.json`; newly blocked stages retain
the capability explanation with `issued=false`. Earlier run journals and
`final-budget.json`, if present, remain preserved. Running commands print a
heartbeat every 30 seconds.

The GPU smoke training run is retained under `gpu-smoke/training-run/`, including
its configuration, training report, event/budget journals and checkpoints. The
broker mounts `/tmp` as a 2 GiB tmpfs, which cannot satisfy this run's frozen
100 GiB free-disk floor. The owning HIP test therefore receives a workspace run
path from the progression worker. Ordinary pytest fixtures still use their
normal temporary paths; retaining the whole pytest tree would also retain
temporary aliases and unrelated test inputs. The run's disk/memory limits,
model and three-update training contract stay unchanged. Disk guard failures
now distinguish run quota from filesystem free-space failures and include the
actual run path, measured bytes and required capacity.

Receipts bind source, settings, recursively referenced input bytes, dependency
evidence and every retained output. Changed/tampered bytes invalidate success.
Inventory, image and cheap GPU probes repeat; stable runtime identities preserve
expensive successful results. Ordinary failed attempts move to `history/`.
Remote locks prevent duplicate work and moving a live output directory. A
successful receipt survives temporary dependency failures.

The VM user and the host rootless project account can have different
UIDs. Lock files are stable, readable coordination inodes: both sides use
read-only handles for Linux `flock`, so a VM-owned `0644` lock does not require
GPU write permission. Settings and atomic report replacements are readable
`0644` files; each writer uses a new temporary rather than truncating the other
identity's file. Existing lock files are never replaced to bypass contention.

Logs receive explicit group/other read permission before a subprocess starts,
so live GPU logs can be read from the VM even if the host later kills the
container. After each worker exits, its own retained stage files receive read
permission and its nested output directories receive read/traverse permission.
This includes third-party JUnit reports and checkpoints, and partial outputs
after failures or watchdog timeouts. A permissive umask alone cannot override
an inherited default ACL with `other::---`. The publication step does not grant
write access, change ownership, or change source/input files outside the stage.
It leaves foreign controller-owned files alone and rejects symbolic links,
multiply linked files and special files rather than touching external inodes.
Permission publication failures block acceptance; unreadable directories cannot
be silently omitted from receipt hashes.

If an older GPU attempt finished but the VM reported `Permission denied` while
hashing `worker.log`, inspect its readable `step-result.json`: worker success
and controller receipt acceptance are separate. Pull this fix and rerun using
the same state directory. Existing failed attempts are archived, and new
outputs are published before acceptance. To inspect an old log immediately,
run this scoped repair on the **host**, with the checkout path filled below:

```sh
OEIS_HOST_WORKSPACE=/absolute/path/to/host/checkout
sudo chmod a+r "$OEIS_HOST_WORKSPACE/tmp-gpu-progress/gpu-environment/worker.log"
```

This repairs the retained log only. Do not make the workspace world-writable,
replace live locks, or remove receipts/budgets. If the host broker kills the
entire container before either worker cleanup or watchdog cleanup runs, live
logs remain readable; unpublished partial artifacts may still need a scoped
host repair for inspection. Such a terminated stage cannot receive a PASS.

The GPU worker first probes settings readability and output-directory creation
under its actual host identity. The host project workspace/default ACL setup
in the linked lab guide must allow both identities to traverse/read inputs and
create/rename outputs, including VM-created directories. A failed probe reports
the exact path, directory mode/owner and worker identity as `BLOCKED` before
the test worker starts.

VM-created `0750` directories can mask an inherited named-user `rwx` ACL down
to `r-x`. The controller now restores group-class/ACL-mask `rwx` on the state
root, history directory and each newly prepared stage root, including after
retry recreation. Owner, other-user permissions, named ACL entries and default
ACLs are preserved. It does not change workspace ownership or source/input
permissions. Both accounts still need actual host ACL entries; changing the
mask alone cannot grant a missing named account access.

Stop the runner and apply **access and default** ACLs to its existing run
directories from the host. Fill these values from the confirmed host policy:

```sh
OEIS_HOST_WORKSPACE=/absolute/path/to/host/checkout
OEIS_HOST_OWNER=your-host-user
OEIS_GPU_ACCOUNT=registered-project-account
sudo find "$OEIS_HOST_WORKSPACE/tmp-gpu-progress" -type d -exec \
  setfacl -m "u:$OEIS_GPU_ACCOUNT:rwx,u:$OEIS_HOST_OWNER:rwx,m::rwx,d:u:$OEIS_GPU_ACCOUNT:rwx,d:u:$OEIS_HOST_OWNER:rwx,d:m::rwx" {} +
getfacl -p "$OEIS_HOST_WORKSPACE/tmp-gpu-progress/gpu-environment"
```

The access entry must show effective `rwx`, with `mask::rwx`; the default entry
and default mask must also be `rwx`. `d:u:…` changes inheritance only. A chmod
on one stage directory is temporary because ordinary reruns archive and
recreate that directory. After pulling this fix, rerun the same command in the
VM with the same state directory; preserve receipts/history and all budgets:

```sh
python scripts/gpu_progress.py --until gpu-environment
```

Exit 75 is broker busy; 124 is deadline expiry. A watchdog inside the broker
container enforces its own worker deadline if the VM transport disconnects.
The host broker owns container cleanup and inference restoration; project code
does not invoke Docker or recursively call lab-gpu from inside that container.
A stale host lease requires the administrator procedure in the linked lab
guide. `--reclaim` now explains that requirement and makes no remote request.
An older failed Docker reclamation report still blocks retries and needs review
at its originally recorded endpoint. Do not
remove/edit receipts, issuance journals or final budget to obtain a fresh
allowance. A non-reissuable measurement failure requires review of its actual
disposition; using another state directory is not automatic retry authorization.

Aggregate exit is nonzero if any selected stage failed or blocked. Successful
orchestration does not assert full spec-007 completion. Required source,
historical weights, sustained launcher behavior, native capability and hardware
evidence must come from actual owning runs. Tasks are not automatically checked.
