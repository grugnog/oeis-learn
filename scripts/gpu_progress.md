# Temporary VM GPU progression runner

Run from the checkout **inside the VM built from the root Dockerfile**. CPU
commands run locally; GPU stages use `lab-gpu exec -- python …`. This disposable
controller does not install packages at runtime or change task checkboxes.

```sh
python scripts/gpu_progress.py --plan
mkdir -p tmp-gpu-inputs
cp scripts/gpu_progress.example.json tmp-gpu-inputs/settings.json
# Fill the required settings below, then:
python scripts/gpu_progress.py --config tmp-gpu-inputs/settings.json
```

Start with `--until gpu-smoke` for the short hardware checks, or `--until
gpu-learning` for all 15 accelerator cases. Rerun the identical command to
continue. Successfully verified stages are skipped. There is no force option,
gate override or CPU fallback.

## Required inputs

Paths in settings are relative to the repository root or absolute shared paths.
Keep local settings and data in top-level `tmp-*` directories. Missing inputs
block their consumers; independent checks continue.

| Setting | Required evidence |
| --- | --- |
| `image_id` | Actual resolved `sha256:…` Docker image ID, inspected on the accepted GPU host after `lab-gpu image ensure`. A tag or registry manifest digest is insufficient. |
| `docker_host` | Optional accepted `ssh://user@host` Docker controller endpoint. The GPU broker container must already have working noninteractive credentials. Null uses the existing Docker endpoint. |
| `preflight_config` | Usually null. The runner generates a configuration from the pinned root Dockerfile and exact `uv.lock`. Supply an existing real configuration if the accepted build needs it. |
| `source_manifest` | Authoritative `indexed-source/v1` manifest, with hashed b-files and offset/keyword/finite/table metadata. Stripped sample data cannot qualify a release. |
| `cohort_config` | Frozen `prefix20_total100_affine_v2` source configuration, with explicit seed/development/final counts. |
| `archive` | Existing strict admitted archive, partitioned against every member's reserved release prefixes. Engineering additionally requires 128 distinct generic training prompts and a nonempty structural development complement. |
| `holdouts` | Exact expanded holdout profile for that archive, accepted by `load_holdouts`. |
| `legacy_cases` | JSON array of actual trained parent checkpoint/configuration pairs, described below. |
| `final_seconds` | Final lifecycle deadline: default 3600, allowed integer 600–43200. Cannot change after issuance. |

The GPU image contains a Docker CLI and SSH client. The accepted controller must
have the pinned image locally and see **identical absolute shared paths**.
Read-only mount/device/quota/kill tests run before the baseline and long work.
Workload roles receive no Docker socket, repository, or private cohort mount.
The runner does not provision host drivers or create privileged containers.
Configure accepted access; it does not guess credentials or endpoints.

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
actual-artifact cases have later owning stages. Unknown skips fail.

| Stage | Scope |
| --- | --- |
| `inventory` | Partial-task map, packages, Linux boot identity, disk, native runtime and broker availability. |
| `cpu-regressions` | Earlier configuration/cache/mask/admission/resume/resource/wrapper/source/decision regressions and the unavailable-GPU negative. |
| `phase13-inputs` | First 256 generic draws and rejections, available archive strata, independent 4096-bit traces, checked/unchecked CPU timings, available checkout artifacts. |
| `release-source` | T113 exact horizon/metadata/group audit and strict structurally partitioned nonempty archive. |
| `image`, `gpu-environment` | Image rebuild/ensure, actual HIP/native identity and matrix multiplication, broker/VM clock agreement. |
| `gpu-smoke` | Three actual HIP optimizer updates, checkpoint/prefix-only generation and effective constructor/backward behavior. |
| `gpu-learning` | Owning G8 software and all 12 FP32 full/prefill and three FP16/GradScaler GPU cases. Skips reject. |
| `host-containment`, `preflight`, `baseline` | Real Docker resource/isolation tests; three-update 20-term/1024-token preflight; G7/native, admitted pool smoke and checkpoint-bound G0–G6 bundle. |
| `legacy-cpu`, `legacy-gpu` | Actual historical artifacts and bounded train/reload/warmstart/evaluation paths. |
| `overflow-gpu` | At most 32 actual greedy draws from the GPU-trained baseline on training-only prefixes; 1024-token context. Retain failures/shortfalls, replay provenance, reuse CPU traces, trace/time only outstanding model/verified records, seal 20+80 development scoring and record synchronized sensor/allocator evidence. Zero valid samples blocks acceptance. |
| `measurement` | Fixed T119 then conditional T120 owning schedule: 128 prompts, seeds 7001–7003, 10 warmups/30 repeats, 4/8 workers, ten-minute steady measurements, two-hour engineering cap, at most one six-hour pilot. No extra sweep. |
| `release-handoff` | Native G9, separate software G8 for installed negative tests, clean wheel/source-distribution rebuild, current dispositions and G0–G10 replay. Publish/replay bounded immutable handoff; pending readiness stays pending. |
| `end-to-end` | Last: fresh isolated GPU training, pause/resume, development evaluation, immutable stopping/finalization, sealed final evaluation and planning report. |

Engineering and conditional pilot form one **non-reissuable** stage because the
owning service has one immutable issuance schedule. Their separate two/six-hour
caps and decision rules remain required. Baseline-retained, not-triggered or
inconclusive decisions may be legitimate. Interrupted work cannot silently
receive a fresh allowance.

## Final lifecycle and planning evidence

The final run uses strict FP32 batch-four **three-update** training: pause after
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

`end-to-end/planning-report.json` retains loss, gradient, token, phase, resource
and sensor events; parameter count; checkpoint size; pause/resume identity;
training and lifetime budgets; development/final scores; measured phase throughput.
Use T119 ten-minute measurements to plan steady-state scaling. Three updates
measure lifecycle cost; they cannot establish convergence.

## Logs, reruns and recovery

Default output is `tmp-gpu-progress/`. Custom state directories must be top-level
`tmp-*` directories in the checkout. Each step preserves `transport.log`, GPU
`worker.log`, actual reports/JUnit and `step-result.json`. Role stdout/stderr are
retained. Root files include `receipts.json`, `summary.json` and the independent
`final-budget.json`. Running commands print a heartbeat every 30 seconds.

Receipts bind source, settings, recursively referenced input bytes, dependency
evidence and every retained output. Changed/tampered bytes invalidate success.
Inventory, image and cheap GPU probes repeat; stable runtime identities preserve
expensive successful results. Ordinary failed attempts move to `history/`.
Remote locks prevent duplicate work and moving a live output directory. A
successful receipt survives temporary dependency failures.

Exit 75 is broker busy; 124 is deadline expiry. A watchdog inside the broker
container enforces its own deadline if the VM transport disconnects. Docker
roles receive a unique host-only lease label; cleanup targets only that lease.
Failed cleanup blocks retries. Retry reclamation at the saved endpoint with:

```sh
python scripts/gpu_progress.py --state-dir tmp-gpu-progress --reclaim baseline
```

Then rerun the original command. Reclamation does not reset budgets. Do not
remove/edit receipts, issuance journals or final budget to obtain a fresh
allowance. A non-reissuable measurement failure requires review of its actual
disposition; using another state directory is not automatic retry authorization.

Aggregate exit is nonzero if any selected stage failed or blocked. Successful
orchestration does not assert full spec-007 completion. Required source,
historical weights, sustained launcher behavior, native capability and hardware
evidence must come from actual owning runs. Tasks are not automatically checked.
