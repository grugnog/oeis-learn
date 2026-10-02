# Foundation training and continuation

The phase 6 implementation improves the existing package. It uses the production
tri-stream encoder, WAT Transformer decoder, and a shared teacher-forcing
operation from `rl/sft_trainer.py`. It does not call that legacy trainer's
demonstration generator or accept its weights-only checkpoints.

The supported run is the frozen three-update SFT smoke. Every run currently
requires `--diagnostic`: a completed smoke is not workstation readiness or an
authorization for long training. Coordinated container isolation and the
provenance-backed qualification path remain incomplete.

## CPU diagnostic

First build a verified pool using the phase 5 `foundation build-pool` command.
Use its output directory below. Pool verification reads the full admission
archive in the controller; the learner receives a copied, pinned view containing
only program IDs, twenty visible terms and complete body tokens.

```bash
python -m oeis_learn.cli.main foundation train \
  --config configs/foundation/wat_smoke.yaml --pool /path/to/verified-pool \
  --run-dir /path/to/new-run --device cpu --diagnostic --json

python -m oeis_learn.cli.main foundation inspect --run-dir /path/to/new-run --json
```

The default resource floors remain 100 GiB free disk and 16 GiB available host
memory. On a small CPU test host, `--diagnostic-small-host` explicitly freezes
lower floors of 1 GiB free disk and 256 MiB available host memory. This deviation
is recorded in `run.json`, inspection and the training report and cannot qualify.
It does not change the model, arithmetic, batch, objective or checkpoint format.

Use `--stop-after-update 1` for a clean interrupted smoke. Resume takes no model,
optimizer, device, pool, seed or resource overrides:

```bash
python -m oeis_learn.cli.main foundation resume --run-dir /path/to/new-run --json
```

The optional `--checkpoint` is an assertion about the newest valid manifest, not
permission to force an older state. JSON output supplies the actual checkpoint
manifest path. `--prepare-only` validates and freezes inputs without starting the
training arm. A finalized run cannot resume training.

Continuation pins the complete package source/resource inventory, software and
device identity, and the full run settings including resource floors. Changes
require a new run; matching tensor shapes alone cannot establish compatible
forward computation. FP32 matmul precision is explicitly `highest`, with TF32
disabled and deterministic algorithms enabled.

## What is restored and charged

At each completed update, gradients are cleared and the writer saves model,
AdamW, explicit null scheduler/scaler, Python/NumPy/CPU/all-active-GPU RNGs, named
initialization/data-order/sampling RNGs, epoch/permutation/cursor/next IDs and all
manifest counters. BOS is input only; all nonpadding target tokens including EOS
contribute to each program's mean cross-entropy, followed by the batch mean.
The smoke uses FP32, AdamW 3e-4/0.01, norm cap 1, batch 4 and three updates.

Checkpoint blobs are flushed, fsynced and hashed before immutable publication.
The external manifest identifies final bytes; the latest pointer is only a hint.
Recovery scans newest-to-oldest, quarantines incomplete/corrupt generations,
preserves generation numbers across rollback, and retains two valid generations
plus a separately pinned final. Checkpoints including quarantine have a 40 GiB
cap, with twice the largest estimated/existing blob reserved before saving.
Quarantine is retained under quota rather than silently deleted.

A separate controller durably writes start/phase/one-second heartbeat events
while the learner runs. A second process enforces the controller deadline. Clean
pauses exclude downtime. An unclean gap is charged through recovery and labeled
estimated; same-boot time uses monotonic intervals, cross-boot recovery uses UTC,
and backward/inconsistent clocks fail closed. Model rollback never rewinds this
ledger. Training, recovery and checkpoint costs are integrated. The ledger API
also supports bootstrap, validation and evaluation phases; the existing separate
pool/evaluation commands are **not yet one integrated arm controller**. Their
own bounded timing reports must not be represented as a complete multi-phase
training ledger.

Events are append-only bounded segments (64 MiB each; 1 GiB for budget events
and 1 GiB for metrics), with duplicate/conflicting event checks and torn-tail
quarantine. Retried update attempts remain visible; the selected checkpoint's
`completed_update`, not a count of update events, is committed learning progress.
Typed metrics preserve measured zero and use null for unavailable/disabled values.
RSS, available host memory, and device allocator statistics overlap on an APU;
do not add them into a purported total allocation.

## Ryzen checks still required

Use the existing root Dockerfile and `lab-gpu` workflow. No second Dockerfile,
host ROCm installation or driver change is introduced. On the configured machine:

```bash
lab-gpu exec -- python -m pytest -q tests/integration/test_foundation_gpu.py -ra
```

The GPU test requires actual HIP, uses the full configured backbone and a real
admitted generic pool, and checks three finite parameter-changing updates,
encoder/decoder residency and allocator evidence. CPU equality is tested within
one pinned software/device environment; no cross-device or cross-version bitwise
GPU continuation guarantee is made. Unsupported deterministic HIP operations
must remain visible failures, not trigger a silent precision/device fallback.

`scripts/foundation/run_container.sh` launches the existing local image by its
immutable `sha256:...` image ID. It applies two explicit, nonoverlapping role
mounts, read-only input/root filesystem, no network or Docker socket, equal
memory/swap caps, CPU/PID caps, and GPU device exposure only for learner/preflight.
Workers get 1 GiB/one CPU/64 PIDs; other roles get 88 GiB/four CPUs/512 PIDs.
It uses host-side `docker rm --force` at the deadline to reclaim the cgroup.
Do not launch multiple 88 GiB roles concurrently or infer an aggregate cap from
this single-container launcher.

This launcher is scaffolding for the existing environment, **not** a completed
multi-role execution protocol. The current evaluator's generation/worker process
transport has not been moved to coordinated role containers; aggregate admission
of up to eight worker containers and mounted-view enforcement need integration
and actual Docker tests. The shared development filesystem is not isolation
evidence. The repository does not contain the machine's `system/lab-gpu` broker
implementation, so its actual mount/device configuration was not audited here.

The Docker-marked test requires Docker and `OEIS_FOUNDATION_TEST_IMAGE` set to the
resolved local image ID; it actually checks visible/read-only mounts and absent
truth/device/socket paths. Hardware skips are pending evidence, not passes.

`foundation inspect` keeps lifecycle separate from qualification. Readiness
remains fail-closed until adopted-contract, exact-runtime, provenance, isolation,
resume, real-device/containment and proof-boundary evidence are all bound to the
run. Full spec 007 also needs G7–G10; a three-update smoke cannot satisfy them.
