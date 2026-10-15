# Portable fresh training on ROCm

Run `scripts/foundation_run.py` from a Linux host with Python 3.11+ and an already
usable Docker or Podman connection. Python on the host needs only its standard
library. Model dependencies, PyTorch/ROCm and the Rust evaluator are built into
the public `Dockerfile` GPU target. `foundation_local.py` is a compatibility alias.
The wrapper uses your current engine connection and never changes host services,
accounts or permissions. Rootless engines need working device access and delegated
cgroup memory/CPU controls. The `check` command verifies these inside the roles.

Choose an engine explicitly (replace `podman` with `docker` throughout), or omit
`--engine` to select the first usable connection. Engine options precede the
subcommand. All bind paths must be visible to the engine on the same Linux host;
a remote engine with a different filesystem is not supported. Run the wrapper
on the host, rather than inside a training container.

## Build and validate

```bash
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate --suite cpu --output tmp-validation/cpu
python3 scripts/foundation_run.py --engine podman validate --suite gpu --output tmp-validation/gpu
```

CPU validation runs the foundation, fresh-workflow and runtime unit/contract/
integration regressions plus native Rust/reference parity. Historical experiment
tests remain available through ordinary pytest, separately from this workflow.
GPU validation requires actual HIP matrix operations, full-model updates and exact
checkpoint continuation, checkpoint-to-generation device restoration, deterministic
loss, G8 learning/cache/FP16 acceptance, then private-role containment probes.
It fails if a selected GPU regression skips. Every stage writes stdout/stderr,
JUnit where applicable, and a receipt bound to the image, code, runtime and output
hashes. Repeats skip verified successful stages; changed inputs or logs invalidate
them. GPU environment checks always rerun and bind subsequent receipts to the
observed device and runtime. PASS receipts are written only after container reclamation. A passing smoke
or validation suite is diagnostic evidence, not release qualification.

To reuse a prebuilt image, pass `--image IMAGE` before the subcommand. It is
inspected to an immutable ID. `--gpu-base-image REF` overrides the public pinned
ROCm base for a new build; qualify that runtime again. The default build target
is `gpu`; `docker build --target qualification -t oeis-learn:cpu .` builds the
separate CPU/native CI image. Supply it with `--image oeis-learn:cpu` only for CPU
operations and `validate --suite cpu`.

With multiple GPUs, select the matching render device using
`--render-node /dev/dri/renderD128`. Only `/dev/kfd` and that render node enter
GPU roles. CPU roles and execution workers receive no GPU devices. GPU paths
never fall back to CPU. The ROCm base must support your GPU and host driver.

## Reuse existing prepared data

Your sealed `tmp-fresh-inputs-extended-002` and bulk downloads are reusable. The
launcher change does not require admitting the archive again. Check its bytes
and start with a new smoke run:

```bash
python3 scripts/foundation_run.py --engine podman check --inputs tmp-fresh-inputs-extended-002
python3 scripts/foundation_run.py --engine podman smoke \
  --inputs tmp-fresh-inputs-extended-002 --run runs/portable-smoke-001
```

If an earlier admission stopped at 2,008/2,048 training prompts, explicitly seal
that retained journal only if accepting the shortfall is your intended choice:

```bash
python3 scripts/foundation_run.py --engine podman seal-inputs \
  --inputs tmp-fresh-inputs-extended-002 --accept-shortfall
```

The command records the actual counts and original requested counts; it neither
relabels a shortfall as full admission nor reissues the spent preparation budget.
Already sealed inputs only need `check`. Historical pre-007 checkpoints remain
historical and are not required to train fresh models.

## New data and preparation

```bash
python3 scripts/foundation_run.py --engine podman data fetch --config configs/foundation/fresh_pilot.yaml
python3 scripts/foundation_run.py --engine podman prepare \
  --config configs/foundation/fresh_pilot.yaml --output tmp-fresh-inputs
python3 scripts/foundation_run.py --engine podman check --inputs tmp-fresh-inputs
```

Acquisition uses the bulk `oeis/oeisdata` Git/LFS repository and freezes its exact
revision. It does not download sequences individually. `data.cache`, disk/download
limits, seed and record limits live in the configuration. A local bulk snapshot
can also be supplied through the data configuration. Inspect `foundation_tasks.py`
help for the image's CPU task interface; ordinary operations use the host wrapper.
Only bulk acquisition receives network access. Preparation resumes its existing
journal and lifetime budget; preserve its configuration and output directory.
An exhausted admission needs explicit shortfall sealing or a new planned dataset.

## Benchmark, then train

Use the existing inputs first; changing container transport alone is not a reason
to rebuild them. Measure batch sizes and length bucketing on the selected GPU:

```bash
python3 scripts/foundation_run.py --engine podman benchmark \
  --inputs tmp-fresh-inputs-extended-002 --run runs/portable-benchmark-001
python3 scripts/foundation_run.py extended-settings \
  --benchmark runs/portable-benchmark-001 --output tmp-analysis/portable-extended.json
```

`extended-settings` derives a new preparation config from the measured winner;
use a new inputs directory if adopting different frozen settings. Do not overwrite
a sealed dataset's config. Benchmark results require actual parameter changes,
finite gradients and memory observations. GPU allocation is capped at 80% of the
reported device memory, and selection keeps a 30% measured peak-memory margin.
This is not a claim of optimal utilization: compare throughput, phase timing,
allocator peaks and available hardware sensors in the archive.

Once the portable smoke succeeds, a complete fresh run has a 12-hour lifetime cap,
including evaluation, checkpointing and cleanup:

```bash
python3 scripts/foundation_run.py --engine podman train \
  --profile fresh-extended-12h --inputs tmp-fresh-inputs-extended-002 \
  --run runs/fresh-portable-12h-001
```

Keep it in your usual terminal/session manager. Each run path is single use.
`--memory-gib 24 --cpus 8` can increase container resources when the host has room;
use the same options for subsequent commands on that run. Defaults use at most
half the engine's reported RAM (16 GiB cap) and four CPUs, with a bounded private
execution worker pool. `--host-free-memory-gib` sets the host reserve (default
1 GiB); `--free-disk-gib` sets the workspace free-space reserve (default 5 GiB). No unrelated GPU job or inference service is stopped.
Check hardware availability yourself before committing to a long run.

## Status, recovery and analysis

```bash
python3 scripts/foundation_run.py status --run runs/fresh-portable-12h-001
python3 scripts/foundation_run.py cancel --run runs/fresh-portable-12h-001
python3 scripts/foundation_run.py --engine podman resume --run runs/fresh-portable-12h-001
python3 scripts/foundation_run.py --engine podman evaluate \
  --run runs/fresh-portable-12h-001 --split development
python3 scripts/foundation_run.py pack --run runs/fresh-portable-12h-001 \
  --output tmp-analysis/fresh-portable-12h-001.zip
```

Run/resume/evaluate automatically publish a compact analysis ZIP, including on
failure once a run exists. It includes configuration, image/runtime identities,
input digests, budgets, training metrics, development/final results, role logs and
explicit omissions. Share this ZIP for analysis. Add `--checkpoint` to `pack` only
when you need the larger checkpoint payload. `pack`, `status`, `cancel` and
`extended-settings` do not need an engine or GPU. Private source/truth material
is not a general-purpose role mount.

Resume inspects the original image and rejects runtime/resource/controller drift;
it does not silently rebuild the image or reset elapsed budget. Preserve the
checkout location, `run-state/`, image store and run files while a run is active.
Completed/finalized runs cannot restart training or reissue their budgets. Old
controller runs are historical: begin a new portable run with the existing inputs.

## State and containment

| Artifact | Location under the checkout |
| --- | --- |
| Bulk OEIS snapshot/cache | `data.cache`, normally `tmp-oeis-bulk/` |
| Admission journal and sealed inputs | `--output` inputs directory |
| Private controller, roles, locks and ledgers | `run-state/` |
| Role-private compiler scratch and caches | `run-state/**/roles/**/.runtime/` |
| Published metrics, manifests and role logs | `--run` directory |
| Persistent container caches | `container-cache/` |
| Validation receipts/logs and analysis ZIPs | `tmp-validation/`, `tmp-analysis/` |

Container paths `/workspace`, `/work/input`, `/work/output` and `/cache` map to
these persistent host directories. Temporary `/tmp` and `/run` mounts are
intentionally disposable. Engine-managed images/layers remain in your engine's
normal storage. Build context exclusions prevent state, checkpoints, downloads
and old local credentials from entering an image.

Role processes use `TMPDIR`, `TMP`, `TEMP`, `HOME` and `XDG_CACHE_HOME` under
their existing `/work/output/.runtime/` mount. ROCm/LLVM compilation scratch and
caches therefore use workspace storage rather than the small `/tmp` tmpfs or
shared host caches. They count toward run storage limits and are excluded from
published runs, analysis ZIPs and validation receipts. `launch.json` records the
runtime paths and workspace free bytes/inodes at launch. Leave `.runtime/` alone
while its role is running; it is disposable after that role has been reclaimed.

After updating the launcher for a storage fix, rebuild with `image ensure` and
use a new smoke run name, such as `runs/portable-smoke-002`, with the same sealed
inputs. Resume rejects the changed launcher identity; do not delete the old
run's ledger or regenerate its inputs.

The host is the only engine client. Learner, generation and CPU scoring containers
receive separate enumerated inputs and outputs; no workload receives an engine
socket or the full private run directory. Containers have read-only roots,
explicit memory/CPU/pid bounds and no network except bulk acquisition. Cleanup
removes only this run's labeled containers, waits for reclamation, then publishes
success. Failure and interruption preserve the durable ledger and diagnostics.
