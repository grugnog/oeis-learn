# Development in SmolVM and on the lab GPU

## Project images and host policy

This repository owns project software: edit the root `Dockerfile`, shared
`docker/dev/*.sh`, `pyproject.toml` and `uv.lock`. The `grugnog/lab` repository
owns host users, allowed workspace paths, SSH keys, device exposure, network
policy and inference coordination. Adding a project package needs no Nix rebuild.

| Target | Environment | Use |
| --- | --- | --- |
| `smolvm` (default) | Pi image + separate Python 3.12.12 environment | Agent, CPU tests, native evaluator |
| `gpu` | New foundation's pinned ROCm/PyTorch base, `/opt/venv` | Offline GPU build/test commands |

Both targets install compiler/Rust/maturin/native-evaluator dependencies and
runtime/dev Python packages from the root `uv.lock`. SmolVM explicitly installs
CPU PyTorch 2.9.1. The GPU target preserves and checks the base's HIP-enabled
PyTorch identity. Native wheels are built separately, never copied across base
images. PyO3 0.20 cannot use the Pi image's Python 3.14; the separate project
interpreter avoids replacing it. The project interpreter is first on PATH.

Use `python`, `pytest` and `maturin` directly in these images. Do not use plain
`uv sync`/`uv run`: that can create a separate environment with generic Torch.
If using uv's runner, use `uv run --no-sync --active` deliberately.

Existing `docker/foundation/Dockerfile` and `Dockerfile.simple` remain standalone
snapshot/preflight images with their original commands. Their historical
`runtime.lock.json` does not describe these new development builds. The new
targets share the foundation's dependency *roles*, but consume the current root
lock (including its newer versions), not the older foundation lock whose hashes
the existing installer strips. New exports preserve non-Torch package hashes.
Torch/setup-tool pins and base-owned dependencies are handled separately.

Rust 1.90.0 is pinned. Cargo.lock is currently ignored by this repository;
builds generate it if absent and retain it plus a Python inventory under
`/opt/oeis-build-info/`. Bitwise reproduction of independent rebuilds is not
claimed. Commit Cargo.lock to freeze native dependency resolution. A lock/source
or base change is a deployment change: rerun the CPU/GPU smoke checks.

## Build/update the SmolVM image (on the host)

Use the Podman store containing the local Pi base:

```bash
podman build --target smolvm -t localhost/oeis-learn:smolvm .
podman save --format docker-archive -o oeis.tar localhost/oeis-learn:smolvm
```

Use `sudo podman` for both commands if the base is in the rootful store. No GPU
is needed for image builds. Downloads happen at build time, including Python
from GitHub, Rust from static.rust-lang.org, crates from index.crates.io and
static.crates.io, and CPU wheels from download.pytorch.org. Host builds do not
inherit the guest's hostname allowlist. Retain the Pi base locally or set the
`PI_BASE_IMAGE` argument. Pin it by digest when available for your local image.

`Smolfile` already points to `./oeis.tar`. Preserve your workspace mount/sync
settings and create a replacement guest under a new name: rebuilding an archive
and restarting an existing VM does not update its image. Preserve guest-only
data before retiring it. Existing Docker-in-guest storage setup is left intact
for other workflows; the host GPU runner does not need it.

## Enable the broker once

Follow `grugnog/lab/USAGE.md`, **GPU commands from SmolVM**: register the actual
host workspace, provision the pinned base into its dedicated rootless store,
grant access only to that workspace, and install `lab-gpu` plus a dedicated SSH
key in the guest. The host draft keeps the supplied path
`/home/owen/workspace/oies-learn`; verify its spelling before enabling it.

Client settings (add to Smolfile's env array for future guests; export these
inside existing guests):

```bash
export LAB_GPU_HOST=10.77.0.1
export LAB_GPU_USER=labgpu-oeis-learn
export LAB_GPU_IDENTITY=/root/.ssh/lab-gpu
```

`LAB_GPU_PORT` defaults to 2222. Pin the host key in guest `known_hosts` using
the trusted host-side instructions. No administrator key or Podman socket goes
in the VM. The SSH key permits arbitrary code in this project's GPU container.

## Harness instructions

Run CPU-only tests locally. From `/workspace`, for GPU work:

```bash
lab-gpu image ensure
lab-gpu exec -- python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name())'
lab-gpu exec -- python -m pytest -q
lab-gpu exec -- python -m oeis_learn.cli.foundation_preflight \
  --config docker/foundation/preflight.yaml --output docker/foundation/evidence --hardware-only
```

`exec` ensures the image automatically. Output and exit status return to the
harness; timeout is 124 and a busy GPU is 75. The broker stops inference for GPU
execution and restores its prior active state afterwards. Other clients' active
generations may be interrupted: coordinate use before running GPU tests.

Edit packages in the Dockerfile/setup script, update `uv.lock`, then rerun
`lab-gpu image ensure`. The host snapshots `gpu-project.json` inputs and hashes
their content, including native source. Unchanged inputs reuse the image;
changed inputs rebuild with Podman layer caching. List new `COPY` dependencies
in that manifest. Do not include secrets, datasets or outputs. Changing Rust
source invalidates native-build layers; ordinary Python imports use the live
`/workspace/src` source. Synchronize guest edits first if using sync rather than
a live mount, and avoid editing files concurrently with tests.

GPU execution is offline with a read-only image and writable workspace/private
cache. Put package installation in the Dockerfile, not `exec`. Large outputs
belong in `/workspace`; no automatic image or cache pruning is performed.

## Validation checklist

1. Both images import `oeis_wasm_evaluator`, `oeis_learn` and `torch`.
2. SmolVM uses project Python 3.12 and imports source from `/workspace/src`.
3. GPU execution performs a real GPU operation without CPU fallback.
4. Native source changes invalidate the image; identical inputs reuse it.
5. Failure, timeout and SSH disconnection remove the GPU container. Inference
   restarts only if it was active beforehand.
6. GPU test code cannot reach external networks or unrelated host directories.

Image builds and live GPU checks require the lab. Source-only unit tests cannot
establish compatibility of the supplied base image with the current driver.
