# Running with Docker or Podman

Use Python 3.11+ on a Linux host with a working Docker or Podman connection and a
ROCm-supported GPU/driver. The application uses the current user's engine access.
All training dependencies and the native evaluator are built in the public image.

```bash
python3 scripts/foundation_run.py --engine podman image ensure
python3 scripts/foundation_run.py --engine podman validate --suite cpu --output tmp-validation/cpu
python3 scripts/foundation_run.py --engine podman validate --suite gpu --output tmp-validation/gpu
python3 scripts/foundation_run.py --engine podman smoke \
  --inputs tmp-fresh-inputs-extended-002 --run runs/portable-smoke-001
```

Replace `podman` with `docker` to use Docker. Omit `--engine` for auto detection.
The host wrapper needs only the Python standard library; workload dependencies
are pinned in the image. Run it on the host rather than inside a workload.

See [fresh training](docs/fresh-training.md) for bulk OEIS acquisition, reuse of
prepared inputs, benchmarking, a 12-hour training command, recovery, resource
options, artifacts and containment. `foundation_local.py` remains an alias.
Persistent data and controller state live under the checkout; temporary container
files can be discarded. Existing sealed inputs and bulk caches remain reusable.

The root Dockerfile has two targets: default `gpu` for ROCm and explicit
`qualification` for CPU/native CI. Both build the Rust extension and locked Python
dependencies. CPU-only validation does not count as GPU acceptance.
