**CPU commands run locally; prefix GPU commands with
`lab-gpu exec --`; install packages through the Dockerfile/locks, then use
`lab-gpu image ensure`**. Current directory must be under `/workspace`, or pass
`lab-gpu --cwd relative/path exec -- ...`. Pipelines require an explicit
container shell: `lab-gpu exec -- bash -lc 'a | b'`. Interactive stdin/PTY and
arbitrary environment overrides are unsupported. Output/exit status propagate;
75 means busy and 124 means timeout.

For direct foundation work on the lab host, use `python3 scripts/foundation_local.py`
from the registered checkout. It reuses the Dockerfile GPU target with rootless
Podman, native host bind mounts and the installed GPU broker without SSH.
Host Python needs only its standard library. Run checkout code as the normal
host user; the entrypoint uses sudo only for the installed broker or the
rootless project account. The VM `lab-gpu` path remains available for tests.
