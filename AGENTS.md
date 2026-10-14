Use the portable host CLI: `python3 scripts/foundation_run.py --engine docker|podman COMMAND`.
The host wrapper requires only Python's standard library and a usable engine connection.
Do not add host account, service, distribution, VM or SSH dependencies.
Build dependencies through the root Dockerfile and committed locks. The default target
is `gpu`; `qualification` provides CPU/native tests. Roles get explicit mounts and no
runtime socket. Persistent downloads, inputs, run state, caches and reports belong in
the workspace. Keep GPU tests marked and distinguish actual HIP evidence from CPU tests.
