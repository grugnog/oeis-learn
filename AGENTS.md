**CPU commands run locally; prefix GPU commands with
`lab-gpu exec --`; install packages through the Dockerfile/locks, then use
`lab-gpu image ensure`**. Current directory must be under `/workspace`, or pass
`lab-gpu --cwd relative/path exec -- ...`. Pipelines require an explicit
container shell: `lab-gpu exec -- bash -lc 'a | b'`. Interactive stdin/PTY and
arbitrary environment overrides are unsupported. Output/exit status propagate;
75 means busy and 124 means timeout.
