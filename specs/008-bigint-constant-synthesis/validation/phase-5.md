# Phase 5 qualification (T089–T098)

Code and reproducible qualification commands are implemented in
[PR #58](https://github.com/grugnog/oeis-learn/pull/58). **The Phase 5 execution
and promotion gate is pending.** No Phase 6 screen or final-cohort study ran.

Entry is merged Phase 4 PR #57 at
`be6070bead5f30147774b6517c0f50eb1cb3f6cc`. T088 is closed by reviewed required
CI jobs and digest-verified installed/native artifacts at `8556cce`; see
[phase-4-exit.json](phase-4-exit.json). The non-critical Podman rebuild was
cancelled, separately from successful Podman execution. Historical evidence is
not relabelled as execution of this new image.

The requirements checklist passes16/16. All design documents, task list,
contracts, AGENTS and constitution were reviewed. No extension hooks exist.
The runtime change is observation-only timing around the same production run
method; numerical, solver, encoder, selection and resource policies remain frozen.
Third-party manifests/locks and long Docker dependency layers are unchanged.

| Task | Current execution status / evidence |
| --- | --- |
| T089 | [Arithmetic](arithmetic.md): new shared-image CI pending. |
| T090 | [Fitting](fitting.md): new shared-image CI pending. |
| T091 | [Containment](containment.md): mandatory Docker and Podman partitions pending; host role inspection/cleanup is also recorded. |
| T092 | [GPU](gpu.md): actual current-image HIP run required; prior device evidence remains historical. |
| T093 | [Checkpoint rescue](checkpoint-rescue.md): BLOCKED on actual retained matched-run weights/metadata. |
| T094 | [Performance](performance.md): native runtime measurements pending CI; neural measurements pending HIP; >5x results require review. |
| T095 | [Data and cost](screen-preflight.md): complete512/128/128 preparation and matching HIP p95 preflight pending. |
| T096 | [Learnability](learnability.md): bounded diverse32 run waits for its real data/HIP/cost gates; negative results block promotion. |
| T097 | [Operations](operations.md): new-image and public hardware/checkpoint-dependent commands pending. |
| T098 | This report plus [phase-5-local.json](phase-5-local.json) records pass/fail/pending identities; overall qualification remains ungranted. |

The CI partitions reuse the existing shared qualification image and uploaded
image provenance. They require nonempty zero-failure/error/skip JUnit, clean
arbitrary-UID packaging and unchanged retained numerical controls. GPU execution
is separate. New CI is left running after pushes as requested.

## Local checks and limits

The affected CPU/policy/codec/reference and tooling set passes **135 tests with
one explicit packaged-native measurement skip**, zero failures/errors. Ruff
F/E9, Python compilation, workflow YAML/shell syntax, Spec Kit prerequisites,
artifact structure and source retirement pass. The source/JUnit hashes and
per-task statuses are in [phase-5-local.json](phase-5-local.json). Small mocked
measurement tensors test checkpoint and fixture plumbing only; they are not
reported as hardware performance, actual full-model training or learned accuracy.
This agent environment has no Docker/Podman/Rust/HIP toolchain. Native compilation,
new-image execution/benchmarks, real data preparation and GPU learning are not
claimed locally. Required CI/image runs fail rather than skip missing artifacts.

## Copy-pastable hardware commands

From this PR checkout, one CPU/native/retirement/performance qualification:

```bash
python3 scripts/qualify_bigint_phase5.py --engine docker \
  --output tmp-validation/008-phase5-docker-001
```

On the HIP machine, run the same CPU/native checks, actual GPU update/cache
suite, capacity/shape cost measurements, and (only after passing preflight) the
bounded diverse32 gate:

```bash
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --prepare --learnability \
  --output tmp-validation/008-phase5-hip-001
```

The second command may spend up to the registered preparation allowance plus
7200s of training/diagnostics. It launches no Phase 6 screens. Use Docker instead
if preferred. With multiple GPUs, add `--render-node /dev/dri/renderD128` for the
intended device. For existing prepared data, replace `--prepare` with
`--inputs PATH`. Omit `--prepare --learnability` for a hardware/performance-only
run. An explicit `--image IMAGE` must contain the exact current source snapshot;
stale images fail before later qualification. New code means a new image/run.

Each invocation prints its ZIP path, e.g.
`tmp-validation/008-phase5-hip-001.zip`. Share that archive, including failed or
blocked outcomes. It contains summaries, exact image/source/artifact hashes,
receipts, JUnit, logs, observed isolation/cleanup, measurements and optional
bounded run bundle, with an omission manifest. It omits private data/caches and
weights. Use a new suffix (`002`, etc.) for a retry; earlier evidence stays intact.

A matched retained-run diagnostic can additionally be requested with
`--checkpoint-run runs/ACTUAL_RUN`. No such run was supplied locally. Supplying
only a canonical program or a newly trained bigint model cannot satisfy T093.
No task is checked off merely because these commands/tests exist. The phase can
be closed only after required execution/review, including genuine HIP, matched
checkpoint diagnosis, cost and learnability results, has a recorded outcome.
