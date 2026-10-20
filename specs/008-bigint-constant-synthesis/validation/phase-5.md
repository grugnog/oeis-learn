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

Update this PR checkout before running; the wrapper builds and binds the exact
current source rather than reusing an older image:

```bash
git fetch origin
git switch implement/008-phase5
git pull --ff-only
```

One CPU/native/retirement/performance qualification:

```bash
python3 scripts/qualify_bigint_phase5.py --engine docker \
  --output tmp-validation/008-phase5-docker-001
```

On the HIP machine, run the same CPU/native checks, actual GPU update/cache
suite, capacity/shape cost measurements, and (only after passing preflight) the
bounded diverse32 gate:

```bash
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --prepare --learnability \
  --output "tmp-validation/008-phase5-hip-$(date -u +%Y%m%dT%H%M%SZ)"
```

The second command may spend up to the registered preparation allowance plus
7200s of training/diagnostics. It launches no Phase 6 screens. Use Docker instead
if preferred. With multiple GPUs, add `--render-node /dev/dri/renderD128` for the
intended device. For existing prepared data, replace `--prepare` with
`--inputs PATH`. Omit `--prepare --learnability` for a hardware/performance-only
run. An explicit `--image IMAGE` must contain the exact current source snapshot;
stale images fail before later qualification. New code means a new image/run.

Each invocation prints its ZIP path, e.g.
`tmp-validation/008-phase5-hip-20261020T120000Z.zip`. Share that archive, including failed or
blocked outcomes. It contains summaries, exact image/source/artifact hashes,
receipts, JUnit, logs, observed isolation/cleanup, measurements and optional
bounded run bundle, with an omission manifest. It omits private data/caches and
weights. Use a new suffix (`002`, etc.) for a retry; earlier evidence stays intact.
The timestamp in the command creates a fresh output automatically on each rerun.
If preparation finished successfully, reuse its inputs while preserving the
earlier evidence (replace the example timestamp with that run's directory):

```bash
python3 scripts/qualify_bigint_phase5.py --engine podman --gpu --learnability \
  --inputs tmp-validation/008-phase5-hip-20261020T120000Z/inputs \
  --output "tmp-validation/008-phase5-hip-$(date -u +%Y%m%dT%H%M%SZ)"
```

A matched retained-run diagnostic can additionally be requested with
`--checkpoint-run runs/ACTUAL_RUN`. No such run was supplied locally. Supplying
only a canonical program or a newly trained bigint model cannot satisfy T093.
No task is checked off merely because these commands/tests exist. The phase can
be closed only after required execution/review, including genuine HIP, matched
checkpoint diagnosis, cost and learnability results, has a recorded outcome.

## CI discovery follow-up

At `c2a91dcb44d99278ef17e45e4f397a5a6f7dccca`, all five new bigint
qualification partitions passed, as did native qualification, both preparation
partitions, the image/Podman builds and portable Podman execution. Foundation
review run39364753190 passed. Container run39364753246 failed only portable
Docker's CPU regression: the two-program archive test returned no relation
claims (1089 other tests passed). These job statuses do not replace review of
the qualification artifacts or qualify the follow-up source/image.

The test passes locally. Its exact relation can be silently omitted when the
contained fitter exhausts its default two-second spawn/import/IPC deadline.
A controlled 2.2-second cold SymPy import reproduced UNKNOWN at two seconds
(2.021s total), then FIT with coefficients `[1, -1]` at ten seconds (2.526s).
The original CI log did not retain the fit status, so its precise resource
failure cannot be recovered. The discovery entry point now explicitly uses
the existing ten-second maximum per candidate and records every fitting
disposition, including UNKNOWN, in the sealed report. The killable worker,
1-GiB memory cap, coefficient bounds and independent finite validation remain
in force. No bigint numerical/solver profile or training recipe changed.

Regression coverage checks the successful real archive path, propagation of
the configured deadline, preservation of UNKNOWN without a claim, rejection
of unbounded deadlines, and the fitter's existing immediate-timeout behavior.
The earlier [local report](phase-5-local.json) remains evidence for its original
source, not this follow-up. HIP and matched-checkpoint gates remain pending.

Follow-up local checks: **68 passed, 3 skipped** across complete discovery,
Phase 5 runner/benchmark and symbolic-proof suites. All three skips require the
packaged native certificate solver. Ruff F/E9, formatting, whitespace checks
and standard-library-only runner help pass. A broader CLI-contract attempt had
two environment failures because this sandbox lacks
`/proc/sys/kernel/random/boot_id`; those container-only paths await CI rather
than changing their boot-identity checks. Follow-up CI is left unmonitored.

## Bulk fetch hook compatibility follow-up

The HIP attempt `008-phase5-hip-20261020T185302Z` reached bulk fetch but stopped
at `git lfs install --local`: Git LFS 3.4.1 was present and rejected an existing
pre-push hook. This is initialization failure, not a missing image dependency
or HIP test result. Fetch now uses `git lfs install --local --skip-repo` to
configure repository filters while preserving all hooks, including custom
`core.hooksPath`. The download cache never pushes. Pinned plans, selected paths,
checksums and batched downloads are unchanged.

Actual Git/LFS regression tests reproduced the failure before the fix and now
verify repeated cached fetches, filter setup, untouched older/custom hooks and
unchanged plan/receipt hashes. Bulk and qualification-runner tests pass23/23;
Ruff and whitespace checks pass. No image dependency layer changed. New CI is
left running without monitoring; actual HIP qualification remains pending.

After updating this branch, rerun the timestamped `--gpu --prepare --learnability`
command above. Retain `tmp-oeis-bulk`: its pinned snapshot, plan and verified
downloads are reused. Use a fresh output directory; do not use `--inputs` for
this attempt because fetching failed before preparation produced complete
inputs. The wrapper rebuilds for the new source and prints the new result ZIP.

## Unresolved LFS object recovery

The next HIP attempt `008-phase5-hip-20261020T191140Z` passed retirement, then
bulk fetch reported success for a batch but checkout left
`files/A373/b373464.txt` as a 130-byte pointer. User diagnostics confirm that
its plan, pinned Git blob and working pointer all identify the same
29,047-byte object, SHA-256
`c0896a08bd0248bc1ee96779af5fdf1d21a1d69f2be38fc330d7b1e9849218f8`,
at pinned commit `f0fb257e855fb356cc50a580e41e081a41443bd0`.
A clean Git/LFS status therefore did not establish materialized input data.
The original cause of the batch omission has not been established.

Fetch still batches the frozen selected paths and verifies every size/hash.
For an unresolved pointer or missing file only, it now checks the exact Git
pointer against the existing plan and requests that object through `git lfs
smudge`. It clears smudge-skip/filter exclusions for that command, allows one
120-second attempt under the existing aggregate fetch/container deadline,
monitors output size, and kills/reaps the process group on failure. Output
stays in a temporary workspace file until exact size/SHA-256 verification
passes, then replaces the pointer atomically. A changed working file is never
overwritten. An unavailable object fails with its path/OID and leaves the
pinned plan, pointer and earlier verified downloads intact. No HTTP API,
cohort change, new snapshot, threshold change or dependency layer was added.

Local tests use actual Git LFS 3.4.1 and a local file-backed remote, with the
observed successful-but-incomplete batch injected explicitly. They cover
successful recovery and reuse, missing remote data, changed working files,
stale plans, temporary-file cleanup and untouched unselected pointers. The
bulk, qualification-runner and benchmark tooling suites pass **34/34**, with
zero skips. Ruff and whitespace checks pass. The exact remote omission and
actual HIP execution still require the user's run; CI is left unmonitored.

To test fetch first without repeating the qualification suites:

```bash
git pull --ff-only
python3 scripts/foundation_run.py --engine podman data fetch \
  --config configs/foundation/bigint_smoke.yaml
```

After fetch succeeds, run the timestamped `--gpu --prepare --learnability`
command above. Keep `tmp-oeis-bulk`; verified objects and the frozen plan are
reused. If recovery fails, share its `RECOVER LFS object` line and subsequent
stderr; the object is not silently excluded or counted as complete.
