# HIP acceptance — qualification passed; feature acceptance failed

Current status after operator acceptance-007 (2026-10-31): CPU/native and actual
HIP qualification passed on source `2501929`; overall feature acceptance remains
FAIL / PACKAGE_FAILED / UNMEASURED. Replay/resume and reporting failed. Their
terminal result records are absent from the uploaded recovery archive and must
be retrieved before a targeted repair or another hardware run. See the final
section below. Earlier NOT RUN statements are historical records.

## Historical local baseline, before operator runs

Status recorded 2026-10-31: **NOT RUN / UNQUALIFIED**. The user explicitly excluded
running HIP commands. No HIP parameter update, checkpoint load, replay/resume run,
real acceptance archive or new scientific result is claimed.

The serial helper and canonical commands are implemented. A synthetic sealed-session
dry run verifies ten stages, one qualification, 13,800 seconds of reservations plus
600 seconds of recovery margin. CPU tensor tests and fixture acceptance/archives
are documented in [local-checks.md](local-checks.md); they are not HIP evidence.

Before an operator starts the single normal session, complete T063's pinned-image
regressions, T064's image build/inspection and source/build closure recording, and
T065's reduced real-input preflight. Docker/Podman/Cargo and the native synthesis
executable are unavailable here; local historical tests also lack `wasmtime`.
No immutable production image identities or hardware observations are invented.

T066–T068 remain open. After an authorized hardware session, record its original
producer and exact image/source identities, input/session/helper hashes, command,
start/end times, stage reservations and measured costs, five case receipts, complete
or partial archive identity and independent verifier result here. Preserve separate
software/scientific statuses and any omitted evidence. A failed session remains
failed; record any targeted repair, invalidated producer and justified minimal rerun
before replacing evidence. No rerun was attempted in this environment.

## Current launch input status — 2026-10-31

No HIP commands executed. T080–T084 completed the missing input/setup dependency:
`generalization_inputs.py acceptance` constructs the reduced session from the
verified uniform import and explicitly inspected image tags. During that session,
retained evaluation publishes a new selection-based weights-only initializer
and resolves replay/resume through a verified dependency transition. The import's
`initializer_available: false` and `resume_available: false` remain accurate.

The final local selection passed 272 tests (one hardware test deselected). Actual
image builds, native validation, real-input preflight, HIP updates and the real
archive remain unverified here. See [local-checks.md](local-checks.md) and follow
[quickstart.md](../quickstart.md) for the single normal serial invocation.

## Operator acceptance-002 — failed before HIP (2026-10-31)

The uploaded `acceptance-002.zip` verifies as an intact nine-member recovery bundle
with overall FAIL / PACKAGE_FAILED. Its workflow records preflight exit 0 at
59.49 seconds, qualification exit 1 at 59.78 seconds, and report exit 1 at
60.47 seconds (cumulative elapsed times). Both `image.qualification` and
`image.gpu` failed immediately; no CPU semantic or HIP stage start is logged.
The report then failed because no `qualify/qualification.json` was produced.
This secondary missing-authority error does not explain the image failure.

The recovery bundle omits detailed qualification receipts, which remain in the
operator's `qualify/pipeline.zip`. Inspection of the published source identified
a deterministic integration defect matching the immediate failures: the runner
creates each stage directory, then the adapter passes that existing directory to
an inspector requiring a fresh directory. A local real-runner/real-inspector
regression reproduced FileExistsError before any container launch. T085 fixes the
call to use a fresh `inspection/` child and emits bounded original failure reasons
into stdout retained by recovery packaging. The operator's exact exception remains
unconfirmed until its original detailed receipt is read.

Preserve acceptance-001/002. Source and qualification identity changed, so build
both images and assemble new inputs from the existing uniform import before the
next session, using Podman and fresh acceptance-inputs-003/acceptance-003 roots.
No prior qualification PASS exists to reuse. Do not edit old inputs or overwrite
evidence. Local reproduction is not hardware qualification; T063–T068 stay open.

## Operator acceptance-003 diagnostics — 2026-10-31

The uploaded recovery archive and separately supplied diagnostics confirm
preflight failed with TimeoutError at the 120-second external deadline while
launching the parameter-targets `validate-runtime` role through Podman. The
process record reports 121.42 seconds including cleanup and return code 125.
Both captured output streams are empty, with zero omitted bytes. Container
reclamation was acknowledged within 0.30 seconds. These facts do not locate the
stall inside engine startup, Python import or runtime validation. No role result
was published; qualification and HIP did not start. The later missing
qualification.json report error is secondary. No timeout increase or new
hardware rerun is justified by this diagnostic-capture change alone.

T086 now captures these records automatically in both normal and fallback
archives, with explicit bounds and omissions. The timeout cause remains open;
T063–T068 retain their evidence gates.

## Operator acceptance-004 — qualification failed (2026-10-31)

The 48-member recovery archive passes inventory/hash verification and the captured
original receipt seals verify. Preflight passed all four cases. The first case
took 116.86 seconds; the remaining cases each completed in about 0.18–1.16 seconds.
The CPU qualification image inspection passed in 0.41 seconds. The CPU regression
suite failed after about 336 seconds; its original stage receipt identifies
`bigint-cpu-regressions/results.xml` and stdout/stderr, but T086's capture omitted
those nested suite files. The failing tests cannot yet be identified from this
archive. They remain in the operator's original `qualify/pipeline.zip` and run
directory. T087 fixes the capture omission for future archives.

The GPU image inspection independently timed out at its 240-second external
deadline with empty streams. Cleanup was acknowledged. The HIP stage was BLOCKED
by these two prerequisites; no HIP test or learner started. Reporting then failed
for absent qualification.json and the wrapper preserved the failed recovery
archive. Image inspection is CPU-only and its timeout does not prove a GPU fault.
No timeout increase or speculative cause is asserted. Retrieve existing test
evidence before any new qualification run; T063–T068 remain open.

### Acceptance-004 qualification detail received

The original qualification archive confirms exactly three failing setup fixture
tests due to insufficient free space in `/tmp/bigint-pytest`. The other 1,081 tests
and 77 subtests passed; eight tests were deselected. T089 repairs test resource
observations and verifies production rejection remains effective. This supersedes
the earlier statement that the failing test identities were unavailable. It does
not change the original failed qualification receipt or resolve the independent
GPU-image inspection timeout. Rebuild both images and freeze fresh inputs for
the next complete serial session; no original PASS qualification exists to reuse.

## Operator acceptance-007 — qualification PASS, feature FAIL (2026-10-31)

The 117-member recovery ZIP passes inventory/hash verification. Archive SHA-256:
`bb4d781fb1d8bf8ca3d6425f41eeca1510399008ebf38c4f7cc213980e648cb8`.
The four original qualification stage receipts have valid seals and PASS states.
CPU stdout reports **1,101 passed, 77 subtests passed, eight deselected** in
335.65 seconds; JUnit records 1,178 successful cases, zero failures/errors/skips.
HIP stdout reports **27 passed** in 345.24 seconds, with zero JUnit failures or
skips. The HIP environment reports PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, HIP
`7.2.53211-e1a6bc5663`, native worker available and AMD Radeon Graphics `gfx1151`.
These are actual operator container/HIP results, not local fixture qualification.

Captured image/source identities:

- CPU: `sha256:750d1bb3e05c90f1a1e19d83da975f49258a35d5b5804b787729df2a0ae9d17a`.
- GPU: `sha256:b2f639bd31a9b8a78e1e881d3bc546d9d586f8733249a187803ed2322eda0a30`.
- Source closure artifact: `sha256:e0cd8dc3869cd16b80e2a54caadd1855a8e128488a84fa286b87c1ebf93a25a7`.
- Serial helper: `sha256:692a5b9dfc11e08b5b06d6465c433bb7f3ad298e81c9c13c88e670037bdd9317`.

The first CPU preflight completed in 116.86 seconds; its Python portion took about
0.64 seconds. The GPU image inspection completed in 309.88 seconds, exceeding the
old 240-second deadline; Python inspection took about 0.09 seconds. Thus the larger
startup allowances enabled this run to progress, with most cold-start time before
Python. The exact overlay/storage mechanism remains unproven.

The wrapper records preflight, qualify, prepare, retained-eval, refresh and
parameter-targets exit 0; replay-resume exit 1; report exit 1. Failure-packaging
and verify-bundle were not executed. Total recorded elapsed time was 1,328.41
seconds (22 minutes 8 seconds). Overall status remains FAIL / PACKAGE_FAILED;
scientific status remains UNMEASURED. Passing qualification and earlier cases
do not establish full feature acceptance or replay/resume correctness.

The recovery collector omitted top-level acceptance results, case receipts and
report packaging-role results; replay/report stdout and stderr are empty. Their
exact failures cannot be diagnosed from this archive. T092 adds those fixed public
paths and a read-only stdlib exporter. Retrieve the existing acceptance-007 records
using that exporter before deciding a repair/rerun. No additional HIP invocation
is requested for evidence collection. Source changes invalidate automatic reuse
of this producer for a new revision, but do not alter the historical PASS results.
T063 and T068 are recorded complete for this observed scope; T064–T067 remain open.
