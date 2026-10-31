# HIP acceptance — NOT RUN

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
