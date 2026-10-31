# HIP acceptance — qualification passed; feature acceptance failed

Current status after operator acceptance-009 (2026-10-31): CPU/native and actual
HIP qualification passed; feature acceptance remains FAIL / UNMEASURED. The
canonical archive verifies. The first retained-checkpoint generator exited139
during its first scheduled diagnostic stack dump, before its90-second external
deadline. T100 replaces automatic stack walking with timestamp-only heartbeats.
This removes a strongly suspected diagnostic regression; the precise native crash
cause remains unproven. Replay and its longer learner allowance were not exercised.
See the final section below. Earlier NOT RUN statements are historical records.

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

## Acceptance-007 supplemental records and repairs — 2026-10-31

`acceptance-007-diagnostics.zip` passes independent inventory/hash verification;
SHA-256 `b4fd28e03d9467cbde59f2b554c221804577ecb837e93f62982d21da31a1c6cd`.
The replay/resume and report role terminal record seals verify. Replay/resume
failed before bootstrap validation/learning when Podman removal of a consumer
image-inspection container exceeded its remaining4.999-second cleanup allowance.
The record reports ResourceBreach and unacknowledged termination; this is preserved
as failure, not treated as a successful or automatically reusable inspection.
Report packaging returned ValueError `private acceptance payload`. Retained
evaluation did publish an initializer; missing selection was not the failure.

T093 reproduces the packaging failure using the canonical producer's pinned build
filenames and corrects their public archive policy without changing original
authority. T094 reserves30 seconds for image-inspection cleanup inside the
existing wall bound and retains failure on unacknowledged removal. It also captures
consumer inspection controls. These are local policy/containment regressions, not
new Podman timing measurements. T095 fixes a separate downstream defect found in
review: initializer schema validation ran on the stdlib-only host. It now runs in
the existing preparation role and returns the plan for reuse, with no extra launch.

All three repairs have independent failing-before/passing-after regressions. The
combined local selection passed334 tests and4 subtests, with one hardware test
deselected in19.34 seconds. No HIP command was executed locally. Preserve the
original failed acceptance-007 and its passed qualification receipts as evidence
for their original source. The new source changes invalidate automatic producer
reuse; rebuild both images, freeze new acceptance-inputs-008, and use one normal
serial acceptance-008 invocation for requalification and all remaining cases.
No manual one-off probe or research run is required. Feature acceptance remains
unqualified until that actual session and archive verification complete.

## Operator acceptance-008 — archive verified, replay blocked (2026-10-31)

Uploaded archive SHA-256:
`46dc1b71e3b962b4132fc1e2b1ab7a41952e83d532fd3490352aaaa089442f61`.
Independent `generalization_archive.verify_archive` and `bundle.verify` both
pass integrity checks for282 inventoried members. Reconciliation reports software
FAIL, scientific UNMEASURED, retained-eval/refresh/parameter-targets PASS,
replay-resume FAIL and missing failure-packaging; feature_qualified is false.
This is the canonical archive, not recovery packaging. Its workflow is the
pre-packaging snapshot and does not establish the outer wrapper's final exit.

Original qualification logs report1105 CPU/native tests and77 subtests passed
in336.07 seconds, and27 HIP tests passed in342.29 seconds. Captured images:

- CPU: `sha256:cb07fd322232a92348301d062367b3f99a37d38e5bd65ae8486751a0ac6e6154`.
- GPU: `sha256:2686b50d900ad6bf9993a7db2548fe99968a897017ed0f86883166feed251f20`.
- Source closure: `sha256:c83dc301c43b76fea8b47ea529224286a1c1e88f12fb74d1f8e02e97a77385f2`.

Replay bootstrap validation completed. Replay then reported `container termination
unacknowledged: TimeoutError: cleanup deadline exhausted before container reclamation`.
Captured consumer inspection/validation controls completed and acknowledged
cleanup; the nested replay roles' controls are absent. The initial failure and
its location cannot be determined from this archive. A reproduced supervisor
defect explains how a stuck client can consume the whole cleanup envelope,
masking that initial failure; T096 bounds its default graceful wait without
extending any deadline. T097 captures nested controls in future normal/recovery
archives and exports them from existing runs with no containers or HIP execution.
Export acceptance-008's existing files before selecting another repair or timeout.
T064–T067 remain open; the original passed qualification is preserved for its
original source, and these new repairs do not inherit that qualification.

## Acceptance-008 supplemental replay controls — 2026-10-31

`acceptance-008-diagnostics.zip` independently passes inventory/hash verification
with291 inventoried members and diagnostic-only PARTIAL status. SHA-256:
`39e8c2730e672c0886418e1365256189e7248bad9ba61e019ed48de85d6c5cc6`.

All four `search/private/attempts/*/audit/role` processes completed successfully
in1.097–1.127 seconds with acknowledged cleanup. The first continued_sft learner
process then hit its30-second external deadline; its process record measures
35.002887 seconds including the exhausted5-second cleanup. Its command binds
the original retained architecture (width768,6 encoder/decoder layers, FP32),
512 target tokens and the selected initializer. Stderr contains Torch model and
attention warnings, no traceback. It proves the role started and exceeded its
wall allowance, but does not identify the exact active training phase.

T098 raises the P4 learner and current-policy evaluation allowances to180 seconds
and the replay/resume case to35 minutes, retaining the same work/seed/model.
The entire serial schedule still fits five hours by reallocating20 minutes from
CPU qualification margin; the actual CPU suite took336.07 seconds and still gets
35 minutes. All suites, qualification integrity, cleanup and packaging requirements
remain enforced. Do not edit or reuse the old frozen input as fresh evidence.
Rebuild the two images, assemble acceptance-inputs-009, and run one normal serial
acceptance-009 session to perform qualification and all cases, then package and
verify results. T064–T067 remain open pending that actual evidence.

T099 additionally captures learner/generator startup and periodic stack diagnostics
plus bounded learner phase markers in the normal nested control inventory. This
avoids another manual log-export step if the next run fails. Combined local tests
after T098/T099:348 passed and4 subtests, with one hardware test deselected. These
tests do not establish that180 seconds is sufficient on the operator's hardware;
the fresh serial session must measure that outcome.

## Operator acceptance-009 — generator crash during stack dump (2026-10-31)

Archive SHA-256:
`31a4f1175fd13a26a227d8e95a9b1b42dd79f876df6bd0fd49b2e8e118ecde52`.
Independent inventory and canonical evidence verification pass for187 inventoried
members. Reconciliation reports retained-eval FAIL and the other four cases
missing, scientific UNMEASURED and feature_qualified false. The workflow snapshot
records preflight, qualification and preparation success, then retained-eval failure
at1162.55 cumulative seconds. It is a pre-packaging snapshot, not the final outer
wrapper ledger.

Original suite output:1119 CPU/native tests and77 subtests passed in339.16 seconds;
27 HIP tests passed in345.79 seconds. Image/source identities:

- CPU: `sha256:179475758c4014440636907300455d7224a0c8addc4279f5f13ad6364539fc77`.
- GPU: `sha256:e69a12205a397e9457bb71f4399dc1798bf26bc70f5dd0fdfc2b0335eb23b332`.
- Source closure: `sha256:3a37fe9fae2098c38a9bd68cf5c739b469725bc76943523f5d28d36c4b65ea52`.

The first retained checkpoint's generator had a90-second limit. It started Python,
emitted the model warning, and began the scheduled30-second traceback. The dump
ends midway through the first `torch/nn/functional.py` frame, without a line number
or function. Podman then reports exit139. The process record measures32.390348
seconds, with zero omitted log bytes and no cleanup failure; container removal was
acknowledged in0.034760 seconds. The diagnostic collector copied the entire41656-byte
stderr file. This is a crash, not an external deadline or excerpt truncation.

The timing and interrupted dump make T099's asynchronous stack walking a strong
suspect, but no native core/backtrace establishes causality. Primary references
read during investigation:

- [CPython issue158200](https://github.com/python/cpython/issues/158200) reports
  `dump_traceback_later` crashing while code metadata is freed; its reported build
  is CPython main/3.16, not this operator's3.12 runtime.
- [CPython issue116008](https://github.com/python/cpython/issues/116008) records
  threaded faulthandler crashes accessing frame/code metadata. It is supporting
  context, not proof that this run hit the same defect.
- [Python3.12 faulthandler documentation](https://docs.python.org/3.12/library/faulthandler.html)
  specifies that timed dumps use a watchdog thread and describes descriptor
  lifetime requirements. The supervisor kept stderr open until process exit.

Three bounded one-second attempts of the upstream frame-churn reproducer on local
CPython3.12.14 all exited0. The operator crash was not reproduced locally. T100
therefore removes the suspect mechanism from roles and inspection while preserving
startup/phase/engine logs, external containment and all acceptance/science limits.
The replacement only prints timestamps and never inspects frames. Fresh rebuilt
images and acceptance-inputs-010 are required; run one consolidated acceptance-010
invocation rather than separate hardware probes. T064–T067 remain open.
