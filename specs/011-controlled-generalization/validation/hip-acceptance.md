# HIP acceptance — original qualification passed; feature acceptance failed

Current status after acceptance-014 (2026-10-31 Pacific): CPU/native and HIP
qualification, retained evaluation and refresh passed. The parameterized learner
expired during checkpoint persistence within its30-second envelope (about24 seconds
of process work). Replay/resume was not reached. Independent canonical archive
reconciliation returns integrity PASS, software FAIL, science UNMEASURED.

T113 enlarges both small-model learner/proof envelopes and case reservations. The
current serial ceiling is395 minutes (6h35). New-source hardware validation remains
pending; acceptance-014 does not exercise the prior post-training evaluation repair.
Earlier records below retain their original qualification scopes.

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


## Operator acceptance-010 — missing negative role outcome (2026-11-01)

Archive SHA-256:
`bc28e3d8190ce92c1d8e3aa8397fea9c71df6be2118018604fa3da4467d79709`.
Independent inventory/hash and canonical evidence checks pass for354 inventoried
members. Retained-eval, refresh and parameter-targets pass; replay-resume fails;
failure-packaging is missing. Software is FAIL, scientific status UNMEASURED and
feature_qualified false. The pre-packaging workflow snapshot records replay failure
at1388.11 cumulative seconds; it does not establish the final outer wrapper exit.

Original qualification output records1120 CPU/native tests and77 subtests passed
in339.34 seconds, and27 HIP tests passed in343.93 seconds. Original identities:

- CPU: `sha256:6590aa333a952c4b823f3b87bca786254ee942ab9f231037e660c698270c144c`.
- GPU: `sha256:a6112e942016395ea9b62b896f53d20f1f102417cd8dd51cc636dcbd6153e477`.
- Source closure: `sha256:9d4e83502510e5eef6dfaf61fd740c832b754d6b3c8b760be6e245d847b3e75d`.

All four replay audit processes completed. The first continued_sft learner emitted
three completed updates and checkpoint writes in approximately86 seconds. The
180-second allowance accommodated this execution; its log is not the independent
paired finite-update proof, which was not completed. Timestamp heartbeats ran
without the previous stack-dump crash in this run; this does not prove that earlier
crash's cause.

The verified_replay learner's process completed with exit0 in1.815868 seconds and
acknowledged cleanup. It emitted Python-start but no batch-input marker or traceback.
The canonical parent reports incomplete paired updates/replay/resume/admission.
Installed role handling writes negative application results to `role/result.json`
while returning0 for transport. The nested result was omitted from the archive,
so the original failure reason is unknown. It is not evidence of another timeout;
replay restoration/native setup is a possible location, not a diagnosed defect.

T101 exports only bounded status/reason/error fields from existing nested results
and emits negative outcomes on stderr in future images. Other role payload fields
remain excluded. Export the existing acceptance-010 files with the quickstart's
stdlib-only command before choosing another repair or HIP run. T064–T067 remain
open; no changed source inherits this run's original qualification.


## Acceptance-010 supplemental outcome — 2026-10-31

`acceptance-010-diagnostics.zip` passes independent bundle inventory/hash validation
with313 inventoried members and diagnostic-only PARTIAL status. SHA-256:
`dd11c83fe80c2b3829863c27109fef0674933dd49aa188f5fdae48e18b9615f3`.
All18 nested result projections are present. The continued_sft learner reports
PASS/COMPLETE; the verified_replay learner reports FAIL/ValueError with exactly
`replay evidence changed on restore`. This localizes failure to the comparison
following a successful independent re-admission, before tensor training begins.
It does not indicate an exhausted timeout or failed100-term agreement.

The adapter used the qualification image for initial replay audits and the GPU
image for learner restoration. Each image independently builds its synthesis
worker in `build-native-wheel.sh`; the shared synthesis build does not replace
that worker. `SynthesisClient.runtime_identity` binds exact worker bytes, guest,
profile and engine policy. This routing assumes binary identity across separate
builds. The supplemental projection does not include the original entry or newly
computed entry, so it cannot prove which field differed in the operator run.

T102 removes that unsupported assumption: run the isolated CPU audit in the
learner's frozen image with GPU access explicitly disabled. A fixture with distinct
image runtime identities reproduced the same exception in both proposal paths and
the real paired/resume adapter path before the change. Same-image routing fixes
those regressions while retaining exact admission/restore checks. Restore mismatch
errors now name only canonical field names for any further discrepancy. This is
routine image routing and diagnostics, not a new Phase-B admission policy.

Preserve acceptance-010 and its qualified source/image scope. T064–T067 stay open
for fresh images and acceptance-inputs-011, followed by one consolidated serial
acceptance-011 run. Do not rerun standalone HIP probes or extend training merely
to bypass the mismatch. This acceptance remains software validation, not research.

## Operator acceptance-011 — checkpoint deadline and cleanup (2026-11-01 UTC)

The original `acceptance-011.zip` independently verifies all 372 inventoried
members. Archive SHA-256:
`8d75032bf8289056a96084b63ae84790fe2d73fcce34ab8da1f5d71449c86d61`.
CPU/native JUnit contains 1,213 records and HIP JUnit contains 27, with zero
failures, errors or skips in either. JUnit record counts are not substituted for
pytest's test/subtest totals. Retained-eval, refresh and parameter-targets pass;
replay-resume fails and science remains UNMEASURED.

The continued-SFT process completed in 122.78984 seconds. Verified replay passed
admission/restoration, executed updates and timed out during checkpoint persistence
at its 180-second external deadline. Completed checkpoint saves took approximately
25.90 and 45.23 seconds. The process observation elapsed 185.00169 seconds with
no return code and `process-group termination unacknowledged within cleanup
deadline`. Engine cleanup was unacknowledged with a five-second allowance and
`cleanup deadline exhausted before container reclamation`. These records establish
the timeout location and cleanup exhaustion; they do not establish full resume
correctness or completion of downstream proofs.

T106 charges setup/copy/persistence to enclosing allowances, preserves pending
update state, and reserves cleanup before work. T108 avoids duplicate full-state
copies and includes paired/paused/resumed/proof peaks in preflight. The generated
acceptance session allows 600 seconds per replay learner, 180 seconds per replay
proof and 70 minutes for replay/resume, within 335 minutes overall. Existing
26 GiB P4 / 32 GiB aggregate checkpoint limits and scientific budgets are unchanged.

The broader T103–T109 repairs and local regressions are in
[operational-repairs.md](operational-repairs.md). The existing HIP smoke check now
also performs installed full-state learner/resume verification in the same serial
qualification run; its execution on changed code is still pending. Preserve this
original archive and do not transfer its PASS qualification to new source/images.

## Operator acceptance-012 — qualification tmpfs exhaustion (2026-10-31 Pacific)

`acceptance-012.zip` passes independent inventory/hash verification: 74 inventoried
payloads plus the manifest, 383,055 archive bytes. SHA-256:
`ff4a316cc43667a1c194150f9702319c30ef932a8ee6d80a80e251f3cd5d2736`.
The final workflow records preflight PASS at 156.576 cumulative seconds, qualification
FAIL at 691.687 seconds and report FAIL at 692.656 seconds. Report lacked the
qualification authority because qualification failed; the wrapper correctly
preserved FAIL / PACKAGE_FAILED / UNMEASURED recovery evidence.

CPU qualification reports **33 failed, 1,146 passed, 8 deselected and 77 subtests
passed in 329.17 seconds**. JUnit has 1,256 records, 33 failures, no errors or skips.
The first recorded failure is checkpoint copying in
`test_pilot_actual_worker_state_and_metric_resume[cpu]`, with ENOSPC under
`/tmp/bigint-pytest`. Twenty-nine failures directly report ENOSPC; three subprocess
failures have tensor-writer errors in captured stderr, and one subsequently reads
an incomplete checkpoint (EOF). This pattern supports storage-exhaustion fallout;
it does not demonstrate 33 independent algorithm defects.

The real-model `test_installed_resume_probe_with_qualified_model_dependencies`
passed in 4.051 seconds; its tiny CPU wiring counterpart also passed. Both image
inspections passed. `hip.preflight` is BLOCKED by `cpu.semantic`; there is no actual
HIP-test result or feature-acceptance result from this run.

The canonical bigint CPU/GPU and retirement launchers explicitly selected `/tmp`
for pytest, bypassing their configured disk-backed TMPDIR. The engine bounds this
tmpfs to 2 GiB. T110 sends every validation suite's test scratch to
`/work/output/.runtime/pytest` under the existing private writable output mount.
Pytest releases passed-test fixtures and retains failed ones. Regular scratch files
remain charged to disk/checkpoint limits, while private aliases are never followed
or exported and scratch is excluded from command evidence inventories. Public
symlinks and symlink runtime roots still fail closed. No tmpfs or storage cap was
raised and no test was removed. Fresh images/inputs and a single acceptance-013 run
are required; no standalone HIP probe is needed.

## Operator acceptance-013 — paired updates pass; evaluation process times out

Inspected 2026-10-31 Pacific. Archive `acceptance-013.zip`:1,497,064 bytes,
465 inventoried payloads plus manifest, SHA-256
`7e7f91a9cf77bb29a71794020a29139f1c5c3ec5f449d29d1f3bd25c8ce72554`.
Both basic inventory/hash verification and independent canonical
`generalization_archive.verify_archive` reconciliation pass. The latter returns
integrity PASS, software FAIL, scientific UNMEASURED, feature_qualified false.
Final workflow/summary/terminal bindings also verify. Report and verifier exit1
correctly encode the verified negative scope; they are not infrastructure errors.

Original stdout:1,181 CPU/native tests and77 subtests passed (8 deselected),
27 HIP tests passed. Original JUnit:1,258 CPU/native records (383.158 seconds),27 HIP records
(351.942 seconds), zero failures/errors/skips. Duplicated diagnostic copies are
not counted as additional runs. Actual HIP device evidence records finite forward
and backward, changed parameters,2 updates and `resume_equal: true`; resume checkpoint
`sha256:5c31560a3618dd7f6993e8d66c748b478ee5f1c60b570d3bc52fb02233a5d360`.
This proves the bounded installed static resume probe, not the still-incomplete
retained-model admitted-replay acceptance comparison.

Workflow cumulative seconds:preflight130.702, qualification1132.888,
prepare1173.074, retained-eval1287.145, refresh1306.888,
parameter-targets1341.025, replay-resume1728.953, report1731.031,
verification1732.299. The three earlier acceptance cases passed. Replay admission
and both paired learners progressed:continued-SFT process109.889 seconds,
verified-replay189.980 seconds, each exit0 / PASS / COMPLETE, acknowledged cleanup,
and three logged updates. Final checkpoint writes reached roughly57 seconds.

The first continued-SFT post-training generator wrote a PASS application result,
but its process was killed with returncode-9 after71.996 seconds; supervision
records TimeoutError. Cleanup was acknowledged in0.329 seconds, inside17.939 seconds.
The outer case allowance was not exhausted. Source inspection and virtual-clock
counterexamples show that model loading reset the internal clock and the internal
allowance left no explicit process-startup/exit reserve. A PASS application result
cannot overrule the failed process receipt. Full-state paired/resume proofs,
paused/resumed retained learners and failure-packaging case were not reached.

T111/T112 address these boundaries together, retaining original evidence and all
scientific/proof/storage gates. See [operational repairs](operational-repairs.md)
and the updated [one-command guide](../quickstart.md). One fresh acceptance-014
session is the next hardware gate; no separate probe or diagnostic export is needed.

## Operator acceptance-014 — small-model checkpoint persistence deadline

Recorded 2026-10-31 Pacific. Original archive:1,322,612 bytes,369 inventoried
payloads plus manifest; SHA-256
`4b1deb4d532c17fcf045915dc31ec8dd25a6a28fe1bc7eca85ed06686bd77598`.
Basic inventory/hash verification and independent `verify_archive` reconciliation
both pass; final workflow and terminal verifier evidence bind a valid negative
scope. Report/verifier exit1 encodes FAIL, not a packaging infrastructure failure.

CPU JUnit:1,262 records in380.253 seconds; HIP:27 records in369.546 seconds,
zero failures/errors/skips. Duplicate diagnostic copies are not additional runs.
Workflow cumulative seconds:preflight130.199, qualify1248.433, prepare1288.368,
retained-eval1403.240, refresh1462.879, parameter-targets1520.109,
report1530.082, verify1531.418. Replay/resume and failure-packaging were unstarted.

The literal learner completed with PASS/COMPLETE, exit0 in23.871 seconds. The
parameterized learner loaded its small model, completed an update, saved a
checkpoint, then began another checkpoint save. Its process timed out after25.247
seconds with returncode unavailable and unacknowledged client termination. Engine
cleanup subsequently succeeded in4.029 seconds within its6-second allowance.
The envelope was30 seconds with approximately24 seconds of process work. The
write already in progress could not finish before that external deadline. This
establishes a persistence timeout, not a semantic failure of parameterized targets.

T113 gives both small-model learner cases180 seconds per learner and180 per proof;
both case reservations become25 minutes. It accounts for both proofs during
preflight. The total ceiling becomes395 minutes; scientific work/storage limits
and research budgets remain fixed. Full replay/resume remains pending. The next
consolidated session uses acceptance-015 paths, as in the quickstart.

## Acceptance-015 and temporary diagnostic iteration

Archive SHA256: `22a98409a60e28220cffde4ae41b173c159fd1ec91c74825e77839af9253c6b4`.
247 ZIP members. Independent canonical verification: integrity PASS, software
FAIL, scientific UNMEASURED, feature_qualified false. Only retained-eval ran as
a feature case; refresh, parameter-targets, replay-resume and failure-packaging
are missing, not failed measurements.

Underlying stdout records1187 CPU tests and77 subtests passed in379.46s;
27 HIP tests passed in360.33s. Cumulative workflow times:preflight149.993s,
qualification1443.184s, preparation1483.071s, retained-eval1594.649s,
report1596.048s, verification1596.967s. Roughly24 of26.6 minutes preceded
preparation. Duplicate diagnostic copies do not count as additional tests.

The e14604b9 checkpoint's generator reported FAIL/NonQuiescentWorker:
"nonquiescent reference/native task; terminate role via external watchdog".
Its process exited0 after60.237s, so the outer process timeout was not the failure.
The82db144b checkpoint generator passed and exited0 after46.646s. The scheduler
raises this exception when a worker remains alive after its shared five-second
drain. The archive does not identify the underlying operation or prove the cause.
T113's small-model learner/proof changes were not exercised by this run.

T114 adds a temporary, independently hashed helper and tests beside this note.
No production source, qualification-inventory rule, image or canonical acceptance
contract changes. A local comparison confirms the production inventory still
equals acceptance-015's original `payloads/source.json` byte-hash mapping. Real
current-device/image/source compatibility remains checked by the canonical
qualification consumer before probes. The helper supports focused reproduction,
tiny installed native/HIP paths and independent canonical cases in one archive.
It does not fix or claim to explain the nonquiescent worker. Its traces are the
next evidence needed; final T064–T067 gates remain open.

Local helper/affected-adapter validation:76 tests passed in3.60s (23 helper tests
plus53 existing acceptance, scheduler, container and evaluation tests). This
includes real host subprocess supervision and command-construction checks for
both engines; it is not a Docker/Podman/native/HIP execution result. Tests observed
and repaired invalid single-file mounting and a malformed native contract hash
before operator execution. No CI completion was awaited.

## Diagnostic-001: final resume comparison reached

ZIP SHA256: `1d5dc65ecbe0737fa77c5fda5052f9fdada191bc63936eaeddc62dea7243be33`;
375 members. The archive's mandatory inventory and hashes independently verify.
Its summary FAIL is the diagnostic outcome, not an integrity failure. No canonical
feature qualification is claimed. Stage durations total726.993 seconds (12.1min).

| Selected operation | Seconds | Observed outcome |
| --- | ---: | --- |
| Original qualification compatibility | 2.57 | PASS |
| Native admission/restore probe | 1.56 | PASS;1 admitted, exact restore |
| Small HIP full-state resume | 8.95 | PASS;2 updates, exact resume |
| Traced retained reproduction | 108.86 | PASS |
| Refresh | 19.19 | PASS |
| Parameter targets | 19.05 | PASS |
| Failure packaging | 3.37 | PASS |
| Canonical retained evaluation | 112.96 | PASS |
| Replay/resume | 450.49 | FAIL at exact full-state comparison |

Underlying role controls show paired learners PASS in66.89/85.41s; post-training
generators PASS in69.67/65.22s with successful selection roles; paired finite-update
proof PASS; intentional paused learner PARTIAL; resumed learner PASS in93.17s.
The resume-proof role fails with `resumed state differs from uninterrupted state`
after6.21s. No unacknowledged cleanup was reported. This exercises the earlier
timeout repairs through evaluation and reaches the final state comparison.

The previous NonQuiescentWorker failure did not recur in either retained evaluation.
That does not establish it is fixed. Trace logs are bounded excerpts, so unmatched
start/end counts cannot establish a missing operation. The uploaded archive lacks
the private checkpoint components needed to locate the new mismatch. Do not infer
numerical nondeterminism, a replay cursor defect or a harmless RNG difference from
the generic error, and do not relax exact comparison.

T115 supplies a temporary read-only CPU inspector for the existing local states.
It reports per-component equality and bounded tensor statistics without exporting
private values or modifying production code.32 local tests pass, including real
serialized-state comparison, corruption rejection, scoped path discovery and
Docker/Podman command construction. No actual container/HIP work ran locally;
the production source inventory still exactly matches acceptance-015. Final
T064–T067 gates remain open, and a production repair awaits the inspection result.

## Resume-inspection-001: numerical state differs, logical state matches

ZIP SHA256: `d5c51800b0e985e95d54d33d115c7103f3347502ed67b6f9a447e23e40d62a91`.
Independent archive inventory/hash verification passed. CPU inspection completed
in12.166 seconds, exit0. Inspection PASS means successful measurement;
`comparison_equal=false` retains the failed resume comparison.

The three saved learners have identical compatibility IDs. Full and resumed runs
each completed3 updates, consumed456 of512 target tokens and retain the identical
next batch of151 tokens, which exceeds the remaining budget (`debited=false`).
The paused run completed1 update and consumed154 tokens. Initial weights match
between full/paused and full/resumed. Final RNG, data (including pending batch),
replay and token-ledger states all compare exactly equal.

232 weight leaves and482 optimizer leaves differ. The first128 exported details
per component contain weight absolute differences up to1.7142119759e-5 and
optimizer differences up to6.6938810050e-9. These are maxima of exported detail,
not global maxima:104/354 details respectively were omitted. Do not infer a
tolerance or a backend cause from those incomplete magnitude summaries.

Code audit confirms `models()` seeds Python/NumPy/Torch and enables deterministic
algorithms; the retained-model learner also uses its scoped FP32 math policy.
Therefore merely enabling the already-active deterministic flag is not a fix.
The evidence narrows divergence to numerical training state but does not yet
distinguish independent-run variation before pause from variation after restore.

T116 extends the temporary inspector to compare all retained full-run snapshots
at the paused update count against the saved paused numerical/RNG state. It also
aggregates statistics across omitted tensor detail rows. This uses existing
read-only states and no HIP work.35 local helper tests pass, including boundary
match/mismatch/unavailable cases. No production code, equality rule, image or
qualification source inventory changed; T064–T067 remain open.

## Resume-inspection-002: missing historical pause boundary

ZIP SHA256: `d56a4b8f716a07d9b887ccadbced739ac2c60972a86505aa1d868d1a4b9910e4`.
Independent archive verification passed; CPU process completed in10.978s.
The pause-boundary report is explicitly unavailable: no retained full-run state
has the paused run's completed-update count1. This cannot establish whether the
divergence began before or after resumption. No further read-only inspection can
recreate that discarded boundary.

Complete aggregate statistics (including omitted detail rows): weights differ in
232 leaves/29,733,134 elements, maximum finite absolute difference1.7142119759e-5;
optimizer differs in482 leaves/206,741,247 elements, maximum6.2864273787e-8.
Neither comparison has nonfinite tensor pairs. RNG/data/replay/ledger remain
exactly equal. The optimizer global maximum supersedes the earlier exported-detail
maximum; no acceptance tolerance is inferred from either statistic.

T117 implements one focused same-paused-checkpoint continuation matrix. Independent
A/B continuations test repeatability; an additional update2 checkpoint/restart tests
the restart boundary. It reuses the installed learner, frozen data/pool/initializer,
original qualification compatibility and exact comparisons. All available new and
original final-state pairs are compared together on CPU. No production source,
backend setting or equality rule is changed.48 local helper tests pass; no local
HIP execution occurred and CI was not awaited. Final feature gates remain open.


## Resume-matrix-001: repeat continuations and extra restart agree

Operator upload reviewed2026-11-01 Pacific. ZIP SHA256:
`e9cf84fdfb2f3a1c84d9de2cdc29e9eb235535e112aa93ee7ae67f911d578afc`.
Independent inventory/hash verification passed for78 members with no omissions.
All four helper/test hashes match the committed helper versions at63cfc0e.
Compatibility, four training calls and CPU comparison all completed successfully,
with no unacknowledged cleanup; summed stage elapsed219.628 seconds (3m40s).

| Exact endpoint comparison | Outcome |
| --- | --- |
| Continuation A versus B | Equal |
| A versus extra-restart continuation | Equal |
| A/B/extra-restart versus earlier resumed endpoint | Equal |
| Every resumed endpoint versus original uninterrupted endpoint | Unequal |

All endpoints have the same compatibility identity. Equal means all six compared
components: weights, optimizer, RNG, data, replay and ledger. Only weights and
optimizer differ against the original endpoint. `status=PASS` means measurements
completed; `feature_qualified=false` and scientific UNMEASURED remain accurate.
The generic result projection omitted branch starting-checkpoint IDs from the
archive; unchanged verified helper hashes establish the executed routing, but the
archive does not directly supply those omitted IDs. T118 preserves them explicitly.

These observations show repeatability for the tested saved-state continuations,
including a restart at update2. They do not establish general backend determinism,
locate the original divergence, or clear the original resume gate. Audit of the
production routes found the same frozen configuration, seeded model construction,
initializer, replay pool, exact state restore and scoped FP32 math policy; no
causal code defect was demonstrated. Do not change optimizer or backend settings
speculatively, and do not loosen equality.

T118 extends the disposable helper with `--start fresh`: two fresh complete runs,
a fresh update1 pause and its continuation, all final pair comparisons, full
initial-state comparisons (not just weights), and old/new update1 state comparison.
This is nine updates in one bounded command using the existing canonical roles,
images and qualification consumer. Comparison loads extra boundary pairs serially
to limit memory and verifies checkpoint closures before deserialization. Archive
provenance includes checkpoint/initializer hashes but excludes private tensors.
57 local tests pass; five expected pre-implementation failures were observed.
The production source inventory has no changed files. No local HIP execution or
CI wait occurred; final T064–T067 gates remain open.


## Resume-fresh-001: fresh resume passes; equal pause values do not explain history

Reviewed2026-11-01 Pacific. Archive SHA256:
`8d57e7bbfe4f38e6d0a55f35b38aaa8813963e0ab2197fb503932d8043a8861a`.
All78 inventoried payloads plus manifest verified (79 ZIP entries), no omissions.
All four helper/test hashes match dd9ae26. Stage elapsed total176.265 seconds;
all six stages PASS/exit0 with acknowledged cleanup. Feature qualification stays
false and science UNMEASURED.

| Comparison | Exact outcome |
| --- | --- |
| Fresh A versus fresh B | Equal |
| Fresh A/B versus fresh pause/resume | Equal |
| All three new final states versus original uninterrupted | Equal |
| New versus old one-update pause | Equal |
| All four initial-state comparison pairs | Equal |
| Every new final state versus old resumed | Unequal weights/optimizer only |

The equality checks include weights, optimizer, RNG, data, replay and ledger, with
zero missing boundary comparisons. All fresh branches use initializer
`sha256:a7aad279a0b51fc31cb6a2a8285a88b39b153feda6d4d9b35ed60a76fa5ce10b`.
The fresh pause checkpoint is
`sha256:c803d8b4526f03ca45906a3592fd8f5dc097b2b71405c1a3c9336e02928d0dde`;
the split-rest branch explicitly starts from that exact pointer. All endpoints
retain the original compatibility identity. Old-resumed differences retain the
previous232 weight leaves/482 optimizer leaves and previous global maxima.

This corrects any inference that the original uninterrupted result was the lone
nonreproducible path: three new paths now reproduce it. Conversely, repeated
continuations of the old paused state previously reproduced the old resumed result.
The two paused states compare equal in values. A deterministic failure inherent in
every first restart is not demonstrated. Nor is the historical mismatch repaired.
The current comparison ignores tensor layout/storage aliasing and dictionary
insertion order; it also excludes elapsed time and provenance state by design.
These are remaining observable differences to inspect, not established causes.

T119 extends the disposable CPU inspector with optional `--matrix`, binding the
fresh matrix to the original source diagnostic, frozen images and successful
one-update checkpoint. It reports exact values, CPU-loaded representation metadata,
serialized component hashes and excluded-state differences; bounded layout details
contain no tensor contents, raw program keys or memory addresses. Historical GPU
allocations/backend state cannot be recovered by this inspection. No production
source, equality rule, backend policy or qualification inventory changed.
64 local tests pass after four intended failing tests; no local HIP or CI wait.
Final T064–T067 gates remain open. Next operator action needs only CPU access to
already saved files, not another training run.


## Resume-layout-001: identical serialized numerical state

Reviewed2026-11-01 Pacific. ZIP SHA256:
`c6545602993bd61ed1481c9bc3215521f659733f06fa9a4bbb3392cefe88c249`.
Independent verification passed for7 payloads plus manifest, no omissions.
CPU inspection completed15.233s, exit0. Historical final comparison stays false.

Old and new paused checkpoints have identical weights, optimizer, RNG, data,
replay and ledger values. Tensor layouts, storage aliasing, dictionary order and
recorded types match, with no omitted layout changes. The serialized weights,
optimizer, replay, all four RNG files and selection state have identical SHA256
identities. The only observed excluded differences are scheduler learner_seconds,
history exposure keys and checkpoint pins. These are expected elapsed-time and
provenance distinctions; they do not establish a cause of numerical divergence.
The source audit confirms these fields control budgets/retention, not learning-rate
or optimizer mathematics; both compared terminal runs performed the same updates.

The user's later-fix hypothesis was checked before further diagnostic changes:
`git diff 5ac9d72..37875b4 -- src scripts tests Dockerfile pyproject.toml uv.lock`
has no production differences. Intervening commits changed only temporary helper
code/tests and feature documentation. Fresh-001 and the existing diagnostic use
GPU image `sha256:1afe89cf6e61b3d311e3b1d4e71b2d5ff15e5571e5ef22ac54ed149dc7602069`.
No subsequent committed learner fix can explain the differing outcomes under that
unchanged installed image. Process/runtime conditions remain unobserved candidates,
not a proven backend defect. A successful fresh resume does not erase the failure.

T120 adds a temporary paired diagnostic using the same canonical learner and
qualification consumer: old/new uninstrumented controls and old/new instrumented
continuations, eight total updates. The observer hashes pre-update inputs/state,
forward logits, raw/clipped gradients and post-update state, and records actual
execution facts. Hashing reads tensors back to CPU and synchronizes the device;
control-versus-trace endpoint comparisons explicitly measure this intervention.
The first differing observed phase narrows the boundary without diagnosing a kernel.
The helper makes no algorithm, backend-policy, tolerance or installed-source change.
All available final endpoints are compared on CPU and partial traces survive failed
learners or final comparison.77 local tests pass, including an actual CPU optimizer
update/RNG preservation test, both engine constructors and failure packaging.
No local HIP or CI wait. One operator run is required for the new measurements;
T064–T067 remain open.
