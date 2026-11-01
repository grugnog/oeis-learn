# Operational audit repairs

## T106/T108 — accounting and single-copy checkpoint graph (2026-10-31)

Reopened Phase B: clock, cleanup partition and peak-live-artifact policies were repaired together with their reference tests, before finalization/runtime integration. Learners recheck time after saves and freeze scheduler accounting after tensor serialization. Model setup, initial persistence, channel copying and selection publication consume their enclosing allowance. A stopped pending update retains its debit/batch for exact resume. Role work reserves cleanup inside its deadline; even a SIGKILL-resistant client gets at most one second of reap wait, preserving engine-removal time.

Preflight counts paired plus paused/resumed learners, live resume projections, atomic replacement, proof inputs and weight copies. Direct verified channel projection removes duplicate private input trees. Proofs copy only latest full state and initial weights. Acceptance replay policy evaluation omits unused selected-state/initializer publication; research behavior is unchanged. A metadata-only retained-model calculation reserves16 full-state units plus10 weight units, within the existing26GiB P4 cap. No large test tensors or hardware allocation were needed.

Acceptance-011 is now available: archive SHA-256 `8d75032bf8289056a96084b63ae84790fe2d73fcce34ab8da1f5d71449c86d61`,372 members, inventory/hash verification passed. The continued-SFT process completed in122.79 seconds. Verified replay progressed through updates, then expired during a checkpoint save at180 seconds; its process/cleanup observation took185 seconds and recorded unacknowledged process termination, leaving no engine-removal allowance. Last completed saves took roughly26 and45 seconds. Admission/restore passed the earlier010 boundary; full replay/resume remains unqualified. This is actual operator evidence, not a new local HIP run.

The generated acceptance envelope now permits600 seconds per replay learner,180 seconds per replay proof,70 minutes for the replay case, and335 minutes overall including packaging/margin. Existing token/model/seed limits and research recipes are unchanged. These are reservations, not predicted runtime or a hardware-readiness claim.

Validation:85 local tests passed across operational budgets, training, runtime, replay, supervision, selection I/O, acceptance/setup and recovery. Includes delayed save with resumed debit, clock serialization order, direct copy counts/identity rejection, exact-fit/over-cap metadata and an unreapable client. No HIP command or CI wait.

## T104 — final workflow evidence (2026-10-31)

The finalization policy validates the canonical sealed verifier summary, archive bytes/hash, scope and exit code together. Valid negative-scope verification remains usable; missing/truncated/contradictory results trigger a verified recovery bundle. Final publication preserves every original report member byte-for-byte and adds the complete controller ledger, verifier summary and original member manifest. Independent archive verification checks final status/order and preservation of original payloads. Interrupted publication retains an INTERRUPTED ledger/export. Original report/qualification receipts are never rewritten.

Validation:38 tests passed across serial/canonical acceptance, recovery and archive boundaries, including nine finalization outcomes and tampered final metadata. No HIP command. This completes the reopened operational-policy work; subsequent tasks wire bounded diagnostics, runtime environment and consolidated coverage.

## T105 — complete bounded control capture (2026-10-31)

Diagnostics now inventory outer stage process/disk controls, fresh qualification image/device inspections, authority validation, verifier inspection and every discovered role disk record. Numeric/status controls precede bulky logs, are allowlist-projected without argv/host paths, and retain original hash/size. Missing, malformed, oversized and unsafe sources remain explicit inventory outcomes. Normal report, recovery and read-only export paths share the collector.

Validation:63 diagnostics/canonical acceptance tests passed, including seven previously omitted boundary paths across all three export modes. Existing original qualification/measurement payloads remain unchanged.

## T107 — qualified private runtime environment (2026-10-31)

Production roles now use the same private scratch/cache and thread settings as qualification, mapped through their actual `/output/<role>` mount. The helper retains its historical `/work/output` default for existing callers, rejects escaping destinations/symlinks, and creates private0700 directories. The2GiB tmpfs ceiling matches qualification and remains within the existing container memory limit; no shared cache or extra host mount is introduced. Source closure changes invalidate old qualification reuse.

Validation:87 tests passed across real Docker/Podman command construction, container validation and diagnostic subprocesses. Eight role/engine combinations check destination mapping, permissions, read-only inputs, network/device confinement and thread controls. This establishes command contracts, not actual engine/HIP qualification.

## T104 follow-up — verifier terminal boundary (2026-11-01 UTC)

Combined review found that a valid negative-scope verifier and a packaging error
could both leave a FAIL summary and exit1. The canonical endpoint now publishes
a sealed completion record only after packaging succeeds. Finalization requires
that record to bind the archive, exact summary bytes and exit code, and includes
it in the final export. New regressions inject failure after summary publication
and verify that missing terminal evidence routes to recovery. The combined local
selection passed412 tests, with6 subtests, one dependency skip and one HIP test
deselected; detailed consolidated scope is recorded below.

## T109 — combined boundary regressions and installed resume coverage (2026-11-01 UTC)

The combined suite exercises canonical serial finalization and report/recovery
exports, Docker/Podman command construction, safe watchdog ownership, delayed
checkpoint/copy accounting and cleanup starvation. The paired/replay fixture now
injects failures during post-training evaluation, paired proof, resumed learning
and final resume proof, for both admitted replay and empty-yield paths. External
process boundaries are recorded fixture adapters; installed handlers, tensor
state and orchestration execute locally where dependencies permit. Finalization
also covers failure after a PASS archive exists, invalid summaries, negative
scope, missing verifier completion and interruption. The nested-watchdog test
asserts propagation inside the nested context so outer cleanup cannot mask it.

The existing marked HIP test previously proved weight load, finite gradients and
one changed update only. It now additionally invokes the installed generalization
learner for uninterrupted, paused and resumed two-update runs and compares actual
weights, optimizer, RNG, data and replay state through the production proof handler.
It checks actual HIP availability and nonempty device RNG. This mechanical static
probe does not replace acceptance's native admitted-replay test. The test remains
in the existing qualification selection; there is no extra manual probe or second
qualification invocation.

The same probe has a tiny real-tensor CPU wiring test. A separate real-model CPU
check runs with the pinned model/native dependencies and explicitly skips when
`wasmtime` is unavailable. The historical foundation disposition's current
implementation hash was refreshed using its canonical serializer; no historical
qualification or disposition status was changed.

Final local command (no HIP or engine execution):

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:../phase-a-venv/lib/python3.12/site-packages \
python -m pytest tests/unit/test_generalization*.py \
  tests/contract/test_generalization*.py tests/integration/test_generalization*.py \
  tests/unit/test_bootstrap_bundle.py tests/unit/test_pipeline_contracts.py \
  tests/unit/test_pipeline_bundle.py tests/unit/test_pipeline_supervision.py \
  tests/unit/test_pipeline_disk.py tests/unit/test_container_runtime.py \
  tests/unit/test_container_validation.py -q -p no:cacheprovider \
  -m 'not gpu and not docker'
```

Result: **412 passed, 1 skipped, 1 deselected, 6 subtests passed in 28.97 seconds**.
The skip is the real-model `wasmtime` dependency check; the deselection is the
actual-HIP test. `git diff --check` passed. T103–T109 implementation is complete;
T064–T067 retain fresh hardware/build/input/acceptance gates. No HIP command was
run, no research campaign started and no CI wait performed.

## T110 — qualification scratch containment (2026-10-31 Pacific)

Acceptance-012's original recovery archive verifies (74 payloads plus manifest;
SHA-256 `ff4a316cc43667a1c194150f9702319c30ef932a8ee6d80a80e251f3cd5d2736`).
Preflight passed; CPU qualification exhausted `/tmp/bigint-pytest`, causing ENOSPC
and incomplete tensor writes. The suite reports 33 failures, 1,146 passes and
77 passing subtests. HIP qualification was blocked and no feature cases ran.
The newly added real-model installed-resume CPU probe passed in 4.051 seconds.
See [HIP evidence](hip-acceptance.md) for stage timings and the original failure.

All five validation suites now share `--basetemp=/work/output/.runtime/pytest` and
`tmp_path_retention_policy=failed`. The existing two explicit mounts and 2 GiB
`/tmp` ceiling are unchanged. This routes checkpoint fixtures to private disk-backed
output and releases successful fixtures during the suite. Failed fixtures remain
locally available. Runtime scratch is excluded from command inventories and public
receipts/bundles; its regular files still count against aggregate/category limits.
Private symlinks are never traversed. Public symlinks and a replaced canonical
runtime root remain errors, while intentionally invalid fixtures nested inside
private pytest scratch do not become false outer-watchdog failures.

Ten engine/suite command regressions were red before implementation. They exercise
real command construction with recorded process/device boundaries, mount isolation,
unchanged tmpfs size, scratch accounting and evidence exclusion. A real tiny pytest
subprocess proves successful fixture cleanup and failed fixture retention. Separate
lifecycle tests cover aliases, nested invalid-runtime fixtures, authoritative link
rejection and a checkpoint-cap breach. No large tensor allocation is needed for
these routing/containment counterexamples. Foundation's current implementation hash
was refreshed; historical qualification and disposition statuses are unchanged.

Final combined local gate: **425 passed, 1 skipped, 1 deselected, 6 subtests passed**.
The skip remains the unavailable local `wasmtime` real-model dependency; the HIP
test is excluded. The uploaded pinned-container CPU resume result is recorded
separately above. `git diff --check` passed. No HIP command or CI wait occurred.
T110 is complete; T064–T067 still require fresh images/inputs and consolidated
acceptance. The next normal invocation uses fresh acceptance-013 paths and the
unchanged 335-minute ceiling; no separate manual diagnostic probe is required.

## T111 — evaluation startup/load/exit containment (2026-10-31 Pacific)

Acceptance-013's generator published a PASS application result but its process
exceeded the external work deadline and was killed. The parent correctly retained
FAIL. The installed evaluation clock previously restarted after model loading;
the role also received almost the entire process-work allowance, leaving no
explicit startup/exit reserve. The generator entrypoint now charges torch setup,
the handler charges input/native setup, and retained evaluation starts its deadline
before imports/validation/model loading and checks it before generation. Generator
evaluation reserves min(30 seconds, 20% of process work) for process startup/exit,
inside work and separate from container cleanup. A PASS result still cannot
replace successful process termination. Original 60-second portfolio policies,
attempt counts and scientific semantics remain unchanged.

Six regressions failed before the change: delayed/exhausted model loading, native
setup, torch setup and both engine command boundaries. Targeted evidence/scoring,
role-contract and runtime integration tests passed after the repair. This is the
single reopened operational-policy group; T112 wires enlarged acceptance envelopes
and records combined validation. No local HIP execution or CI wait.

## T112 — complete evaluation envelopes and original 013 reconciliation

Recorded 2026-10-31 Pacific. Acceptance-013 independently passes both archive hash
verification and canonical generalization reconciliation, including final workflow
and terminal verifier bindings. Software remains FAIL / science UNMEASURED. Its
original CPU/native suite passed1,181 tests and77 subtests; HIP passed27 tests,
including the actual two-update installed exact-resume probe. Both retained-model
paired learners completed; continued-SFT post-training evaluation was killed at
its process deadline despite a PASS result file. Cleanup and negative-scope final
publication succeeded. Full acceptance paused/resumed proofs were not reached.

Fresh acceptance inputs allow720 seconds for retained evaluation (two checkpoints)
and1200 seconds per paired post-training policy. Enclosing cases are1500 seconds
for retained evaluation and6000 for replay/resume. All other cases stay900 seconds;
learners600, proofs180, search30, model/tokens/seeds/storage and original portfolios
are unchanged. The maximum serial reservation is375 minutes (6h15), including
20 minutes packaging and10 emergency margin. The remaining replay graph fits:
4*600 +2*1200 +30 +2*180 +540 +120 =5850 <=6000 seconds. This is operational
headroom, not a runtime prediction or authorization for research.

The rest of the path was inspected through selection deadlines, proof projection,
paired/paused/resumed training and final publication. Existing combined failure
fixtures exercise evaluation, paired proof, resumed learning and resume proof;
previous hardware results do not establish those unfinished retained-model steps.
No success gate was removed and no PASS result bypasses process exit/cleanup checks.

The final affected local selection passed **429 tests and6 subtests**, with one
explicit local `wasmtime` dependency skip and one HIP deselection, in41.06 seconds.
This includes updated full worst-case reservation/protected-margin tests and all
six T111 counterexamples, which were red before repair. `git diff --check` passed.
The original archive was not modified. No local HIP command or CI wait occurred.
T111/T112 are implemented; T064–T067 require fresh evidence on changed source.
The quickstart publishes one consolidated acceptance-014 invocation.
