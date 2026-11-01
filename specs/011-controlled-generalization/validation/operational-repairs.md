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
