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
