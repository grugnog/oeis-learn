# Phase 5 arithmetic qualification — T089

Status: pending the new shared-image CI artifact `bigint-qualification-arithmetic-docker`.
The local independent fixture tests pass; no native execution is claimed locally.

`scripts/qualify_bigint_phase5.py --inside arithmetic` runs frozen arithmetic,
independent reference, language/codec, decoder and native signed-256 suites with
zero failed/error/skipped cases required. Its `summary.json` binds the complete
source snapshot and packaged runtime, worker, compiler and lock identities.
The measurement contract also checks successful outputs and classified failures
through the real packaged measurement executable. Resource exclusions are
reported separately from exact arithmetic disagreements; no failed result is
admitted. Existing Phase 2/3/4 evidence retains its original source binding.

Run the public host helper described in [phase-5.md](phase-5.md) to reproduce.
