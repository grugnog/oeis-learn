# Phase E closeout — acceptance verified; one hardware proof blocked

Recorded 2026-11-05 against the operator's `report(10).zip`. This closeout completes
all evidence work available from the uploaded acceptance-only run. It does not
claim every Phase E task passed or that optional research ran.

## Task dispositions

| Task | Disposition | Evidence |
| --- | --- | --- |
| T067 | COMPLETE | [Frozen scope, input inventory and finite graph](operator-handoff.md) |
| T068 | COMPLETE | [Original four-stage workstation PASS](hip-acceptance.md) |
| T069 | COMPLETE for amended v2 | [128 tasks/23 families, all exclusions, zero deficits](structural-admission.md) |
| T070 | COMPLETE for acceptance scope | [Finite HIP updates, paired targets and loaded checkpoint](hip-acceptance.md#t070-finite-hip-updates-paired-targets-and-loaded-checkpoint) |
| T071 | COMPLETE for acceptance scope | [Actual reward/review path, derived probability and measured costs](hip-acceptance.md#t071-actual-rewardreview-path-and-measured-cost) |
| T072 | BLOCKED / unchecked | [Clean HIP resume verified; actual HIP crash debit absent](hip-acceptance.md#t072-verified-clean-resumeexport-actual-hip-crash-debit-blocked) |
| T073 | DEFERRED / unchecked | [No authorized completed research to analyze](research-results.md) |
| T074 | COMPLETE | This reconciliation, task markers and reliability review |

Original A–D: 66/66 complete. Phase E: **6/8 checked**, one hardware-blocked and
one optional research task deferred. All twelve follow-up repair tasks remain
complete. No historical 011 T174 or T177–T181 marker changed.

## What the PASS establishes

The independently verified original archive reports integrity PASS, software PASS,
feature qualified, scientific UNMEASURED. All four stages completed with exit 0
and acknowledged in-bound cleanup. Qualification has 1,499 CPU and 27 HIP tests,
with no failures/errors/skips. Production calibration uses 64 real updates and
actual save/load. Reduced HIP acceptance uses 16 updates, six valid rewards, two
graded anchors, two exact clean-pause comparisons and a reloaded final forward.
The existing runtime acceptance flag is preserved for its exact source/image pair.

The bounded [machine record](evidence/acceptance-004-record.json) contains original
identities and derived counters, with explicit missing measurements. Source and
helper inventories match the original producer at closeout: no runtime, tests,
locks, contract or orchestration change invalidates qualification. These are
source-bound producer attestations and independently checked public artifacts;
private tensors, hidden truth and omitted raw observations were not re-read.

## What is still missing

T072 explicitly demands actual device/failure proof, including discarded crash
work. The original HIP receipt exercises clean pauses; its negative ZIP exercises
synthetic failure export. The real crash-debit test in qualification uses CPU.
None is an actual unsaved-segment HIP crash/recovery receipt. No compatible engine,
HIP device or private run checkpoints are available in the implementation session.
Only an actual source/image-bound HIP exercise can close that gap. All other
available evidence reviews have been completed; no paperwork can substitute for it.

T073 is separately optional and remains deferred: `research=false` authorized no
full B/C learners, baseline scoring, final structural comparison or twelve-case
research analysis. A remains reference-only, no fresh A/D or extra seed is added,
parameter evaluation stays DEFERRED and assessment NOT_EXECUTED. The 65/68 primary
completion floor still applies to a future comparison, not to this smoke test.
No model superiority, speedup, retention or scientific winner is claimed.

The sampled 11.64-hour per-arm update-plus-diagnostic projection and 66.47-hour
learner ceiling are planning figures. Generation/preparation forecast gaps, warm
row-access measurements, unexported peak disk figures and unexported raw event
values remain explicit. Limited calibration is not a worst-case guarantee.

## Failure/retry and publication boundary

The original quota failure, `-002` missing custodian artifact, and `-003` missing
calibration row field remain failed for their original closures. The authorized
v2 amendment changes only the disclosed population; all original draws/exclusions
remain. Production repairs required the successive source-bound plans/images.
`-004` itself needs no acceptance retry. Reuse only complete compatible receipts;
use read-only recovery for reporting and never mutate originals or restore a newer
policy journal onto older weights. A future new proof/code scope must qualify its
affected dependencies, not inherit PASS by filename.

Automatic approval review rejected uploading the entire 453,255-byte original
ZIP to GitHub because raw logs/source metadata may be sensitive. The safer published
alternative contains bounded derived findings and content hashes only; no raw ZIP,
logs, full source inventory, tensors, protected values or host paths are included.
The complete original remains with the operator. This publication limit does not
prevent the public-reference closeout required by T074.

## Validation

The complete standard-library [audit and invocation](phase-e-audit.md) are retained
for reproduction with the original operator ZIP; no ephemeral script is required.

- Independent original archive verification under `PYTHONPATH=src python3 -S`
  reproduces the original verifier identity and all status fields.
- Original archive SHA/size, source inventory, unchanged helper identity, all
  exclusion cells, exact category/family quotas, XML counts, both clean-resume
  proofs, loaded checkpoint, paired-target totals, calibration forecasts, stage
  ceilings and current task counts reconcile against the bounded machine record.
- Altered member bytes, missing terminal and corrupted nested failure ZIP are
  rejected by the existing verifier; the original negative ZIP independently
  verifies FAIL with private members excluded.
- New document links, read-only checklist (16/16), task prerequisites and
  `git diff --check` pass. No extension hooks are registered.

The unchanged code already has the recorded 664-test/2-subtest local selection and
this exact original workstation qualification; documentation changes do not justify
repeating training or a broad runtime suite. No new HIP command, research run,
merge or CI wait was performed. The evidence-only changes are committed per task
and published to PR 63 against main.
