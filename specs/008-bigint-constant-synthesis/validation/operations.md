# Phase 5 operations and exports — T097

Status: targeted new-image check/GPU/performance commands and their bounded ZIP
are reviewed in [the scoped receipt](phase-5-scoped-hip-review.json). Full
current-scope command/export coverage remains pending. T093-dependent historical
checkpoint evaluation is user-deferred2026-10-22.
The completed negative diverse32 run and pack export are now reviewed in
[the learning receipt](phase-5-diverse32-review.json):15 outer archive files,
16 nested bundle files, no manifest omissions and complete core export.

Completed workflow39686633400 now supplies both-engine public validation and
verified-rerun evidence. Each public CLI suite passes3/3 with zero skips; each
host-role suite passes29/29. Broad public CPU validation contains1,115 cases
with nine explicit hardware/legacy skips and no failures/errors; those skips
are not promoted to mandatory qualification passes. The independent
arbitrary-UID packaging/preparation suites pass202+44+27 cases without skips,
and the installed `python -I` retirement audit passes with no violations. All
archive and receipt-file hashes verify in [the execution review](phase-5-execution-review.json).

This completes those portions of T097, not every supported quickstart/export
operation. Historical matched-checkpoint diagnosis is now user-deferred and
does not block T097. A complete current command/export checklist remains open.
Successful
containment inspect/cleanup artifacts are now reviewed
in [the containment receipt](phase-5-containment-review.json). No fixture substitutes
for those missing operations and T097 stays open.

The containment suite exercises complete checkpoint/resume/no-refund, hidden
mount rejection, final-plan access controls, normal/interrupted summaries and
bundle-cap pressure. Installed retirement is rerun by the host helper using
`validate --suite bigint-retirement`; installed audits execute with `python -I`
outside the checkout. Historical readers remain read-only and unreplayed.

The preflight helper runs public prepare/check, GPU validation and optionally
actual matched-checkpoint diagnosis. Learning is a separate reviewed-receipt
step through `scripts/run_bigint_diverse32.py`, which invokes public
check/learnability/pack without repeating measurement. Its final ZIP
is created on success or failure, reserves the root summary first, hashes each
included/omitted report and excludes weights, runtime caches, prepared/private
payloads and scorer inputs. A successful tiny run also uses the supported pack
command and includes that bounded bundle plus core status/checkpoint/summary
records. Reports/inputs/caches stay in the workspace. Use a new output suffix
for each measurement retry. Learning has a durable single-attempt claim;
an interrupted attempt must resume its original run's identical image/recipe
and remaining ledger. The helper does not manufacture a fresh allowance.

PR #59 adds host-only regression coverage for receipt tampering, stale
source/image/prepared shape, cross-scope admission, changed schedule, duplicate
attempt claims, negative learning outcomes and interrupted commands. The last
two tests mock the external engine and verify durable claims, public command
ordering and archive creation; they establish orchestration behavior, not HIP
or learned accuracy. Standard-library-only `python3 -S ... --help` works for
both host helpers. The reviewed current-source measurement ZIP contains24
verified files, no omissions and successful cleanup in0.056s against5s allowed.
Only the performance output directory is mounted into its offline/read-only
measurement role. These observations do not replace the complete containment
suite or actual learning/resume/pack evidence.

Full quickstart research/final-scoring commands remain Phase 6 gates; they are
not launched merely to check off operations. See [phase-5.md](phase-5.md).

The observation-only `--runtime-profile-only` command adds a targeted CPU
operation with600seconds work, separate5s cleanup and a fresh bounded archive.
Unit tests verify scope rejection, source mismatch, failed stage handling and
archive creation. Its actual new-image execution remains pending and is not
counted as complete T097 command coverage.

The uploaded CPU profiling operation is now verified:75 rows/225 pairs, seven
manifest entries plus manifest, no omissions, offline/read-only/output-only
inspection and0.040418s cleanup; see [the receipt](phase-5-runtime-profile-review.json).
The new `--compare-host-lookups` extension adds a bounded ordinary baseline to
the same role. Actual execution of that new source remains pending; it does not
complete the remaining T097 command/checkpoint/export coverage.

The uploaded lookup-comparison operation is also verified in
[its review receipt](phase-5-host-lookups-review.json):75 rows/225 three-way
groups, no manifest omissions and0.047662s cleanup. `--profile-input-load` now
adds nested observation to the same600s CPU role. New-image execution remains
pending and does not close the remaining T097 command/checkpoint/export coverage.


The [current Podman CPU adoption operation](phase-5-engine-adoption-review.json)
passes all three test groups and the75-row comparison, with verified archive,
identity, output-only isolation and cleanup. Its earlier fixture-identity
failure is preserved separately. This closes that new operation's pending
Podman evidence; complete command/export coverage, retained-checkpoint diagnosis
and eventual-image Docker/HIP evidence remain separately tracked. T097 stays open.


Current scope decision2026-10-22: T093-dependent historical-model execution is
excluded from T097 completion; retained reader and compatibility regressions
remain required. Prior receipts and historical pending lists are unchanged.


## Whole-pipeline diagnostic implementation — 2026-10-22

The [profile plan](phase-5-pipeline-profile-plan.md) and [quickstart](../quickstart.md)
document the new stdlib-only host harness. Its parent is read-only, selected
weights use a separate child identity, generation never receives synthetic truth,
and scoring receives only sealed selections and synthetic training truth. Native
requests remain serial. Fresh outputs, per-role external deadlines, log/disk
watchdogs, actual isolation/cleanup audits and bounded report-only archives
retain partial failures. Scope flags for training, resume and final evaluation
are absent/rejected. Existing detailed WASM tools remain intact and disabled for
this diagnostic. Unit tests exercise restoration, identity rejection, missing
sensors, partial work, collector/hook cleanup, audit rejection, watchdog expiry
and archive exclusions. Actual container/HIP execution is still pending W005;
these tests are not device or T097 execution evidence.


### First-profile harness corrections and rerun

The user permits repeating the bounded profile. Trace export now records a
scheduled steady update/eight decode steps, retains capped partial files and
requires actual GPU kernel events. Rootless Podman device metadata is checked
against actual role-visible character devices and host major/minor identities;
CPU roles must expose no accelerator nodes. A preflight validates both audit
and representative model trace export before the paired benchmark. The original
failed archive remains unchanged. The [profile correction tasks](phase-5-pipeline-profile-plan.md#harness-correction-and-approved-rerun--2026-10-22)
record the exact CPU cap reproduction and unchanged isolation/runtime bounds.
Actual corrected container/HIP execution remains pending W005. The quickstart
now gives a fresh full-run command; a PARTIAL portfolio is retained honestly.

### Corrected actual HIP profile reviewed — 2026-10-23

The [280-entry verified rerun](phase-5-pipeline-profile-rerun-review.json)
supersedes the W005 pending state above. Preflight/main GPU and CPU scorer audits
pass actual device-node, offline/read-only and exact input/output-bind checks.
CPU scoring completes and all five kernel traces export within existing caps.
Cleanup is0.031–0.049s against5s; weights/private truth/runtime caches are absent
from the report ZIP, with no archive omissions. The overall PARTIAL status
faithfully retains16 unstarted/15 incomplete started generations per arm at the
unchanged limits. No further identical full run is needed.

The host's first-role gap is localized to244.21s outside the preflight container
lifetime; subsequent GPU/scorer gaps are0.26s/0.21s. The specific engine cause is
unresolved. T097 gains this command/audit/export evidence but its broader
current-scope interruption/bundle coverage remains open. T093 stays deferred,
T094/T098 remain open and no qualification/training admission follows.
