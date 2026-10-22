# Phase 5 operations and exports — T097

Status: targeted new-image check/GPU/performance commands and their bounded ZIP
are reviewed in [the scoped receipt](phase-5-scoped-hip-review.json). Full
both-engine and retained-checkpoint-dependent command coverage remain pending.
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
operation. Matched-checkpoint diagnosis is still blocked on actual retained
weights. A complete current command/export checklist remains open. Successful
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
