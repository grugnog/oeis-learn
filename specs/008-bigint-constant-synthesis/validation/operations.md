# Phase 5 operations and exports — T097

Status: targeted new-image check/GPU/performance commands and their bounded ZIP
are reviewed in [the scoped receipt](phase-5-scoped-hip-review.json). Full
both-engine, checkpoint-dependent and learning/export execution remain pending.

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
