# First learning preflight: CPU cost-probe failure, HIP passes

Review recorded2026-10-26. [Machine review](learning-pilot-operator-20261026T124145Z.json)
verifies `009-learning-pilot-preflight-20261026T124145Z.zip`, SHA256
`8cff0e7fc93083596eb2dbd3da71ab4ecae7716d562dc5396f9e8bc7dd3e0ba0`:
183 members, zero omissions,20 sealed receipts,107 bound public files and the
exact749-entry source closure from the prior handoff (`6b89b808…`). All
receipt/file/prerequisite hashes verify. Elapsed1,347.086s, about22.5 minutes.

| Scope | Result |
| --- | --- |
| Host |220 tests pass |
| Pinned CPU |750 pass, one failure;41 additional subtests reported |
| Actual HIP |23 pass, including the new token metrics and full-state resume fixture |
| Retirement |123 pass |
| Installed workflow |1,106 pass;9 explicit skips |
| UID/export/interruption/recovery operations |PASS; healthy subsequent work |
| Independent Docker |Still unavailable: Docker command reports Podman metadata |
| Corpus preparation and30-minute session |BLOCKED before starting |
| Learning/cost/readiness |No snapshots; cost UNKNOWN; no long-run admission |

The CPU failure is
`test_actual_v3_constructor_and_public_independent_cost_sentinels` at the final
`result['complete']` assertion, after6.532s. The earlier actual104,496,662-model
count checks pass. Neither the failing cost case nor its resource reason was
included in that assertion, so the archive does **not** establish the precise
native cause. This is not an observed training, numerical-update or learned
sequence failure. No corpus or learner work from this invocation needs repeating.

## Scoped repair and its limits

All five original public probe programs reproduce their complete100-index goldens
under the independent interpreter locally. That does not qualify native execution.
Inspection identifies a plausible fixture error: the4096-bit degree4 cancellation
program must pass visible20 admission before scoring, but its100-index production
evaluation can exceed the original aggregate fuel budget. Per-term validity and a
20-index pass do not establish full-horizon resource fit. The archive omitted the
case detail needed to confirm this diagnosis; it remains an explicit inference.

`pipeline-pilot-scoring-cost/v2` now separates six required positive full100-index
paths (identity/quadratic/branch/factorial/4096-bit identity/1024-bit degree4
cancellation) from the **retained original4096-bit stress case**. Every positive
case must agree exactly in production and reference across all six scoring arms.
Stress must either succeed fully or have explicit production term/aggregate fuel
exhaustion with a complete100-index independent reference; it is then recorded as
bounded exhaustion, never full scientific completeness or a learned success.
Unknown transport/deadline/other errors and all positive mismatches still block.
The maximum observed cost includes all42 outcomes, including paid stress failures.
The report binds exact source/work/resources. Scientific scoring populations,
coverage, resource limits, clocks, learning thresholds and admission are unchanged.

Host counterexamples cover the synthetic fuel-limited case, full stress success,
unknown stress failure and positive production/reference disagreement. They use
an explicitly reference-only fake client and make no native claim. The pinned
CPU assertion now includes every returned case/outcome; visible-admission errors
also include their case/status. This removes the diagnostic blind spot even if
the actual target cause differs. Actual repaired native qualification is pending.
See [repair local evidence](learning-pilot-cost-repair-local-evidence.json).

## One corrected consolidated follow-up

All unblocked repair/code/local work is complete before this handoff. Use a fresh
output with the existing approved inputs; no manual probe or package command is
needed:

```bash
git pull --ff-only origin 009-pipeline-throughput &&
python3 scripts/foundation_run.py --engine podman --cpus 16 validate \
  --suite pipeline --phase all-ready --engines both \
  --manifest configs/foundation/pipeline_learning_pilot.json \
  --inputs tmp-validation/008-phase5-hip-20261020T202907Z/inputs \
  --max-seconds 86400 \
  --output "tmp-validation/009-learning-pilot-preflight-$(date -u +%Y%m%dT%H%M%SZ)"
```

Return one printed sibling ZIP even for failure/partial results, retaining the
original output. Build layers are cached where compatible. The conservative
whole-package source guard requires current-source prerequisites after these
changes; old23-HIP/operations successes remain accepted at their original source,
not relabeled current passes. `--reuse` cannot bypass changed closure checks.
There is no repeat size/batch/two-hour soak and no completed pilot training to redo.
The new corpus/session still need their first execution.

Prior measured qualification took22.5 minutes including builds/operations and the
failed CPU fixture; this is not an exact cost forecast for the still-unstarted
corpus/controls. The [original finite limits](learning-pilot-implementation.md)
remain:30 learner minutes plus potentially hours of preparation/full controls,
9,000s corpus,28,800s session and86,400s total ceiling. No long-run cost estimate
or learning-efficiency conclusion is inferred. L004, L012's cost clause, completed
L013 readiness review, L014/L015 and current-source T110 remain pending. Original
counts87/9/15 and008:T096 NEGATIVE remain unchanged. No new PR CI is awaited.
