# Fourth operator archive: scoped engineering pass, no adoption

The actual bundle verifier accepts `009-20261024T141450Z.zip`: 333 listed members plus the canonical manifest, all hashes/sizes, safe paths and consistent summary. The 8,248,378-byte original is unchanged. Its entire 689-file source closure matches the current checkout exactly: `sha256:165655e5ad8299f5879d6af098f6fe584192d040bb73efa69c0d3df2d558546a` at producer `6156215`. The run took 2,819.50 seconds (about 47 minutes). [Exact observations and receipt identities](operator-20261024T141450Z-review.json) are retained.

| Result | Disposition |
| --- | --- |
| Selected scope | PARTIAL; no stage FAIL |
| Selected Podman prerequisite engineering gates | PASS; not whole-feature or both-engine qualification |
| Portability | PENDING; independent Docker not selected |
| Optimization adoption | None; retain the existing sequential baseline |
| Learning / research / longer run | NOT_ADMITTED / NOT_EXECUTED / NOT_ADMITTED |

Target JUnit has zero failures/errors: **602 CPU semantic tests**, **15 actual-HIP tests**, **123 retirement tests**, **1115 supported-workflow cases with nine skips**, and **45 policy-role cases**. The 26 CPU decoder cases cover long replay, batch 1/8, eager/SDPA, mixed controls and invalidation; 14 cases in the HIP module plus the separate device workflow case cover actual-device parity/update/checkpoints. Six finalization tests and 22 local-wrapper tests in the workflow suite pass. These are fixture results, not research execution. Producer host tests pass all 89 cases.

Both current images pass whole-package inspection. CPU/HIP role isolation, operations and resource receipts pass within the selected scope. Minimum observed available output-filesystem space is 664,204,681,216 bytes; the earlier storage problem did not recur. Full diagnostic checkpoint roundtrips complete in the actual neural roles. No new implementation defect was found.

## Measurement outcomes and plan alignment

The review follows `plan.md`, FR-006–011/024–030, SC-004/005/007–010, and `contracts/comparisons.md`/`scheduling.md`. Requirement checklist remains 16/16 checked; Spec Kit prerequisites resolve the existing feature directory. There are no extension hooks. No requirements or numerical, scheduling, threshold, workload, deadline or resource policy was revised.

1. **Baseline:** no complete prompt among the frozen 32. All 576 scheduled records remain: 376 decoded, 32 generation deadline outcomes and 168 unstarted at the parent deadline. Only 408 attempts started. All 500 updates and 256 teacher-forced validations completed. The frozen fixed-work subset is empty, so no child comparison can be admitted. The [phase-1 baseline index](phase-1-baseline.json) keeps resources and omissions; an overall correctness PASS cannot supply missing complete timing work.
2. **Inference:** numerical fixtures pass, but paired performance timing is unstarted because baseline calibration is INCONCLUSIVE. See the [phase-2 index](phase-2-inference.json) and [decision](phase-2-decision.md). No transparent optimization or isolated speedup is inferred from fixtures.
3. **Native:** the cold corpus completed all 18 rows. Calibration repetitions completed 12/18, 12/18 and 18/18; the warmup completed 16/18. Resource exhaustion/expiry remains incomplete, not a conclusive negative fit. The exactly-three-complete calibration gate therefore rejects candidate timing; zero worker-2/4 candidates ran. The wrapper stage PASS means a completed report explicitly retained workers=1, while its sealed comparison remains INCONCLUSIVE. See [native index](phase-3-native.json) and [decision](phase-3-decision.md). Queue residence continues to consume the original four-second service allowance; it is not renewed after dispatch.
4. **Joint pipeline:** unstarted on the empty baseline subset. There are no exploratory winners, five-pair cold/warm confirmations, fixed-budget comparisons, capacity/cache gains or supported optimized envelope. See [joint index](phase-4-joint.json) and [decision](phase-4-decision.md). No missing pair becomes a zero-duration sample.
5. **Costs:** diverse32 retained one partial observation out of the required three, with no complete prompt and missing material/full-control scoring evidence. Transfer scheduled all 8192 records for 256 prompts, started 2891, and completed no full prompt. Its GPU host process hit the external deadline; although a partial role summary/checkpoint exists and the eventual return code is zero, the supervisor observation remains FAIL and no scorer/trial result is inferred. Both cost forecasts stay INCONCLUSIVE with unknown components and no admission. The [phase-5 qualification index](phase-5-qualification.json) retains these independent outcomes and all thirteen carryovers.

A useful negative/no-adoption outcome is permitted by SC-004 and the plan. It does not close missing collection tasks, establish the 10%/5% gain rule, qualify a 60-second research portfolio, complete second-engine coverage or authorize studies. `008:T093` stays deferred; original `008:T096` stays negative at 7730/14400 seconds. The historical 13.29–13.40x small-value gap and >5x review trigger remain unresolved.

## Next authority

No repeat of the unchanged command is requested. The current finite scope has reached a no-adoption decision; an identical retry is not a prescribed path to completion. Per FR-011 and the comparison contract, any future measurement investigation must first have a separately identified prospective scope revision, preserving this result. A narrow complete-work component investigation may be proposed, but must retain original stress/coverage denominators and cannot substitute for full-pipeline confirmation or complete research costs. Increasing group/service/phase allowances, filtering prompts within this comparison, choosing favorable repetitions or changing seeds/thresholds is not authorized by this review.

Independent Docker capability/qualification remains a separate optional collection boundary. A revised learning/research batch remains blocked by complete exact-scope cost and its explicit prospective decision; no final cohort was exposed. The task index closes only executed fixture tasks and the three now-published non-adoption decision reports; collection/adoption tasks lacking their full evidence remain open. No new code or source-dependent rerun is needed for this documentation review, and new PR CI is not awaited.

A [draft narrowed scope proposal](measurement-scope-revision-proposal.md) specifies baseline-only forced replay within existing phase limits, with exact replay payloads and a distinct adapter/manifest still required before review/execution. It grants no timing, optimization or research authority and cannot fill the missing full-pipeline costs.
