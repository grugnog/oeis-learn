# 010-followup-003 target evidence review

All 153 payloads listed in the supplied matched ZIP manifest match their byte counts and SHA-256 hashes. The command plan starts at step 4 and preserves skipped steps 1–3 explicitly. Within this invocation, current-source CPU/native qualification passed 815 tests and HIP qualification passed 26 tests. No memory-ceiling breach is recorded.

## Failures

Snapshots 2 and 6 completed generation for all 296 rows and all 1,184 registered portfolios (9,472 attempts each), in 2,511 and 2,426 seconds respectively. Both generation role summaries are COMPLETE with complete cost clocks. Their enclosing stages failed before scoring because `followup_evaluation.evaluate` used exclusive writes twice on `private/settings.json`. Immutable generation and scoring settings now have separate paths; the scoring descriptor adds the hash of the completed selections. Hidden truth remains mounted only in the scorer.

Snapshot 5 stopped after 619 seconds with `TimeoutError: native/reference task deadline; external watchdog may be required`. Only 224/1,184 portfolio clocks completed; it has no complete selections file. The scheduler previously aborted as soon as active native work crossed its service deadline, even with a safe incomplete-result adapter. Matched policies now allow at most five seconds for that active operation to quiesce, keeping its slot charged and pausing new work. The original service deadline still excludes late candidates. A genuinely stuck operation still aborts after this bounded cleanup, and the existing cancellation/quiescence checks remain in force. Legacy policy behavior is unchanged.

The generated report has no scored points. Neither matched-drain nor the training screen ran. No semantic accuracy or training-policy comparison can be concluded from this archive.

## Available observations

| Snapshot | Attempt COMMITTED | Attempt INCOMPLETE | Cost-complete portfolios |
| --- | ---: | ---: | ---: |
| 2 / scheduled 3-hour | 3,199 | 6,273 | 1,184 / 1,184 |
| 5 / scheduled 24-hour | 367 | 1,427 | 224 / 1,184 |
| 6 / retained selected | 3,142 | 6,330 | 1,184 / 1,184 |

Snapshot 5 additionally has 7,662 unstarted attempts and 16 generated attempts without a terminal event. Clock completeness is cost accounting, not scientific completion. Severe fitting censoring remains in the matched overlap policy: neither completed snapshot has a fully complete equal-budget portfolio on any generic row. Independent raw measurement completes for 127/128 parameter-development and all 128 structural-development rows in both completed snapshots.

The repaired template baseline selects visibly verified candidates for 53/128 parameter-development and 51/128 structural-development rows in each completed snapshot, with all those template portfolios complete. These are visible-prefix admissions, not hidden-continuation successes. They establish that the former unsupported-template routing failure has been removed; independent scoring is still required.

## Repairs and verification

In addition to the observed failures, review of the same path found a second exclusive write of the already-written training evaluation result, and repeated exclusive writes of the live training validation journal. Removed the duplicate evaluation write and made the progress journal explicitly mutable. Public final evaluation records remain exclusive.

Host regressions exercise the actual evaluation adapter and fresh-training stage through generation, scoring and reporting with isolated role stand-ins. Another runs the actual training control flow through initial, periodic and final validation plus best/latest checkpoint export using lightweight tensor-operation stand-ins. Scheduler regressions require cooperative expiry to preserve other prompts, exclude late results, charge the ledger and quiesce; a separate deterministic case requires bounded failure for uncooperative work. These tests verify control flow and file semantics; they are not HIP/native scientific measurements.

The implementation disposition fingerprint is refreshed with the canonical package serializer. Retain the supplied evidence unchanged; rerun from step 4 into a fresh output root such as `010-followup-004` after pulling the repaired PR branch. Each invocation qualifies current source again.
