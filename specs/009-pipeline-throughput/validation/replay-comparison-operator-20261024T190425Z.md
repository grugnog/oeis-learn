# Replay comparison collection: fixture thread failure and repair

The uploaded `009-replay-comparison-20261024T190425Z.zip` verifies all122 listed
members plus its manifest. Archive SHA256 is
`86af612941410a1160b65ff546f1837c7094c7bfec129c5ca89b8b8a51d34414`.
It matches producer `ba8e64e` and all697 source-closure entries exactly, source
`0960ff8fe166dfa0270c4d7da67ecb40aca019f3074c7d025a7afbd75ee2c81c`.
The original upload is unchanged. [Machine-readable review](replay-comparison-operator-20261024T190425Z.json).

## Actual outcome

The invocation took789.81s and ended selected-scope/engineering FAIL, portability
PENDING, feature incomplete and research/learning/long-run admission denied.

| Executed evidence | Result |
| --- | --- |
| Host stdlib suite | 105 passing tests |
| CPU qualification | 621 passing tests, zero failures/errors/skips |
| Actual HIP qualification | 15 passed,1 failed, zero errors/skips |
| Current qualification/GPU images | Source inspection PASS |
| Visible-only CPU reference | COMPLETE; qualified resource observation |
| Five baseline replay roles | BLOCKED after HIP preflight failure |
| Child freeze and five comparison roles | BLOCKED; no candidate dispatch |
| Independent reports | INCONCLUSIVE; no calibration, pairs or gain |

No private tensor or weight payload is present in the ZIP. Stage receipts retain
their canonical identities; source, manifest, member sizes/hashes and failed-path
reporting verify. The prerequisite graph correctly prevented performance timing
after the mandatory HIP fixture failed. This is neither a measured candidate
regression nor evidence of improvement.

The producer's already-completed CI workflows40122875898 (Foundation) and
40122875900 (Container/native) both passed. Their success did not exercise this
actual-HIP-only comparison-fixture branch and cannot override the failure.

## Cause and completed repair

`test_actual_hip_baseline_replay_reference_and_frozen_roles` failed inside
`freeze_comparison -> envelope` with `ValueError: frozen replay threads required`.
The fixture mocked `torch.set_num_interop_threads` to permit several role-function
calls in one pytest process. Its real thread observation consequently retained
that shared process's existing interop setting. The exact incompatible value is
not retained in the traceback and is not invented here. The production contract
correctly requires actual two intraop/one interop threads.

The fixture now runs in a fresh subprocess with a180-second external bound and
sets actual interop=1 before tensor work. Repeated fixture role loads assert that
real setting rather than silently mocking configuration. It checks the GPU role's
actual2/1 observation and verifies the parent pytest thread pools are unchanged.
The child still exercises CPU reference, baseline and both real comparison arms
against the serialized oracle and retains all numerical/mutation/slot rejection
checks. No production code, thread gate, numerical tolerance, hypothesis, pair
order, manifest, workload or operational budget changes.

A host regression constructs correctly sealed incompatible thread observations:
baseline viability may complete, but prospective comparison freezing must reject
them. The full available host registry now passes106 tests. Compilation and
whitespace checks pass. The actual public stdlib CLI again exits3 for the absent
engine and verifies a53-listed-member non-admitting archive at the corrected
fixture source; [local evidence](replay-comparison-thread-fix-local-evidence.json).
There is no local Torch/pytest/engine/HIP runtime, so the repaired CPU/HIP fixture
is still pending real execution. Production Python implementation bytes and the007
implementation binding are unchanged; source qualification includes changed tests.

## Next collection

Use the same registered comparison command after updating the existing branch,
with a fresh output directory. The [combined handoff](replay-comparison-implementation.md#one-target-command)
builds/inspects current images, runs affected prerequisites and only then collects
new baseline calibration and the frozen candidate schedule. No automatic retry,
extra pair, favorable sample reuse or budget increase is introduced. There were
no candidate observations to retain or select from this failed prerequisite run.

The previous run's13m10s is prerequisite/failure cost, not an expected successful
comparison duration. Earlier accepted baseline intervals remain historical
29.60–29.78s; candidate performance is unknown. Existing24-hour global and phase
ceilings,600s role caps and separately derived setup/startup allowances remain.
Return the single new ZIP even if non-PASS. New PR CI is not awaited.

Original task counts remain77 complete/19 runtime-evidence pending/15 blocked
research.009:C004 and new-sourceT110 remain pending. The earlier accepted replay
scope is preserved. Full-generation, native/joint, operations/portability, complete
cost and research gates remain open.008:T093 stays deferred and008:T096 stays
negative at7730/14400s. No optimized default, study or final exposure is admitted.
