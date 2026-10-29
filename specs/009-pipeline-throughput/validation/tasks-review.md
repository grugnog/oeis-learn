# Spec 009 task-generation review

**Date**: 2026-10-23  
**Result**: PASS for task-document structure and design coverage only. No production implementation, hardware measurement, study admission or CI result is claimed.

The repository `speckit-tasks` skill and resolved task template were applied after reading the specification, plan, research, data model, four contracts, carryover register, quickstart and adopted constitution. No before/after extension hooks are configured. The generic template's separate setup/foundation phase numbering was adapted to preserve the approved six delivery phases and public CLI meanings; shared foundations are part of delivery phase 1.

| Scope | Tasks | Count |
| --- | --- | ---: |
| Shared setup/foundations and closeout | T001–T008, T109–T111 | 11 |
| US1 — consolidated validation | T009–T026 | 18 |
| US2 — baseline and inference comparisons | T027–T042 | 16 |
| US3 — deterministic concurrency and overlap | T043–T076 | 34 |
| US4 — qualification and cost | T077–T086 | 10 |
| US5 — gated research | T087–T108 | 22 |
| Total | T001–T111 | 111 |

Automated document checks verified sequential unique unchecked IDs, required story labels, exact file paths, 21 parallel markers, verbatim field constraints for all 14 data-model entities, explicit coverage of all 30 functional requirements and 10 success criteria, and ownership of all 13 unchecked Spec 008 tasks. Spec Kit prerequisite discovery includes `tasks.md`. Historical Spec 008 task lines and runtime source are unchanged. Relative documentation links and whitespace checks pass. See [machine-readable review](tasks-review.json).

Manual dependency review confirms each story has an independent acceptance criterion, tests before the relevant implementation, a checkpoint and parallel examples. Parallel markers describe independent file work after common prerequisites, not concurrent performance measurements. Phase 2 inference and phase 3 CPU-native development can proceed independently; phase 4 requires the qualified common v2 policy and explicit selected-component/baseline choices. A failed optimization may produce a no-adoption report without blocking unrelated engineering work.

Code/local verification and target-machine evidence have separate tasks. Phase 3 fixed-candidate validation does not require HIP. Actual GPU claims and adoption do. The default operator handoff remains one command across all ready engineering phases and one combined archive, with at most 84,000 seconds of declared reservations within the 86,400-second ceiling. The user's 90% physical-memory allowance remains an aggregate ceiling, distinct from expected peak reservations and the old historical16 profile.

Research orchestration can be implemented and fixture-tested without admitting a study. Actual research requires its explicit decisions, source-bound qualification and scope-specific cost receipts. New learnability is not circularly gated on an earlier learning PASS; representation and dependent learning still require that PASS. Final cohort access and a longer run remain separate authorities. The inherited structure-family uncertainty method is explicit: 2,000 fixed-seed bootstrap resamples for descriptive 95% intervals, with seed spread reported separately.

All historical references use `008:` where necessary to avoid confusing the two task-ID namespaces. Historical T093 remains deferred; historical T096 remains negative with its consumed budget. Optional assistance deferral does not complete its execution task. Blocked or negative research cannot be silently promoted by a complete engineering report.

**Suggested MVP**: T001–T026 supplies a failure-safe consolidated validator before optimization. Full phase 1 is T001–T034, adding representative baseline/calibration and frozen comparison contracts. The next implementation request should target phase 1. Pending hardware/research tasks stay visible and unchecked; task generation does not launch any of them.
