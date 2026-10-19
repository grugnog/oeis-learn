# Phase 2 algorithm implementation (T011–T058)

Status: **in progress; exit gate closed**. Draft PR #54 contains the US1
runtime/input milestone. Phase 1 was merged in PR #53. Production remains
disabled; no training or Phase 3 integration is started.

Read [the runtime checkpoint](algorithm-runtime.md),
[local evidence](algorithm-runtime-local.json), and the refined
[language syntax](../contracts/language-v1.md). Local verification passed 202
Python tests plus 16 traceability checks. Native/container and actual HIP
execution are explicitly pending. Checked task markers distinguish completed
Python/test-writing work from unexecuted native acceptance gates.

| Mandatory Phase 2 exit domain | Current state |
| --- | --- |
| Exact integers / containment | Guest, parser, lowering, resource limits and independent fixtures implemented; native CI pending at T025 |
| Independent reference | CPU fixtures, limits and disagreement rejection passed |
| Codec / decoder | CPU round trips, typed masks, parameter teacher forcing and cache/full parity passed; actual HIP pending |
| Encoder | CPU width/padding/finite-gradient and old-feature regressions passed; actual HIP pending; four-component ablations remain T039–T040 |
| Affine / SMT | T026–T033 blocked on T025; not implemented |
| Admission / evidence | T027/T034–T036 blocked; arithmetic comparison alone does not grant public admission |
| State / final access | T037–T038 blocked; no final cohort exposure authorized |
| Data / policies / comparisons | T039–T052 blocked; no training or research results claimed |
| Provenance / archives / discovery | T053–T057 blocked |
| Complete exit | T058 unchecked; Phase 3 remains closed |

The task plan explicitly requires T025 before fitting and downstream work.
There is no local container engine or Rust toolchain, so the native gate must be
established by the PR's shared-image checks. The user requested that slow CI be
left running without monitoring. This checkpoint stops at that dependency;
it does not mark the whole algorithm phase complete.

Continue on the same draft PR after the user reports completed checks. Read the
completed logs/artifact identities, fix any failures, finish T014–T017/T019 native
acceptance and T025, then implement the remaining task groups in order. Every
later semantic discrepancy reopens the corresponding algorithm task.
