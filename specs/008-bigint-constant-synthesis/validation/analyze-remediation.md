# Cross-artifact Analysis Remediation

Date: 2026-10-18. Scope: the five findings from `/speckit.analyze`, resolved in design documents and the implementation checklist. This is document validation, not native build, runtime, GPU or research qualification.

| Finding | Resolution | Contract and task evidence |
| --- | --- | --- |
| A1 — Ambiguous transfer success | `continuation_accuracy_100` requires production and independent agreement for the one visible-only sealed selection across all100 indices. Prefix-only fit is a separate metric. Promotion uses the exact-fitting equal-budget portfolio for every conditioning/template comparator; raw and rescue remain separate reports. | [Experiments](../contracts/experiments.md); FR-022; T044/T050/T072/T074/T101 |
| U1 — Missing final-cohort lifecycle | Sealed throughout the studies; an immutable post-decision final plan freezes arms/checkpoints and budgets. Only explicit registered final evaluation authorizes mounts; first evaluation durably exposes the cohort. Identical-plan resume retains exposure and consumed allowance. T105 executes the final study after input/assistance decisions and before readiness. | [Workflow](../contracts/workflow.md), [data model](../data-model.md), [quickstart](../quickstart.md); FR-025; T037–T038/T053/T060/T064–T066/T079/T099/T104–T107 |
| I1 — Phase 2 test forward dependency | A bounded qualification-only native core driver supplies execution before T025 and fitting/replay before T036. Independent Python admission uses its provisional output. Production worker/client remain Phase 3 integration. | [Solver](../contracts/solver.md), [algorithm handoff](../algorithm-phase.md); T017/T025/T033/T035/T036 |
| U2 — Unspecified queue/cache capacities | Explicit aggregate per-role counts/bytes, one active/eight pending batches, FIFO deadline-charged backpressure, deterministic LRU, oversized-entry bypass and reservation release after failure/cancellation. Policy tests precede Phase 2 implementation; production pressure tests follow in Phases 3/5. | [Solver](../contracts/solver.md), [workflow](../contracts/workflow.md); T027/T035/T038/T059/T062/T074/T091 |
| I2 — Benchmark width conflict | Supplied values stay within4096-bit production limits. Up-to16384-bit intermediates/results are constructed by legal operations; expected overflow is reported separately. Direct wider inputs need an explicitly different benchmark-only profile. | [Runtime](../contracts/runtime.md), [plan](../plan.md); T094 |

## Validation performed

- All107 task IDs are unique, sequential, unchecked and use the required path-bearing checklist format.
- Six implementation phases remain, with counts10/48/22/8/10/9. All hard algorithm work remains in Phase 2, including final-plan and service-capacity policy semantics.
- Story counts are US1=20, US2=15, US3=24, US4=26, US5=9 and shared=13. Eleven conditional parallel test-writing tasks remain.
- All31 functional requirements and nine success criteria retain mapped work. The five findings have explicit implementation/test tasks; no new requirement or story is left unmapped.
- Relative document links, Markdown fence balance, absence of unresolved placeholders, quickstart Bash syntax and `git diff --check` pass.
- Cross-artifact review checks the named metric/portfolio, final state/mount/ledger ordering, early native test access, service-capacity accounting and benchmark input-versus-intermediate widths. The five findings are closed at design level; their runtime acceptance remains unchecked work.

No constitution amendment or new hard-algorithm phase is introduced. Dependency feasibility, actual GPU behavior and research outcomes remain implementation gates. Slow CI is not monitored as part of this documentation correction.
