# Spec Kit cross-artifact analysis — expanded 007

**Date**: 2026-09-16. Followed installed `speckit-analyze` v1.0.7 after regenerating tasks with `speckit-tasks`. The analysis phase was read-only; this file records its output afterward as a requested deliverable. Earlier findings were corrected under the user's explicit instruction before the final analysis. This is semantic review, not an upstream command that proves the design correct.

## Findings

| ID | Category | Severity | Location | Disposition |
| --- | --- | --- | --- | --- |
| GOV-001 | Constitution activation | CRITICAL implementation gate | `constitution-rfc.md`, T001–T002 | Proposed2.0.0 is not ratified. Gather actual standalone device feasibility and record adoption before implementation promotion beyond that gate. No script exit code or scope update waives it. |

No further HIGH/CRITICAL cross-document inconsistency was identified in the reviewed design. This does not establish implemented correctness, hardware feasibility or mathematical completeness. Unknown proof classes and inconclusive experiments are specified outcomes, not fabricated successes.

## Coverage

| Inventory | Covered / total | Evidence |
| --- | --- | --- |
| Functional requirements | 38/38 | Substantive task mapping in tasks.md |
| Measurable success criteria | 14/14 | Story-level implementation and tests, not final aggregate checks alone |
| User stories | 8/8 | Independent acceptance fixtures and dependency joins |
| Implementation tasks | 105/105 | Ordered IDs, concrete paths, story labels, requirement links and acyclic dependency graph |
| External review/decision items | 68/68 dispositions | 20 independent recommendations,19 architecture proposals,17 performance proposals and12 grouped session decisions |

Only three decision rows defer work: native LODA, conditional LODA-to-WASM and the paired language comparison. Unsupported embedding-corruption claims and optional additional CLI aliases are not adopted with recorded reasons. Already-fixed positional buffering retains regression coverage. Every other review item maps to concrete non-final tasks. [The complete input map](../review-coverage.md) and [machine inventory](review-inputs.json) are separate from internal requirement coverage.

## Detection passes

- **Duplication:** Initial `foundation/v1` SFT control and `qualified/v1` extension modes are explicit; initial disabled paths do not satisfy mandatory repairs. Shared immutable identities and isolation rules apply to both.
- **Ambiguity:** Twenty visible terms is confirmed. Horizon, checked arithmetic and experimental numerical thresholds remain explicit planning defaults. Bounded certificate families specify UNKNOWN outside support. Unmeasured and untriggered optimization branches have different dispositions.
- **Underspecification:** Contracts enumerate repaired algorithms, profile semantics, entity fields, CLI boundaries, budgets and meaningful negative fixtures. New modules are implementation targets, not claimed existing functionality.
- **Constitution:** The proposal preserves exact authoritative data, independent acceptance, bounded work, reproducibility, strict provenance and scoped evidence. Ratification/feasibility remains GOV-001.
- **Coverage:** Every actionable non-LODA recommendation has a concrete task/test or a bounded conditional decision; disabling a known defective subsystem cannot mark its repair complete.
- **Consistency:** Source, codec, objective, active checkpoint state, proof scope, streaming retry, split membership and experiment budgets align. G0–G6 mean baseline readiness; G7–G10 are mandatory for full007 completion.

## Material corrections made during this expansion

| Earlier issue | Correction |
| --- | --- |
| Quarantine replaced immediate proof repair | Both original APIs repaired in T057; original counterexample tests T051; G7 before measurements. |
| Formula certificates accidentally gated on late discovery | Initial formula checker/report explicitly at G7; richer program-class/compiler linkage at G9. |
| Generic-only rule conflicted with model self-training | Generic bootstrap/control remains strict; qualified model discoveries from training-only prefixes use explicit provenance and admission rules. |
| Global no-replay/no-RL statements contradicted new tasks | Scoped those statements to the fixed foundation/v1 smoke; qualified active state is separately specified. |
| Trigger acquisition had no budget | One30-minute diagnostic is charged within six training hours, with55-minute paired arms; absent evidence is inconclusive. |
| Unselected triggered branches labeled not-triggered | Use baseline_retained with higher-priority-selection reason; reserve not_triggered for measured false conditions. |
| Self-training could bypass synthetic holdouts | Canonical expanded structure/parameter/composition checks cover replay, retrieval and macros for every origin. |
| Stateful fuel retries could reuse partially mutated memory | Restore pre-next state or reset/replay with charged work; checkpoint committed transitions only. |
| Streaming and shift semantics ambiguous | next emits the sequence value; block adapter bounded; negative/large shifts rejected and logical left shift checked. |
| Internal coverage confused with input completeness | Separate68-item hashed-source inventory and validator checks; explicit resolved-gap map. |

The nine original design corrections remain: acyclic admission IDs; final decision/seal ordering; crash-gap charging; visible-only seed projection; final-blob publication ordering; nonquadratic group witnesses; standalone preflight; correct GPU test role; and failing-before-fixed regression sequencing.

## Next actions

Implement T001's bounded feasibility probe, then record T002 adoption. US1 is the first functional MVP; US5 repairs can start immediately after US1 alongside the data/operation slices. Do not defer known proof/solver repairs until architecture experiments. All105 implementation boxes remain unchecked. Full runtime acceptance and the capped measurements belong to implementation; this document update ran neither training nor GPU/native application tests.
