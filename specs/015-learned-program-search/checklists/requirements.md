# Specification Quality Checklist: Round028 Learned Program Search

**Purpose** Review the requirements before research planning.

**Created** 2026-11-25

**Feature** [Round028 — Learning to Select and Generate Sequence Programs](../spec.md)

## Content Quality

- [x] The spec states research decisions and observable behavior while leaving implementation choices to planning.
- [x] The research and operator needs are explicit, including how a negative result saves GPU time.
- [x] The stories explain the experiment in terms a researcher or workstation operator can follow.
- [x] All mandatory template sections are present and in order.

## Requirement Completeness

- [x] The choices needed to start planning are resolved or given an explicit prospective default.
- [x] Each functional requirement can be tested using identifiable evidence.
- [x] The success criteria have measurable outcomes, denominators or reproducible decisions.
- [x] An independent reviewer can verify the outcomes without depending on an implementation choice.
- [x] Each of the four stories has an independent test and acceptance scenarios.
- [x] Edge cases cover ambiguous continuations, missing or incompatible records, empty candidate sets, leaks, interruption and resource exhaustion.
- [x] The scope distinguishes historical audit, exploratory learning, promotion and future confirmation.
- [x] Dependencies and assumptions identify the feature 013 foundation, the qualified donor, private evidence and the workstation budget.

## Feature Readiness

- [x] Each functional requirement is covered by a story and a verifiable outcome.
- [x] Primary flows cover diagnosis, selection, generation, promotion, restart and independent reporting.
- [x] The success criteria cover both a real learning comparison and a usable outcome when evidence is incomplete.
- [x] Existing campaign contracts supply compatibility requirements. Model implementation, storage format and scheduling choices remain open to planning.

## Review Notes

| Requirements | Where the reviewer can check them |
| --- | --- |
| FR-001–FR-003 | Story 1 and SC-001/SC-003 account for the historical cases, paired differences, comparable checkpoints, missing evidence and the action each diagnosis supports. |
| FR-004–FR-006 | Story 2 and SC-002/SC-004 cover training-only labels, grouped panels, candidate-set identity, shuffled inputs and equal-budget end-to-end comparisons. |
| FR-007–FR-009 | Story 3, the spending-decision table and SC-003/SC-004 cover matched trials, exploratory milestones, fresh families, independent seeds and the inherited promotion criteria. |
| FR-010–FR-012 | Story 4 and SC-005/SC-006 cover qualification, compatible reuse, exact commands, resource measurement, interruption, cost reconciliation and verification. |
| FR-013–FR-015 | Story 4 and SC-004–SC-006 cover critical limits, production evidence or conservative provisional bounds, large valid inputs, slow progress, exhaustion and a recoverable result. |

The review distinguished the eight-hour planning estimate from a worker timeout. It also resolved which development result can earn a second seed and which uncertainties warrant a bounded replay. All 16 requirements-quality criteria passed. Workstation acceptance and scientific promotion are evaluated during later phases.
