# Feature Specification: Trustworthy Synthesis Readiness

**Feature Branch**: `005-trustworthy-synthesis-readiness`

**Created**: 2026-09-04

**Status**: Draft

**Input**: User description: "Turn the recommended next steps after Run 007 into a specification: establish trustworthy end-to-end evaluation, enforce meaningful readiness gates, activate the intended adaptive training behavior, run bounded ablations, improve multi-state recurrence synthesis, and make theorem claims defensible."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trustworthy End-to-End Synthesis (Priority: P1)

As a research operator, I want every synthesis surface to evaluate a real selected model checkpoint through the same complete workflow, so that a reported success represents a program actually generated from the supplied sequence, resolved where necessary, safely executed, and verified beyond its observed terms.

**Why this priority**: Run decisions are only useful when command-line, benchmark, and report results measure the same production behavior without hard-coded programs, mock representations, or skipped post-processing.

**Independent Test**: Select a checkpoint and a fixed sequence cohort, run both interactive and benchmark synthesis with the same protocol and random seed, and verify that they produce identical candidate outcomes and complete provenance records.

**Acceptance Scenarios**:

1. **Given** a valid checkpoint and at least 20 observed sequence terms, **When** an operator requests synthesis, **Then** all candidates are generated from that checkpoint and sequence and proceed through constant resolution, canonicalization, bounded execution, observed-term verification, and extrapolation verification.
2. **Given** identical checkpoint, sequence, candidate budget, evaluation protocol, and random seed, **When** synthesis is invoked through two supported entry points, **Then** candidate programs, classifications, and aggregate results are reproducible.
3. **Given** an absent, incompatible, or corrupt checkpoint, **When** synthesis is requested, **Then** the request fails clearly and produces no fallback program or success result.
4. **Given** a program that executes successfully but disagrees with any observed or unseen target term, **When** results are summarized, **Then** it is classified as incorrect rather than successful.
5. **Given** a generated skeleton whose constants cannot be resolved within the configured limits, **When** it is evaluated, **Then** the unresolved result and reason are retained without being substituted by a demonstration or reference solution.

---

### User Story 2 - Evidence-Based Run Readiness (Priority: P1)

As a research maintainer, I want preflight checks and production promotion gates to enforce demonstrated learning, compilation soundness, runtime health, coverage, and extrapolation, so that expensive training runs cannot be labeled successful when their core behaviors are failing.

**Why this priority**: The current checks can pass despite zero synthesis success, while production telemetry can finish with an excessive trap rate. Strong gates prevent another costly but inconclusive run.

**Independent Test**: Exercise the readiness suite with known-good behavior and with injected zero-success, invalid-program, excessive-trap, missing-coverage, and extrapolation-failure conditions; verify that every failing condition blocks qualification and identifies its cause.

**Acceptance Scenarios**:

1. **Given** a single-prompt learning check with no exact successes, **When** readiness is assessed, **Then** the check fails even if loss, entropy, or advantage variance appears healthy.
2. **Given** a micro-cohort whose minimum task coverage or competence is below its declared threshold, **When** readiness is assessed, **Then** the check fails and names the deficient metric.
3. **Given** generated programs with syntax failures or a runtime trap rate above the allowed ceiling, **When** readiness is assessed, **Then** production qualification is blocked.
4. **Given** any mandatory gate failure, **When** an operator explicitly overrides the block for diagnostic work, **Then** the reason and operator intent are recorded and all resulting reports are marked unqualified.
5. **Given** a failed candidate, **When** the operator reviews its record, **Then** the record identifies whether failure occurred during generation, constant resolution, assembly, bounded execution, observed-term matching, extrapolation, or compactness verification.

---

### User Story 3 - Controlled Training Decisions (Priority: P2)

As a model researcher, I want short, paired experiments that isolate constant resolution, candidate budget, and adaptive curriculum behavior, so that compute is spent only on mechanisms that measurably improve exact extrapolating synthesis.

**Why this priority**: Run 007 plateaued after roughly 40 epochs, and several intended curriculum mechanisms were not controlling the production run. Controlled comparisons provide more information than immediately repeating a long run.

**Independent Test**: Starting from one frozen checkpoint, evaluate the prescribed configurations on the same cohort, seeds, evaluation budget, and held-out terms, then verify that the report attributes outcome differences to exactly one changed factor at a time.

**Acceptance Scenarios**:

1. **Given** a frozen checkpoint and fixed evaluation cohort, **When** constant resolution is compared with unresolved decoding, **Then** both variants use identical candidate samples and the report shows paired exact-match, extrapolation, failure, and cost differences.
2. **Given** candidate budgets of 1, 8, and 16, **When** they are compared, **Then** the report shows the marginal success gain and marginal evaluation cost for each budget.
3. **Given** fixed and adaptive task allocation variants, **When** short training trials are run, **Then** both receive the same total rollout budget and are compared on per-task coverage, competence, retention, trap rate, and extrapolating pass rate.
4. **Given** adaptive allocation is selected, **When** training runs, **Then** measured task progress controls prompt selection, estimated success controls rollout depth, and dormant verified tasks receive replay rather than these mechanisms remaining inactive configuration.
5. **Given** multiple random seeds, **When** results are aggregated, **Then** all seed outcomes and uncertainty are reported; the best seed alone cannot determine promotion.

---

### User Story 4 - Multi-State Recurrence Readiness (Priority: P3)

As a synthesis researcher, I want the system to learn and generate complete bounded state transitions for recurrence sequences, so that it can move beyond polynomial synthesis to extrapolating Fibonacci, Lucas, Pell, geometric, and unseen linear-recurrence programs.

**Why this priority**: Recurrence candidates currently compile and execute but update accumulator state incorrectly, which is the clearest blocker to Curriculum Stage 2 progress.

**Independent Test**: Train on recurrence demonstrations that are disjoint from a fixed evaluation cohort, then verify exact outputs for all observed and 100 unseen terms on the four named canaries and at least one unseen recurrence family member.

**Acceptance Scenarios**:

1. **Given** a recurrence requiring old values of multiple accumulators, **When** a candidate performs an iteration, **Then** every next-state value is derived from the same prior state before any required prior value is lost.
2. **Given** a generated loop, **When** it continues to another iteration, **Then** its transition assignments are complete and its bounded progress condition has advanced.
3. **Given** Fibonacci, Lucas, Pell, and powers-of-two canaries, **When** synthesis is evaluated, **Then** successful programs match all 20 observed and 100 unseen terms within the resource and compactness limits.
4. **Given** a recurrence represented in the training demonstrations, **When** generalization is evaluated, **Then** evaluation sequences with identical term lists or program bodies are excluded from the training data.
5. **Given** a candidate that memorizes observed values or exhausts its execution budget, **When** it is scored, **Then** it cannot count toward recurrence readiness.

---

### User Story 5 - Defensible Mathematical Discovery (Priority: P3)

As a mathematical researcher, I want discovery reports to distinguish candidate patterns, numerical identities, and symbolic proofs while removing duplicates and trivial relations, so that every published theorem claim has evidence commensurate with its label.

**Why this priority**: Duplicate permutations, zero-coefficient relations, and numerical certificates presented as formal proofs undermine confidence in the discovery pipeline.

**Independent Test**: Supply duplicate permutations, scalar multiples, zero-coefficient relations, finite-sample coincidences, and one valid symbolic identity; verify that only the unique nontrivial identity reaches symbolically proven status.

**Acceptance Scenarios**:

1. **Given** candidate tuples that differ only by sequence order, coefficient sign, or common coefficient scale, **When** discoveries are consolidated, **Then** they produce one canonical claim.
2. **Given** a relation containing a zero coefficient, repeated sequence, or reducible subset, **When** it is assessed, **Then** it is rejected as trivial.
3. **Given** a relation that holds only at selected sample positions, **When** it is checked across all validation and unseen terms, **Then** it is rejected.
4. **Given** a numerically verified relation without sufficient symbolic definitions, **When** it is reported, **Then** it remains a numerical conjecture and is not labeled proven.
5. **Given** complete symbolic definitions and an identity that reduces to equality for the general index, **When** verification succeeds, **Then** the report may label one canonical claim as symbolically proven and include its evidence provenance.

### Edge Cases

- A sequence has exactly 20 observed terms but fewer than 100 authoritative unseen terms.
- The requested sequence is missing from the selected benchmark version or has changed since the benchmark was frozen.
- A checkpoint loads but its model or vocabulary metadata is incompatible with the selected evaluation protocol.
- All candidates are duplicates after canonicalization, leaving an effective candidate budget smaller than requested.
- Constant resolution is unsatisfiable, times out, or produces a program that fits observed terms but fails unseen terms.
- A program assembles correctly but traps through fuel exhaustion, arithmetic failure, or another bounded runtime condition.
- A program matches a prefix because execution returns fewer outputs than requested.
- Integer overflow or signedness changes outputs only beyond the observed horizon.
- Adaptive scheduling begins with no task history or no verified replay entries.
- A previously solved task regresses after a long dormant interval.
- A relation is numerically exact because two source sequences are duplicates or aliases.
- Symbolic definitions are missing, ambiguous, or valid only on a restricted index domain.
- An operator interrupts an experiment before all seeds or paired variants finish.

## Requirements *(mandatory)*

### Functional Requirements

#### Unified Synthesis Evaluation

- **FR-001**: The system MUST use one shared production synthesis workflow for interactive requests, benchmark evaluation, and release qualification.
- **FR-002**: Every synthesis request MUST identify the checkpoint, benchmark version, sequence identifier or supplied terms, observed horizon, unseen horizon, candidate budget, random seed, and resource limits.
- **FR-003**: Production synthesis MUST generate candidates from the supplied checkpoint and sequence; hard-coded programs, mock representations, and silent demonstration substitution MUST be prohibited.
- **FR-004**: Every generated placeholder-bearing candidate MUST undergo configured constant resolution before execution, and both its unresolved and resolved forms MUST remain attributable in the result.
- **FR-005**: Every executable candidate MUST undergo canonicalization, bounded execution, exact observed-term comparison, unseen-term comparison, and compactness assessment in that order.
- **FR-006**: A candidate MUST count as an extrapolating success only when it matches all 20 observed terms and all 100 authoritative unseen terms, stays within the 10,000-instruction and 16 MiB limits, and has a description-length ratio no greater than 1.20.
- **FR-007**: Successful assembly or execution alone MUST NOT be reported as sequence correctness.
- **FR-008**: The system MUST support candidate budgets of 1, 8, and 16 under a common evaluation protocol and MUST report both requested and unique canonical candidate counts.
- **FR-009**: Repeating an evaluation with identical inputs and seed MUST reproduce candidate selection, classifications, and aggregate metrics.
- **FR-010**: Evaluation output MUST preserve candidate-level lineage from generated form through final classification, including durations and explicit failure reasons at each stage.
- **FR-011**: Invalid or incompatible checkpoints, missing terms, and incomplete unseen truth data MUST fail explicitly and MUST NOT produce qualified success metrics.

#### Readiness Gates and Observability

- **FR-012**: The readiness suite MUST require nonzero exact synthesis success in its single-prompt learning check and MUST reject a run that only demonstrates stable optimization statistics.
- **FR-013**: The micro-cohort readiness check MUST enforce declared minimums for competence, per-task coverage, advantage-collapse rate, and runtime-trap rate.
- **FR-014**: Grammar-constrained candidates used for qualification MUST achieve 100% syntactic assembly validity; syntax and environment errors MUST be measured separately from bounded runtime traps.
- **FR-015**: A production run MUST be blocked when any mandatory readiness gate fails unless an explicit diagnostic override is recorded.
- **FR-016**: Results produced under an override MUST be visibly marked unqualified and MUST NOT update best-run or graduation records.
- **FR-017**: Telemetry MUST report, by task and sequence family, candidate generation count, unique candidate count, constant-resolution attempts and outcomes, assembly failures, runtime traps by reason, exact observed matches, unseen matches, compactness failures, task selection probability, allocated rollout count, and replay activity.
- **FR-018**: Readiness and production summaries MUST expose minimum coverage and worst-performing task outcomes in addition to means and peaks.
- **FR-019**: Every failed candidate MUST receive one primary failure-stage classification while retaining any secondary diagnostic details.

#### Controlled Experiments and Active Curriculum

- **FR-020**: The system MUST support paired comparisons of constant resolution on versus off, candidate budgets 1 versus 8 versus 16, and fixed versus adaptive task allocation.
- **FR-021**: A paired comparison MUST hold checkpoint, cohort, observed terms, unseen terms, seeds, resource limits, and all non-tested settings constant.
- **FR-022**: Each comparison MUST use at least three declared random seeds and report every seed, aggregate dispersion, wall-clock cost, and evaluation count.
- **FR-023**: Adaptive training MUST select tasks according to measured competence and progress, vary rollout allocation according to estimated per-task success, and replay dormant verified tasks.
- **FR-024**: Adaptive-selection probabilities, rollout allocations, feedback updates, and replay selections MUST be present in the run record so operators can verify that configured behavior was active.
- **FR-025**: Fixed and adaptive training comparisons MUST receive equal total rollout budgets.
- **FR-026**: An experiment that ends early MUST retain partial results but MUST be excluded from promotion comparisons unless every paired variant and seed completed.
- **FR-027**: No new production-length run MUST be authorized until the bounded comparisons identify a qualifying configuration and all readiness gates pass.

#### Recurrence Progression

- **FR-028**: The training corpus MUST include diverse bounded recurrence demonstrations that require two or more state values and complete state rotation.
- **FR-029**: Recurrence demonstrations and evaluation records MUST be checked for duplicate term lists, equivalent program bodies, and other direct leakage before training.
- **FR-030**: The synthesis language and generation constraints MUST represent a recurrence transition in which all next-state values can be computed from one prior-state snapshot.
- **FR-031**: A loop continuation MUST be considered structurally complete only after required state assignments and bounded progress updates are present.
- **FR-032**: Recurrence qualification MUST include Fibonacci (`A000045`), Lucas (`A000032`), Pell (`A000129`), powers of two (`A000079`), and at least one held-out linear recurrence not represented by an equivalent training example.
- **FR-033**: Recurrence readiness MUST use the same exact observed, unseen, resource, and compactness criteria as all other synthesis qualification.

#### Discovery Claim Integrity

- **FR-034**: Discovery results MUST use canonical relation identities so permutations, global sign changes, and common coefficient scaling are deduplicated.
- **FR-035**: Relations containing zero coefficients, repeated sequence identities, or a proper subset that already establishes the equality MUST be rejected as trivial.
- **FR-036**: Numerical relation validation MUST cover every available designated validation term and a disjoint unseen term set rather than selected indices alone.
- **FR-037**: Discovery claims MUST use distinct statuses for latent candidate, numerically verified conjecture, and symbolically proven identity.
- **FR-038**: A claim MUST reach symbolically proven status only when explicit sequence definitions are available and the general identity is independently reduced to equality over its declared domain.
- **FR-039**: Each discovery claim MUST retain its canonical sequences and coefficients, validation ranges, numerical residual evidence, symbolic definitions, proof outcome, and rejection reason where applicable.

### Key Entities

- **Evaluation Protocol**: The immutable rules for a synthesis comparison, including checkpoint identity, benchmark version, horizons, candidate budget, seeds, resource limits, and enabled processing stages.
- **Synthesis Candidate Record**: The lineage of one generated program from raw output through constant resolution, canonicalization, execution, exact comparison, extrapolation, compactness assessment, and final classification.
- **Benchmark Cohort**: A versioned set of sequence records with observed terms, authoritative unseen terms, family labels, curriculum stage, and leakage status.
- **Readiness Gate**: A named qualification condition with a threshold, observed value, pass/fail result, severity, and diagnostic evidence.
- **Experiment Variant**: One controlled configuration in a paired comparison, linked to common seeds, evaluation budget, outcomes, and compute cost.
- **Task Training State**: Per-sequence competence history, progress, selection likelihood, rollout allocation, last visit, and verified replay availability.
- **Recurrence Transition**: A bounded state update defining prior-state values, next-state expressions, update completeness, and progress toward termination.
- **Discovery Claim**: A canonical mathematical relation with coefficients, evidence ranges, verification status, symbolic definitions, proof evidence, and deduplication identity.

### Scope Boundaries

**In scope**:

- Unifying all user-facing and benchmark synthesis evaluation around real checkpoint inference.
- Applying existing constant-resolution, canonicalization, sandbox, extrapolation, and compactness capabilities consistently at evaluation time.
- Replacing permissive readiness checks with outcome-based gates and complete diagnostic telemetry.
- Activating configured adaptive task selection, variable rollout allocation, and dormant-task replay in actual training orchestration.
- Running bounded paired experiments before authorizing another production-length run.
- Improving and qualifying bounded multi-state linear recurrence synthesis.
- Correcting discovery deduplication, triviality checks, validation coverage, and claim status semantics.

**Out of scope**:

- A new encoder architecture, larger model, mixed-precision training, or distributed high-performance scaling.
- Curriculum Stages 3 through 5, unrestricted recursion, dynamic memory growth, or synthesis targets other than the existing executable program format.
- Relaxing sandbox fuel, memory, determinism, exactness, extrapolation, or compactness constraints.
- Treating a numerical relation as a formal proof or asserting novelty suitable for publication without external mathematical review.
- Launching or completing a new production-length training run as part of this feature; this feature determines whether such a run is authorized.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For 100% of fixed-seed test cases, interactive and benchmark synthesis produce identical candidate classifications and aggregate outcomes under the same evaluation protocol.
- **SC-002**: Auditing a representative set of at least 1,000 candidates finds complete checkpoint-to-verdict lineage for 100% of candidates and zero hard-coded, mock, or silently substituted production results.
- **SC-003**: The readiness suite rejects 100% of injected zero-success, syntax-invalid, excessive-runtime-trap, insufficient-coverage, recurrence-transition, and extrapolation-failure cases while accepting the corresponding known-good controls.
- **SC-004**: Across at least 1,000 grammar-constrained qualification candidates, syntactic assembly validity is 100%, the final rolling runtime-trap rate is at most 15%, and no candidate crashes or destabilizes the host process.
- **SC-005**: All required paired comparisons complete for at least three seeds; no individual comparison run exceeds 4 hours and the full pre-production decision cycle consumes no more than 24 Tier 1 workstation-hours.
- **SC-006**: On the frozen Stage 1 evaluation cohort, resolved best-of-8 inference improves exact 100-term extrapolation pass rate by at least 10 percentage points over unresolved single-candidate inference without increasing the median description-length ratio above 1.20.
- **SC-007**: The selected training configuration reaches Stage 1 rolling competence of at least 0.85, minimum task coverage of at least 0.50, competence variance of at most 0.01, and an exact Stage 1 synthesis pass rate of at least 80% before production promotion.
- **SC-008**: Verified-task retention remains at or above 95% after 500 intervening optimization steps during the bounded adaptive-training comparison.
- **SC-009**: Fibonacci, Lucas, Pell, powers of two, and at least one leakage-free held-out linear recurrence each have a compact program that exactly matches all 20 observed and 100 unseen terms within the resource limits.
- **SC-010**: Every failed synthesis candidate in qualification reports one actionable primary failure stage, enabling an operator to distinguish generation, resolution, assembly, runtime, observed-match, extrapolation, and compactness failures without inspecting raw logs.
- **SC-011**: Discovery evaluation rejects 100% of duplicate, scalar-equivalent, zero-coefficient, repeated-sequence, and finite-sample-coincidence fixtures, and reports numerical-only relations as conjectures rather than proofs.
- **SC-012**: At least one unique relation with all nonzero coefficients passes the full unseen-term check and independent general symbolic verification before the system reports a symbolically proven identity.
- **SC-013**: Zero production-length runs receive qualified status when any mandatory readiness gate is below threshold; every diagnostic override is identifiable in its resulting artifacts.

## Assumptions

- Run 007 and its epoch-60 checkpoint provide the initial baseline; this feature does not assume its published summary metrics are sufficient evidence of end-to-end capability.
- Existing generation, constant-resolution, canonicalization, bounded-execution, adaptive-curriculum, replay, extrapolation, compactness, and discovery capabilities remain available for integration and validation.
- Evaluation uses the first 20 authoritative terms as observations and the next 100 authoritative terms as unseen truth; records without that horizon are excluded from qualified aggregate rates and reported separately.
- The benchmark cohort and leakage exclusions are frozen before experiments begin and cannot be revised in response to outcomes.
- Synthetic demonstrations may support training but do not count as real-sequence benchmark successes.
- Exact integer equality is the correctness standard; approximate numerical agreement does not count as synthesis success.
- Tier 1 workstation limits, strict single-precision neural computation, deterministic sandboxing, a 10,000-instruction budget, and a 16 MiB memory ceiling remain mandatory.
- The existing Stage 1 graduation and anti-memorization thresholds from the project constitution remain authoritative.
- External mathematical review is outside this feature; "symbolically proven" denotes successful internal general symbolic verification with complete evidence provenance.
- No branch is created by this specification because no pre-specification branch hook is configured.
