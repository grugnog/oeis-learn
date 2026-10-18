# Feature Specification: Exact Bigint and Parameterized Sequence Synthesis

**Feature Branch**: `008-bigint-constant-synthesis`

**Created**: 2026-10-18

**Status**: Draft

**Input**: User requests arbitrary-size integer execution, exact constant fitting with reusable parameters, Rust-based execution and solving, and a controlled bootstrap strategy that improves OEIS coverage while supporting investigation of new mathematical approaches. Fully remove the old Python solver and other superseded executable paths. Preserve spec 007 correctness and experimental controls.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Execute and condition on large exact integers (Priority: P1)

As a researcher, I can train and evaluate sequences whose visible terms, intermediate calculations or continuations exceed signed 256-bit range without silently changing their mathematical meaning or shortening the evaluation horizon.

**Why this priority**: Fixed numeric limits exclude useful sequences and can bias training toward slow-growing behavior.

**Independent Test**: Evaluate a frozen arithmetic corpus without a model, including factorial through index 99, large intermediate cancellation, negatives and adjacent large integers. Separately exercise the large-value encoder input path.

**Acceptance Scenarios**:

1. **Given** sufficient declared resources, **when** factorial is evaluated at indices 0 through 99, **then** all 100 results match independent exact expectations.
2. **Given** two operands exceeding 256 bits, **when** they are multiplied within the profile limits, **then** the exact product is returned without narrowing either operand to a machine scalar.
3. **Given** an exhausted resource limit, **when** execution stops, **then** it returns a classified failure rather than a numeric answer and a subsequent healthy request succeeds.
4. **Given** visible terms beyond 256 bits, **when** they pass through preparation and encoding, **then** authoritative values remain exact and qualified forward/backward updates remain finite.

### User Story 2 - Fit mathematical parameters exactly (Priority: P1)

As a researcher, I can propose an algorithmic structure and fit its integer parameters using only visible terms, retaining independent evidence that the final substituted program works.

**Why this priority**: Digit prediction should not prevent a useful structure from being evaluated; a solver status cannot replace final execution.

**Independent Test**: Fit hand-authored and concrete decoded programs without training, covering unique, ambiguous, inconsistent, large-coefficient and unsupported cases.

**Acceptance Scenarios**:

1. **Given** a supported polynomial structure and visible terms, **when** fitting succeeds, **then** its final substituted program independently matches every visible term, including coefficients outside the old ±1,000 bound.
2. **Given** repeated occurrences of one parameter identifier, **when** it is fitted, **then** every occurrence uses the same integer assignment.
3. **Given** unsupported control flow or a timeout, **when** results are reported, **then** neither is mislabelled unsatisfiable or verified.
4. **Given** an assignment producing 16 where the visible term is 4, **when** final verification runs, **then** it is rejected regardless of the solver status.

### User Story 3 - Learn structures and measure transfer (Priority: P2)

As a researcher, I can compare literal and parameterized targets on related generic programs, advance through increasing complexity and distinguish algorithm acquisition from fitting a fixed template.

**Why this priority**: The recent offset-only diagnostic can be solved by a fixed template plus a solver without learning algorithm selection.

**Independent Test**: Prepare deterministic matched datasets and evaluate fixed model outputs and a non-neural baseline; perform bounded matched GPU experiments only after runtime qualification.

**Acceptance Scenarios**:

1. **Given** matched literal and parameterized datasets, **when** experiments run, **then** behavior, split and compute differences are disclosed and parameter transfer is reported separately from held-out structural transfer.
2. **Given** correct, shuffled and absent prompts, **when** raw and fitted synthesis are evaluated, **then** the model's contribution beyond fitting and the fixed-template baseline is separately measurable.
3. **Given** search successes, **when** replay admission runs, **then** only independently verified training-target successes enter updates.

### User Story 4 - Compare assistance and investigate discoveries (Priority: P2)

As a researcher, I can compare a generic-only control with a demonstration-assisted candidate and inspect multiple verified algorithms without confusing finite agreement with novelty or proof.

**Why this priority**: Useful prior mathematics can improve synthesis while independent controls preserve attribution and alternative programs support discovery analysis.

**Independent Test**: Audit track manifests and candidate archives, inject forbidden cross-track data and check separate reports with fixed candidate fixtures.

**Acceptance Scenarios**:

1. **Given** a shared generic foundation, **when** training branches, **then** ancestry is recorded, generic-only training excludes assisted data, and assisted provenance propagates to all descendants.
2. **Given** several verified solutions, **when** one is selected for training, **then** declared archive limits still allow retention of structurally or computationally interesting alternatives.
3. **Given** finite continuation agreement, **when** it is reported, **then** proof and novelty remain separate evidence statuses.

### User Story 5 - Operate one supported fitting system (Priority: P1)

As an operator, I can use one portable workflow without accidentally invoking a retired Python solver, fallback algorithm or host-specific integration.

**Why this priority**: Competing legacy paths make results difficult to interpret and reproduce.

**Independent Test**: Audit dependencies and active callers, exercise supported commands in clean Docker/Podman installations and inspect historical artifacts without running the retired solver.

**Acceptance Scenarios**:

1. **Given** a clean installation, **when** a supported fitting command runs, **then** only the new native fitter is used and no old Python fitting implementation is installed.
2. **Given** an interrupted run, **when** it resumes, **then** its contract and consumed budgets are preserved.
3. **Given** a capped analysis export, **when** it is packaged, **then** core contract/status/aggregate records and an omission manifest take priority over large traces.

### Edge Cases

- Large signed neighbors, zero, cancellation, differences wider than their inputs and nonzero sequence offsets.
- Malformed literals, invalid/stale integer references, exhausted memory/work budgets and recovery after traps.
- Rank deficiency, multiple fits, unused parameters, nonlinear dependence, parameter-dependent branches/loop bounds and incomplete programs.
- Duplicate behavior across structures, assisted-data contamination, missing continuation values and partially timed-out evaluation groups.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Provide a versioned exact bigint profile with explicit exceptional outcomes and resource ceilings; preserve signed-256 as an explicitly named compatibility/control profile without silent reinterpretation.
- **FR-002**: Support exact literals, addition, subtraction, full integer multiplication and comparisons. Qualify large visible inputs separately from large execution results, retaining controlled mathematical feature streams.
- **FR-003**: Bound per-term and aggregate work, integer/literal/output size, memory, live values and external time. Independently verify accepted results and report resource exclusions separately from mathematical failures.
- **FR-004**: Support literals plus reusable integer parameter identifiers, deterministic eligible-literal abstraction, exact affine fitting and a bounded declared symbolic subset. Protect structural/control constants and retain original candidates.
- **FR-005**: Substitute, validate and independently execute every fitted final program on all visible terms. Distinguish verified, scoped-unsatisfiable, unsupported, unknown and resource-exhausted outcomes; expose bounds, ambiguity, ranking and consumed budgets.
- **FR-006**: Provide batched native fitting/execution orchestration without Python callbacks for every trial assignment. Retain the independent Python arithmetic reference, not a Python fitting fallback.
- **FR-007**: Freeze matched literal/parameterized experiments, related generic curriculum, raw/fitted search portfolios, conditioning controls, non-neural baselines and parameter/structure holdouts before execution. Permit only verified training-target replay.
- **FR-008**: Maintain generic-only and explicitly demonstration-assisted branches with complete ancestry/provenance. Prevent hidden evaluation data from influencing fitting, learning, candidate selection or curriculum progression.
- **FR-009**: Preserve alternative verified programs under declared archive limits and distinguish finite coverage, structural transfer, conjectures, proof and novelty review.
- **FR-010**: Preserve portable Docker/Podman ROCm workflows, workspace-resident persistent state, complete resumable checkpoints, cumulative budgets and interpretable capped analysis bundles.
- **FR-011**: Remove the old Python solver, obsolete fallback algorithms and solver-only dependencies; migrate every supported caller or remove its retired path. Retain correctness counterexamples as replacement tests, independent verification and historical artifact meaning.
- **FR-012**: Inventory other executable paths as supported, migrated or retired; remove unused superseded commands/configurations/examples rather than retaining competing implementations. Age alone does not justify deleting required verification or controls.

### Key Entities

- **Execution profile**: Exact operations, authoritative values and indices, resource limits and failure semantics.
- **Parameterized program and fit**: Complete structure, reusable identifiers, occurrence mapping, assignments or classified failure and independent evidence.
- **Dataset and experiment contract**: Structures, parameter families, splits, provenance, model/input settings, budgets and selection rules.
- **Checkpoint and candidate archive**: Resumable learning state, track ancestry, raw/transformed candidates, alternatives, verification scope and discovery status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All mandatory arithmetic and historical false-fit fixtures return independently expected values or classified failures, with zero admitted disagreements.
- **SC-002**: Every mandatory exhaustion/interruption case terminates within declared deadlines and cleanup allowance, leaves no active candidate and permits a subsequent healthy request.
- **SC-003**: Large-input qualification covers signed adjacent values at 256-, 512-, 1,024- and 4,096-bit scales and wider first differences, preserving exact serialization and finite actual-device updates.
- **SC-004**: At least three matched seeds produce reproducible raw/fitted/conditioning/template comparisons with complete denominators and separately reported resource-limited coverage. Negative or inconclusive results remain valid outcomes.
- **SC-005**: All supported fitting commands pass migration checks with zero reachable Python solver implementations or silent fallbacks, while replacement regression tests and independent verification remain operational.
- **SC-006**: Clean portable-engine qualification demonstrates input validation, bounded evaluation, budget-preserving resume and mandatory-summary analysis export. Real GPU updates are qualified separately from CPU tests.

## Assumptions

- The user's acceptance of practical synthesis with a generic-only control and an assisted candidate is the explicit assisted-track decision required by constitution v2.0.0. It does not authorize assisted data in strict results or hidden-data leakage.
- Rust execution/fitting machinery and a WASM bigint runtime are user-requested constraints. Python remains for neural training and independent reference checking. Candidate dependencies are dashu-int behind a replaceable boundary and Z3 through Rust bindings; exact versions, interfaces and feasibility qualification belong to planning.
- Default comparison remains 20 visible terms and 100 total terms where source data exist. Previously inspected final cohorts remain diagnostic; revised experiments need an unexposed frozen final cohort.
- Fit existing concrete candidates before changing training targets. Preserve the small backbone and controlled hybrid/tri-stream inputs; widening input representation or vocabulary requires explicit new identities and compatibility checks.
- Initially exclude division/remainder/bitwise additions, parameter-dependent control fitting, unrestricted symbolic solving, GRPO, automatic completion repairs, floating-point fitting, privileged scaffolds and automatic proof/novelty claims.
- Persistent data use the configured workspace and pinned bulk OEIS sources. No sudo integration, lab broker/accounts, SSH or VM-specific dependencies are introduced.
- This specification defines required outcomes, not an implementation plan or evidence of runtime/GPU readiness. Resource defaults and experiment promotion thresholds must be frozen before execution.
