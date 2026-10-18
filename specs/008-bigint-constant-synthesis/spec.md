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
5. **Given** an existing signed-256 artifact, **when** it is evaluated under its original profile, **then** its arithmetic, exceptional outcomes and identity remain unchanged; changing profile starts a distinct experiment.

### User Story 2 - Fit mathematical parameters exactly (Priority: P1)

As a researcher, I can propose an algorithmic structure and fit its integer parameters using only visible terms, retaining independent evidence that the final substituted program works.

**Why this priority**: Digit prediction should not prevent a useful structure from being evaluated; a solver status cannot replace final execution.

**Independent Test**: Fit hand-authored and concrete decoded programs without training, covering unique, ambiguous, inconsistent, large-coefficient and unsupported cases.

**Acceptance Scenarios**:

1. **Given** a supported polynomial structure and visible terms, **when** fitting succeeds, **then** its final substituted program independently matches every visible term, including coefficients outside the old ±1,000 bound.
2. **Given** repeated occurrences of one parameter identifier, **when** it is fitted, **then** every occurrence uses the same integer assignment.
3. **Given** unsupported control flow or a timeout, **when** results are reported, **then** neither is mislabelled unsatisfiable or verified.
4. **Given** an assignment producing 16 where the visible term is 4, **when** final verification runs, **then** it is rejected regardless of the solver status.
5. **Given** multiple exact fits, **when** one is selected, **then** a frozen visible-only ranking policy selects among discovered assignments and reports known ambiguity without claiming a unique rule or global optimum.
6. **Given** a concrete candidate, **when** mathematical literals are abstracted, **then** the original remains recorded, structural constants are protected, and the final transformed source is independently verified.

### User Story 3 - Operate one supported fitting system (Priority: P1)

As an operator, I can use one portable workflow without accidentally invoking a retired Python solver, fallback algorithm or host-specific integration.

**Why this priority**: Competing legacy paths make results difficult to interpret and reproduce.

**Independent Test**: Audit dependencies and active callers, exercise supported commands in clean Docker/Podman installations and inspect historical artifacts without running the retired solver.

**Acceptance Scenarios**:

1. **Given** a clean installation, **when** a supported fitting command runs, **then** only the new native fitter is used and no old Python fitting implementation is installed.
2. **Given** an interrupted run, **when** it resumes, **then** its contract and consumed budgets are preserved.
3. **Given** a capped analysis export, **when** it is packaged, **then** core contract/status/aggregate records and an omission manifest take priority over large traces.
4. **Given** a retired command or setting, **when** migration is checked, **then** the replacement or removal is documented and unsupported settings fail explicitly rather than selecting an old algorithm.

### User Story 4 - Learn structures and measure transfer (Priority: P2)

As a researcher, I can compare literal and parameterized targets on related generic programs, advance through increasing complexity and distinguish algorithm acquisition from fitting a fixed template.

**Why this priority**: The recent offset-only diagnostic can be solved by a fixed template plus a solver without learning algorithm selection.

**Independent Test**: Prepare deterministic matched datasets and evaluate fixed model outputs and a non-neural baseline; perform bounded matched GPU experiments only after runtime qualification.

**Acceptance Scenarios**:

1. **Given** matched literal and parameterized datasets, **when** experiments run, **then** behavior, split and compute differences are disclosed and parameter transfer is reported separately from held-out structural transfer.
2. **Given** correct, shuffled and absent prompts, **when** raw and fitted synthesis are evaluated, **then** the model's contribution beyond fitting and the fixed-template baseline is separately measurable.
3. **Given** search successes, **when** replay admission runs, **then** only independently verified training-target successes enter updates.
4. **Given** a frozen curriculum advancement rule, **when** synthetic competence reaches its threshold, **then** training advances while retaining earlier-stage practice without consulting final-cohort results.
5. **Given** a trained hybrid model, **when** component sensitivity and matched retraining studies run, **then** their reports distinguish reliance of these weights from the capability of models trained without each component.

### User Story 5 - Compare assistance and investigate discoveries (Priority: P2)

As a researcher, I can compare a generic-only control with a demonstration-assisted candidate and inspect multiple verified algorithms without confusing finite agreement with novelty or proof.

**Why this priority**: Useful prior mathematics can improve synthesis while independent controls preserve attribution and alternative programs support discovery analysis.

**Independent Test**: Audit track manifests and candidate archives, inject forbidden cross-track data and check separate reports with fixed candidate fixtures.

**Acceptance Scenarios**:

1. **Given** a shared generic foundation, **when** training branches, **then** ancestry is recorded, generic-only training excludes assisted data, and assisted provenance propagates to all descendants.
2. **Given** several verified solutions, **when** one is selected for training, **then** declared archive limits still allow retention of structurally or computationally interesting alternatives.
3. **Given** finite continuation agreement, **when** it is reported, **then** proof and novelty remain separate evidence statuses.
4. **Given** a named demonstration or historical checkpoint, **when** strict-track admission is attempted, **then** it is rejected; explicitly admitted assisted data label the receiving run and descendants as assisted.

### Edge Cases

- Large signed neighbors, zero, cancellation, differences wider than their inputs and nonzero sequence offsets.
- Malformed literals, invalid/stale integer references, exhausted memory/work budgets and recovery after traps.
- Rank deficiency, multiple fits, unused parameters, nonlinear dependence, parameter-dependent branches/loop bounds and incomplete programs.
- Duplicate behavior across structures, assisted-data contamination, missing continuation values and partially timed-out evaluation groups.
- A fitted candidate whose actual intermediate size or execution cost exceeds the symbolic model's prediction; fitting does not exempt final execution from any limits.
- Coefficients larger than machine integers, bounds exhausted without a solution, an uncompleted ambiguity check and different numeric profiles sharing superficially identical source text.
- Log-feature calculation on extremely large integers, exact differences requiring an additional bit, unavailable hardware sensors and exports too small for mandatory summaries.

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

The following requirements make the acceptance boundaries of FR-001 through FR-012 explicit:

- **FR-013**: The numerical profile MUST specify signedness, canonical serialization, indices/control values and exceptional behavior. Authoritative values, literals and solver assignments MUST remain exact across all language boundaries, including values exceeding machine integers. Resource exhaustion MUST be distinguished from an incorrect mathematical result; excluded preparation/evaluation cases MUST be counted by reason and integer-size range.
- **FR-014**: Independent verification MUST not derive its expectations from the production arithmetic or lowering under test. Production work limits and reference-checker limits MAY use different accounting, but exhaustion of either MUST prevent admission and be reported separately. Large intermediate cancellation MUST not bypass a declared limit merely because its final result is small.
- **FR-015**: Initial arithmetic qualification MUST include addition, subtraction, full multiplication and comparisons beyond 256 bits. Division, remainder and bitwise operations are outside the initial new profile; unsupported operations MUST be rejected. Existing signed-256 operations retain their original semantics. Candidates MUST access only the approved arithmetic interface, not allocator internals, host capabilities or another request's values.
- **FR-016**: Eligible-literal abstraction and parameter numbering MUST be deterministic and versioned, preserving repeated parameter identity and final-source provenance. Fitting MUST not parameterize compiler helper values, storage identifiers or structural control constants by default, nor manufacture a missing program completion. Unsupported parameter dependence MUST be diagnosed from semantics rather than silently treated as affine.
- **FR-017**: Each fit MUST declare its numerical profile, supported symbolic subset, maximum parameter count, coefficient-size/search bounds, escalation policy and aggregate attempt/time budget. The supported representation MUST allow coefficients beyond both ±1,000 and machine-integer range. Scoped unsatisfiability MUST identify the exact domain, bounds and replayable evidence; timeout, incomplete enumeration and unsupported operations MUST never be promoted to unsatisfiability. An unfinished uniqueness/optimization check MUST be reported as unfinished.
- **FR-018**: The default evaluation contract MUST expose 20 terms and score a total horizon of 100 terms where source data permit. Exact source offsets and missing terms MUST be preserved. Other horizons MUST be explicit experiment changes, never automatic responses to overflow or poor fit. Hidden values MUST remain inaccessible to generation, fitting, ranking, curriculum, replay and selection of model checkpoints.
- **FR-019**: Curriculum data MUST comprise short connected generic structures with multiple parameter variants, progressing through expressions, interacting values, meaningful bounded loops and conditions/compositions. Data preparation MUST record constant/identity proportions, duplicate behaviors and target-selection decisions; simplification MUST preserve exceptional behavior within the declared domain. Advancement and retained practice MUST be governed by frozen rules.
- **FR-020**: Literal-versus-parameterized experiments MUST match underlying structures, behaviors, splits, backbone/encoder configuration, seeds, initialization rules, optimizer and selection policy. Changed vocabulary, parameter counts, token lengths and realized compute MUST be disclosed. Search comparisons MUST report both rescue of the same decoded candidates and equal-total-budget portfolios; the latter MUST count generation, solving and verification rather than comparing solver-assisted search with only one raw attempt.
- **FR-021**: Reports MUST cover training examples, held-out parameters in familiar structures, held-out structures and frozen OEIS cohorts, with detectable duplicate/prefix/shift overlap and limitations disclosed. Report termination, legality, visible fit, independent continuation fit, diversity, fitting rescue/ambiguity, unsupported cases, time and resource coverage for every declared denominator. Incomplete token streams MUST be retained under bounded evidence policies.
- **FR-022**: Validation cadence, learning rate, stopping/patience, selection and advancement criteria MUST be explicit. Selection MUST prioritize declared synthetic synthesis competence with teacher-forced loss as supporting evidence. Selection/advancement MUST not gain an advantage from evaluating an easier subset after timeouts. The offset-only 32-example diagnostic MUST not qualify structural transfer or authorize a full-scale run.
- **FR-023**: Replay MUST accept only independently verified training-target programs, retaining original/grounded sources, exact assignments, numerical profile and source/track identity. Capacity, deduplication, replacement, contribution to updates and lifetime budgets MUST be explicit and resumable. Development/final target successes MUST not enter replay, even without their hidden values.
- **FR-024**: Generic-only experiments MUST start from random initialization or a recorded generic-only foundation descended from random initialization. The practical assisted branch MAY share that foundation, but additional demonstrations, imported programs or historical weights require a frozen reviewed manifest and assisted labelling of all descendants. Shared foundation compute MUST be reported. Named teachers, metadata-guided scaffolds and unaudited legacy replay remain excluded from strict results.
- **FR-025**: Demonstrations and exposed evaluation cases MUST undergo detectable overlap review. Reusing an exposed cohort MUST label it diagnostic; a revised generalization claim requires an unexposed frozen final cohort. Candidate archives MUST retain mathematical alternatives within declared limits separately from choosing concise training representatives; omission/deduplication decisions MUST remain auditable. Finite matches, conjectures, bounded verification, proof and novelty review MUST have separate evidence labels.
- **FR-026**: Encoder studies MUST separately measure post-training component ablations and matched retrained variants, stratified by visible integer size, target length and structural complexity. Mathematical streams and the exact-byte branch MUST have explicit, reproducible configurations. Reports MUST disclose parameter counts and measured throughput. Weight magnitude or one ablation MUST not be presented as universal feature usefulness, and exact serialization MUST not imply an injective floating-point embedding.
- **FR-027**: Supported workflows MUST not require sudo integration, lab-specific brokers/accounts, SSH, VM file synchronization or a runtime socket inside candidate containers. Requested GPU work MUST prove actual device updates without silent CPU fallback. CPU and GPU roles MUST retain explicit input mounts and hidden-data isolation. Persistent bulk downloads, datasets, caches, run state and reports MUST use the configured workspace; temporary storage MAY be disposable.
- **FR-028**: Every resumable run MUST identify input/track/ancestry, checkpoint, effective model/input/decoder settings, arithmetic/runtime/solver versions, fitting policy, splits, seeds, hardware and cumulative budgets. Checkpoints MUST restore optimizer, randomness, data order, curriculum, replay and selection state. Changing the effective contract requires a new experiment identity, not refunded time or silent in-place reinterpretation.
- **FR-029**: Analysis exports MUST reserve space for run contract, final/interrupted status, selected-checkpoint identity, aggregate split/size/complexity outcomes, raw/fitted comparisons and an omission manifest before optional traces. Too-small export limits MUST be rejected clearly before creating an apparently complete bundle. Selected exact examples, incomplete tokens, assignments and verification evidence MUST be exportable with hashes; weights and restricted split payloads MAY use separate explicit exports.
- **FR-030**: Solver retirement MUST include old fitting implementations, duplicated literal-edit grounding advertised as fitting, fallback algorithms, installed compatibility solvers and solver-only dependencies. Each active caller MUST migrate or be deliberately removed. A thin data-transport binding is permitted; a Python algorithm choosing/fitting trial assignments is not. Required independent verification/certificate checks MUST be migrated or separated from retired fitting code, not deleted to make cleanup pass. Regression fixtures for known false fits MUST test the replacement.
- **FR-031**: Cleanup MUST publish a disposition for inventoried commands, public imports, configurations, tests and examples. Supported controls and historical artifact readers MUST retain their declared semantics; obsolete executable paths MAY be reproduced from version-control history rather than shipped. No renamed copy or automatic fallback to the retired solver is allowed. Historical specs, provenance and saved results MUST not be rewritten as if produced by the replacement.

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
- **SC-007**: A frozen diverse memorization diagnostic contains at least 32 distinct prompts across at least four structures, at most eight offset-only examples and at least 16 non-affine-in-index examples. Its initial gate is at least 95% independently correct greedy-plus-fitting continuation with full coverage. Raw decoding and conditioning/template controls remain mandatory; this gate alone does not establish structural transfer.
- **SC-008**: Before a full-scale run, a matched short screen MUST show a positive conditioning advantage and improvement over the declared non-neural baseline on held-out structures under equal budgets and the predeclared coverage rule. The exact transfer margin, uncertainty reporting and coverage threshold MUST be frozen before training. Failure or insufficient coverage blocks promotion and is reported as such, rather than prompting an unrecorded threshold change.
- **SC-009**: Qualification reports MUST compare small-value execution with the retained numerical baseline and progressively larger operands, separately measuring solving, generation, verification and training time. Reports MUST include admitted/excluded data counts by size and reason, without claiming speed or coverage superiority in advance.

## Assumptions

- The user's acceptance of practical synthesis with a generic-only control and an assisted candidate is the explicit assisted-track decision required by constitution v2.0.0. It does not authorize assisted data in strict results or hidden-data leakage.
- Rust execution/fitting machinery and a WASM bigint runtime are user-requested constraints. Python remains for neural training and independent reference checking. Candidate dependencies are dashu-int behind a replaceable boundary and Z3 through Rust bindings; exact versions, interfaces and feasibility qualification belong to planning.
- Default comparison remains 20 visible terms and 100 total terms where source data exist. Previously inspected final cohorts remain diagnostic; revised experiments need an unexposed frozen final cohort.
- Fit existing concrete candidates before changing training targets. Preserve the small backbone and controlled hybrid/tri-stream inputs; widening input representation or vocabulary requires explicit new identities and compatibility checks.
- Initially exclude division/remainder/bitwise additions, parameter-dependent control fitting, unrestricted symbolic solving, GRPO, automatic completion repairs, floating-point fitting, privileged scaffolds and automatic proof/novelty claims.
- Persistent data use the configured workspace and pinned bulk OEIS sources. No sudo integration, lab broker/accounts, SSH or VM-specific dependencies are introduced.
- This specification defines required outcomes, not an implementation plan or evidence of runtime/GPU readiness. Resource defaults and experiment promotion thresholds must be frozen before execution.
- The four input groups for attribution are magnitude, modular, difference/valuation and exact-byte features. The actual grouping, finite-difference orders, primes, valuation limits and interaction settings must be identified in each encoder contract rather than inferred from an architecture name.
- Constitution v2.0.0 remains applicable to numerical semantics, independent acceptance, finite budgets and provenance. Its historical workstation description does not override the user's explicit portable Docker/Podman requirement. No constitution or upstream Spec Kit template is changed by this specification.
- The specification records user-imposed Rust/WASM and removal constraints in context; it intentionally leaves ABI signatures, memory-management strategy, library build/linking details and code layout to planning. A development-only transition is allowed, but feature completion requires full solver retirement and migrated call sites.
