# Feature Specification: Controlled Generalization Experiments

**Feature Branch**: `011-controlled-generalization`

**Created**: 2026-10-30

**Status**: Draft — ready for implementation planning

**Input**: Independently assess the supplied research reports and primary sources, integrate the completed pilot findings, and specify the next changes. Concentrate complex algorithm work into one phase and HIP validation into as few runs as possible, ideally one serial invocation with shareable results.

**Research decision**: Which of broader structural coverage, refreshed training examples, explicit parameterized targets, or verified training-only replay improves useful conditional exact synthesis at a declared training and inference cost?

The completed pilot supports retaining the repaired drain evaluation and earlier checkpoints. It does not establish encoder failure or a universal inability to generate constants. Its historical structural cohort contains no loops. [research-review.md](research-review.md) documents the primary-source assessment, corrections to the reports, code constraints and pilot evidence; [source-register.md](source-register.md) records reading scope and limitations.

This feature provides separate experiments and trustworthy evidence. A negative or inconclusive experiment is a valid outcome. It does not promise an accuracy improvement or treat software qualification as a research result.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Explain existing evidence before spending more compute (Priority: P1)

As the researcher, I can distinguish proposal quality, fitting, search coverage, conditioning and incomplete measurements, and evaluate retained checkpoints without retraining unrelated arms.

**Why this priority**: Aggregate fitted yield cannot explain the pilot's mechanism.

**Independent Test**: Analyze recorded outcomes and bounded retained-weight evaluations, producing reconciled fixed-denominator summaries without changing training or reading sealed assessment truth.

**Acceptance Scenarios**:

1. **Given** a historical public bundle, **when** a comparison is requested, **then** totals reconcile to its original denominator and unavailable source bodies or private weights are marked unavailable.
2. **Given** compatible best and latest weights, **when** only those checkpoints are selected, **then** both use identical registered prompts, attempt seeds and budgets; their identities are recorded and no training arm is implicitly rerun.
3. **Given** conditioning controls, **when** contributions are reported, **then** one-proposal raw/rescue measurements are distinguished from eight-attempt portfolios and every comparison includes completion bounds.
4. **Given** a prompt-independent proposal bank, **when** it is applied to multiple targets, **then** proposal identities are shared, each fitter sees only its target's visible prefix, and cold generation and amortized reuse costs are separate.

### User Story 2 - Measure broader generalization (Priority: P1)

As the researcher, I can freeze a new challenge benchmark exercising loops, branches, larger values and longer programs while retaining the historical cohort for comparison.

**Why this priority**: Broad learning claims need broader coverage than stage-A structural holdouts.

**Independent Test**: Prepare a small mechanical corpus with declared coverage and contamination counterexamples. Verify deterministic membership, coverage and fail-closed shortfalls before training.

**Acceptance Scenarios**:

1. **Given** feasible quotas, **when** preparation succeeds, **then** its manifest identifies covered stages, executed control flow, families, lengths and integer sizes, plus finite duplicate/prefix/shift screening results.
2. **Given** an impossible quota or exhausted budget, **when** preparation stops, **then** it reports the shortfall without borrowing rows, weakening novelty rules or relabeling historical data.
3. **Given** distinct training, selection and assessment roles, **when** unauthorized access or a duplicate group crossing their boundary is detected, **then** execution is rejected before training.

### User Story 3 - Compare data and target changes individually (Priority: P1)

As the operator, I can run selected paired experiments isolating refresh, structural support, target representation and checkpoint selection.

**Why this priority**: The literature motivates competing hypotheses, not a guaranteed replacement recipe.

**Independent Test**: Run bounded fixture experiments from a common initializer, audit each changing factor and saved state, and compare fixed-exposure checkpoints.

**Acceptance Scenarios**:

1. **Given** static-versus-refreshed training, **when** examples change, **then** structural support and sampling weights stay fixed and all accepted examples pass frozen exclusions.
2. **Given** paired literal/parameterized targets, **when** they are prepared, **then** both represent the same concrete behavior, control-sensitive literals stay protected, and unsupported cases remain counted.
3. **Given** fixed milestones and separate loss/behavior selection rules, **when** checkpoints are selected, **then** the same candidate set is considered, assessment truth is inaccessible, and best/latest/selected identities stay distinct.
4. **Given** an interrupted run, **when** it resumes, **then** learning state, random state, example order, replay and consumed budgets are restored; weights alone cannot serve as a resume source.

### User Story 4 - Test bounded verified replay (Priority: P2)

As the researcher, I can distill verified alternatives discovered on training tasks and compare against continued supervised training from the same checkpoint.

**Why this priority**: Search-and-learn papers justify testing this route, with separate costs and information boundaries.

**Independent Test**: Exercise one bounded round with correct, incorrect, duplicate, over-budget and prohibited-source candidates; admission must be correct independently of accuracy gains.

**Acceptance Scenarios**:

1. **Given** a candidate matching a training prefix but failing its independent training audit, **when** admission is attempted, **then** it is rejected with the failed check recorded.
2. **Given** several alternatives for an easy task, **when** replay is sampled, **then** caps and task-first sampling prevent alternative count from silently increasing the task's weight.
3. **Given** selection, assessment or OEIS-derived evidence, **when** offered to replay, **then** it is rejected even if successful.
4. **Given** a replay comparison, **when** results are packaged, **then** current-policy performance, cumulative discoveries, training cost and added search/verification cost are separate.

### User Story 5 - Validate and share in one hardware session (Priority: P1)

As the workstation operator, I can invoke one serial workflow performing qualification, exercising the new capabilities and producing a compact evidence bundle.

**Why this priority**: Repeated qualification consumed time in the pilot and complicates evidence reconciliation.

**Independent Test**: Check orchestration with bounded failure fixtures, then perform one real HIP acceptance invocation on an unchanged compatible environment.

**Acceptance Scenarios**:

1. **Given** successful preflight, **when** the workflow runs, **then** qualification occurs once, canonical commands reuse its verified receipt, and the bundle identifies every required stage/result.
2. **Given** a changed source, image, acceptance contract or hardware identity, **when** reuse is attempted, **then** stale evidence is rejected and the smallest safe requalification scope is reported.
3. **Given** a stage failure, **when** execution terminates, **then** a partial bundle, bounded log tail and precise restart guidance are produced without a false overall pass.
4. **Given** a purported HIP pass containing only CPU evidence, **when** the bundle is checked, **then** acceptance fails.

### Edge Cases

- Infeasible coverage cells, single-family strata, inactive control-flow syntax and generation-budget exhaustion must remain distinguishable.
- Different programs may agree on all checked terms. Finite fingerprints do not prove equivalence or uniqueness.
- Parameter dependencies cross assignments and loop iterations. Repeated multipliers need not be affine; equal literal values need not share a parameter.
- Shorter target representations change examples and updates per token budget. All three budgets cannot simply be called matched.
- Canonical, fitted and target-family agreement can disagree. Canonical mismatch alone is not semantic failure.
- Timeouts, queue starvation, unsupported fitting, inconsistent systems and incomplete generation are distinct; missing measurements are not silently dropped.
- Private latest weights may be absent. Intermediate checkpoints cannot be reconstructed from loss logs.
- Empty replay yield must not trigger weaker verification, extra budgets or assessment-label access.
- Interrupted exports and stale qualification receipts must fail integrity checks. Changed semantics create new evidence identities rather than overwriting results.

## Requirements *(mandatory)*

### Functional Requirements

#### Experiment identity and permitted information

- **FR-001**: Each experiment MUST freeze its hypothesis, primary changing factor, paired baseline, seeds, cohorts, model/precision, target format, sampling rules, checkpoint candidates, budgets, selection/stopping rules and decision criteria before execution. Interaction experiments MUST name every changing factor.
- **FR-002**: The strict track MUST retain random-start provenance and mechanically generated generic-program supervision. Imported OEIS/LODA programs, named sequence teachers, externally pretrained weights and unaudited replay MUST be rejected. Historical manifests/recipes MUST remain immutable; new capabilities require a prospective version.
- **FR-003**: Training, selection supervision, historical exposed evaluation and sealed assessment MUST have distinct identities and access policies. Selection supervision MUST be designated before generation and MUST NOT be described as hidden evaluation. Assessment continuations, target programs and protected family/stratum labels MUST be inaccessible to training, generation, fitting, replay and all selection decisions; assessment scores only frozen selections. Only registered visible inputs may reach candidate generation/fitting. Selection targets MUST NOT enter training or replay.
- **FR-004**: Accepted/transformed programs MUST preserve exact integers and source indices under declared bounded semantics. Final-source validation, production execution and independent exact verification MUST precede acceptance. Finite 100-term agreement MUST NOT be labeled proof.

#### Evidence and matched diagnostics

- **FR-005**: Researchers MUST be able to analyze existing evidence without training and independently select compatible retained checkpoints or training arms. Missing artifacts MUST produce explicit unavailability, not replacement weights or fabricated measurements.
- **FR-006**: Each arm/split/checkpoint MUST report successes, resolved failures, unresolved outcomes, fixed denominator, completion fraction and lower/upper success bounds. Reports MUST include row-weighted and family-macro results, family counts and paired outcomes. Completion bounds and statistical uncertainty MUST be distinct; seed variability cannot be inferred from rows alone.
- **FR-007**: Historical comparisons MUST preserve correct-input, shuffled-memory, synthetic-memory absent-input and fixed-template controls, plus raw/rescue diagnostics. New comparisons MUST freeze identical attempt seeds and applicable inference budgets, describe the intervention precisely, and preserve the correct visible prefix for fitting and candidate selection.
- **FR-008**: New evidence MUST identify pre-fit proposals, parsed structure/family signatures, length, control-flow attributes, parameter count, duplicates and terminal outcome for all attempts, including failures. First-error diagnostics and post-fit outcomes MUST join by immutable identity. Target-relative assessment diagnostics belong only to scoring after selections are frozen. Historical unavailable fields MUST remain unavailable; structural similarity is not correctness.
- **FR-009**: Fitting evidence MUST expose actual phase, outcome/reason, bounds and measured queue/service/cleanup costs where supported. Unsupported, nonlinear, inconsistent and resource-exhausted outcomes MUST be distinguishable. Reports MUST not invent continuous-optimizer diagnostics or infer a unique timeout cause without evidence.
- **FR-010**: Operators MUST be able to create/reuse a frozen prompt-independent proposal bank. Generation MUST depend only on declared model, absent-input intervention and seeds, never recipient targets. Each target receives the same registered proposals/order and cap; failed/duplicate draws remain charged. Cold costs and any amortization denominator MUST be explicit.
- **FR-011**: Offline proposal replay and oracle-union analysis MUST preserve candidate identities and declared budgets, use only visible evidence for executable selection, and distinguish retrospective oracle coverage from deployable performance. A mixed portfolio MUST share one total attempt/time budget across constituents.

#### Broader, separately frozen benchmark

- **FR-012**: The historical cohort MUST remain reproducible and separately reported. A challenge cohort MUST have a new version, deterministic construction and explicit size/family/coverage commitments; results MUST not be spliced into the historical denominator.
- **FR-013**: Challenge preparation MUST freeze feasible marginal quotas for supported stages, families, lengths, integer sizes and active loops/branches. It MUST include later-stage programs, values above 64 bits, programs longer than the historical maximum, and multiple independently generated control-flow families. Syntax-only control flow MUST not satisfy execution quotas. Counts and distribution differences MUST be published without pretending every factor combination is feasible.
- **FR-014**: Parameter holdouts MUST reserve numeric variants within declared training families; structural holdouts MUST reserve complete structural families under a frozen identity rule. Training, selection and assessment MUST exclude cross-boundary detectable duplicates and declared prefix/shift groups. Novelty claims MUST state finite-observation limits and measured operator/composition overlap.
- **FR-015**: Quota/isolation shortfalls MUST stop preparation with counts and reasons. Silent redistribution, reduced denominators, fallback to simpler examples or use of assessment results to repair the cohort is prohibited. Revision requires a new manifest before execution.
- **FR-016**: Refreshed/replay data MUST pass frozen exclusions. A preparation custodian MAY use protected data solely to enforce isolation; learner-facing output MUST reveal only admission decisions/identities, not hidden terms, target programs or optimization hints. Generation/verification/rejection costs and rejection categories MUST be recorded.

#### Isolated learning and selection comparisons

- **FR-017**: Independently selectable comparisons MUST cover (a) static versus refreshed variants within fixed structural support, (b) broader support at fixed admitted-example budget and declared coverage/sampling policy, (c) literal versus supported parameterized targets, and (d) replay versus continued supervision. Coverage/family-weight changes MUST be separately named factors. No full Cartesian sweep is required.
- **FR-018**: Within-support refresh MUST preserve family pool, mixture and numeric sampling distribution while changing accepted instances. It MUST be bounded/reproducible, record unique examples/families and exposures, and restore data order on resume. Exhaustion MUST not silently switch to static data or introduce families.
- **FR-019**: Target comparisons MUST use paired concrete programs and independently verify grounding round trips. Whole-program control dependence, unsupported cases and parameter-count bounds MUST remain enforced. Control-sensitive literals remain literal; equal numeric values MUST not be tied merely because they coincide.
- **FR-020**: Both representation arms MUST use the same declared exact-fitting policy for the main comparison, with raw behavior separate. Unsupported rows MUST retain their stratum and explicit disposition. Fitted-versus-unfitted advantage MUST not be attributed solely to representation.
- **FR-021**: Paired seeds MUST share initialization, model and nonexperimental settings. One primary training budget MUST be declared; default is processed nonpadding target tokens. Actual examples, updates, unique-source exposure, learner/generation/verification time MUST also be reported. Token undershoot MUST be bounded by one prospective batch and recorded.
- **FR-022**: Best-loss, latest, fixed-exposure and behavior-selected checkpoint identities MUST remain separately available. Candidate milestones and tie-breaks MUST be frozen. Semantic selection MUST use only FR-003 selection supervision; assessment MUST never rank checkpoints or candidates.
- **FR-023**: Default behavior selection MUST require at least 95% completion on each primary selection split, maximize structural family-macro confirmed success, then parameter family-macro confirmed success, then prefer the earlier checkpoint. If none qualifies, return inconclusive selection and retain the baseline. Selection-method comparisons MUST consider the same retained candidate set; historical continuous best-loss selection stays separately labeled.
- **FR-024**: Runs MUST stop at frozen token/work/wall/disk budgets and record why. Resume MUST restore optimizer, scheduler, random state, refresh cursor, replay, checkpoint history and consumed budgets at a consistent update boundary. Weights-only artifacts may initialize an explicitly new run but MUST NOT masquerade as resume.

#### One bounded training-only search-and-distill round

- **FR-025**: Replay MUST be independently enabled. Default scope is one bounded search-and-distill round with a paired continued-supervision arm from a common frozen initializer. Search tasks MUST be mechanically generated training tasks disjoint from selection/assessment identities.
- **FR-026**: Admission MUST require supported final source, exact prefix agreement and full independent training-audit agreement against the training oracle, including production/reference agreement. Default audit covers all 100 registered training terms. Prefix-only matches, timeouts, disagreements and forbidden provenance MUST be rejected. Training verification is not unseen assessment performance.
- **FR-027**: Replay MUST retain provenance, original proposal, fitted assignment, final source, verification scope, origin checkpoint and admission reason. Original mechanical targets remain available. Default policy admits strictly shorter verified alternatives under a frozen target-length measure, at most two per task; broader policies require separately registered comparisons.
- **FR-028**: Replay MUST enforce capacity, per-task limits, maximum batch fraction and deterministic eviction/sampling. Defaults are 4,096 alternatives and at most 25% replay examples per update, sampled task-first so alternative count does not multiply task weight. Duplicate removal MUST distinguish source identity from finite behavior; it cannot claim unproved semantic equivalence.
- **FR-029**: Search MUST have frozen task/attempt/time/storage budgets. Empty admission yield is valid, without automatic increases or weaker verification. Replay/control optimization budgets MUST match and added search costs remain explicit. Compute-efficiency claims additionally require registered total-cost-matched comparisons.

#### Decisions and evidence delivery

- **FR-030**: Confirmatory learning comparisons MUST use at least three paired seeds, freeze one primary outcome, report all seeds and retain unsuccessful/inconclusive runs. Single-seed screens stay exploratory. Assessment occurs after selection is frozen; adaptive tuning afterwards requires fresh assessment or an exploratory label.
- **FR-031**: Conditioning-improvement claims MUST compare matched correct-input/control portfolios with bounds and costs, rather than raw-to-fitted gains. Broad-generalization claims MUST report challenge strata, not just historical aggregate yield.
- **FR-032**: A decision rule MUST freeze effect size, nonregression guard, completion floor and uncertainty treatment. The default promotion screen requires a mean structural family-macro gain of at least 5 percentage points across three paired seeds, no seed with a negative observed structural difference, no more than a 5-point mean row-weighted parameter-yield decline, and at least 95% completion per primary split/seed. If unresolved-outcome bounds cannot establish these conditions, the result is inconclusive. Passing is not a population-level statistical guarantee.
- **FR-033**: Sharing bundles MUST contain experiment/qualification identities, commands/order, checkpoint/data/source hashes, requested/actual budgets, attempt/family results, selection decisions, costs, bounds, failures and scoped interpretation. Inventories MUST verify payload size/hash and list intentional omissions. Private optimizer state and sealed truth MUST be excluded.
- **FR-034**: Software acceptance and scientific outcome MUST be separate statuses. A passing workflow with censored measurements or no gain MUST not claim scientific success. Negative research results do not fail otherwise correct software acceptance.

### Non-Functional Requirements

- **NFR-001 — One complex-algorithm phase**: The plan/task list MUST put all complex algorithm design and implementation in **one contiguous, explicitly named phase**, suitable for a more capable model. This covers split/coverage selection, refresh/exclusions, parameter safety changes, replay admission/sampling, proposal-bank accounting, checkpoint selection and statistical/completion calculations. Algorithm-specific reference cases and tests belong in this phase. Routine scaffolding before and integration after MUST not hide additional algorithm work.
- **NFR-002 — Reviewable phase boundaries**: Contracts, inputs/outputs, resource limits, invariants and counterexamples MUST precede that phase. Its exit MUST establish tested algorithms behind stable contracts. Later algorithm defects MUST reopen this same phase and relevant tests, with the change recorded, rather than create scattered algorithm phases. Model choice remains the operator's decision.
- **NFR-003 — One normal HIP invocation**: On an unchanged compatible environment, complete hardware acceptance MUST require **one operator invocation**, run serially through canonical project CLI endpoints, and package evidence automatically. A temporary orchestration script is acceptable. It MUST orchestrate acceptance contracts rather than duplicate them. The normal path performs required CPU/native and HIP qualification once and reuses valid evidence downstream.
- **NFR-004 — Safe qualification reuse**: Canonical endpoints MUST verify receipt integrity and compatibility with the complete qualification closure: relevant source/build dependencies, container/runtime, hardware, precision and effective acceptance contract. Helper-level skip flags or stage numbers are not proof. Incompatibility MUST invalidate reuse and identify the smallest safe rerun. Failures or changed closures may require another invocation; every repeat records why and preserves prior evidence.
- **NFR-005 — Bounded hardware scope**: Complete appropriate local contract, arithmetic, isolation, resume and failure tests before the hardware session. One HIP session MUST demonstrate real checkpoint loading, finite forward/backward updates, and representative refresh, both target formats, replay, resume and export under small fixed budgets. Full 25M-token experiments, all seeds and all research combinations MUST not be qualification prerequisites. CPU fallback cannot satisfy a HIP gate.
- **NFR-006 — Workstation and containment**: Respect the project's single-learner workstation, portable container workflow and bounded execution. Aggregate memory, queues, workers, caches, checkpoints, logs, disk and wall budgets MUST be explicit/enforced. No host driver changes, new host services or unbounded online generation are required.
- **NFR-007 — Failure and sharing**: Preflight MUST report stages, identities, budgets and required artifacts before costly work. Every terminal path MUST preserve a machine-readable ledger and verified complete bundle or explicitly partial bundle with bounded log tail and restart guidance. Reusing completed stages requires validated identities/artifacts; learner restart obeys FR-024.
- **NFR-008 — Compatibility and reproducibility**: Historical commands/recipes/evidence MUST retain their meaning. Identical preparation inputs reproduce membership/target identities. Hardware timing variation is measured, not mistaken for determinism. No speed or HIP-readiness claim is accepted without actual target-hardware evidence.

### Key Entities *(include if feature involves data)*

- **Experiment contract**: Question, intervention, baseline, paired seeds, budgets, checkpoint candidates, selection and interpretation rules.
- **Cohort manifest**: Role/version, sources/families, quotas/coverage, exclusions, visible inputs and protected scoring scope.
- **Target pair**: Concrete source, literal/parameterized representations, protected constants, grounding and independent agreement evidence.
- **Proposal record/bank**: Generated source/structure, conditioning intervention, seed, bank membership, charged costs and outcomes.
- **Checkpoint record**: Weight identity, exposure, lineage, complete resume state or explicit weights-only status, selection eligibility/reason.
- **Replay entry/round**: Training task, alternative, origin, bounded retention/sampling status, audit evidence and search cost.
- **Qualification receipt**: Actual device/test evidence and compatibility identity for canonical dependent stages.
- **Evidence bundle**: Auditable inventory, stage ledger, measurements, omissions and conclusions, excluding private state and sealed truth.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An analyst can reproduce 100% of aggregate counts from included row/attempt evidence, with all missing/unresolved measurements represented and no changing denominators.
- **SC-002**: Preparation satisfies every frozen coverage/isolation commitment or stops with an explicit shortfall; no tested leakage, unsupported substitution or silent fallback is accepted.
- **SC-003**: Each intervention can run independently, best/latest/fixed-exposure checkpoints can be evaluated without unrelated retraining, and paired initialization/preparation identities are reproducible.
- **SC-004**: Known counterexamples for control-sensitive parameters, false prefix matches, prohibited replay, stale receipts and incomplete resume state are rejected; valid fixtures are admitted with independent evidence.
- **SC-005**: Completed comparisons report all seeds, matched-budget conditions, additional costs, selection reasons and a justified positive/negative/inconclusive decision. Feature acceptance requires no minimum learning gain.
- **SC-006**: The successful unchanged-environment path needs exactly one serial hardware-validation invocation, qualifies once per required scope and produces a sharing bundle passing integrity checks.
- **SC-007**: Every injected stage failure produces a partial result identifying the failure/restart requirements; none yields an overall pass. Real accelerator evidence remains distinct from local checks.
- **SC-008**: The plan contains exactly one complex-algorithm phase, with every algorithm-bearing task/reference test mapped to it. Other phases cover contracts/scaffolding, integration and consolidated validation/delivery.

## Assumptions

- Constitution 2.0.0 governs this feature; no amendment is proposed. Planning resolves concrete contracts and feasibility budgets without silently relaxing requirements.
- The audited 010 follow-up is the historical baseline. Public results are exploratory; private checkpoints require availability checks. No new HIP run or original-archive re-audit was performed for this specification.
- Default research scale is 25M processed nonpadding target tokens per fresh arm, paired seeds 20261016/20261017/20261018, and milestones 4M/8M/16M/25M. Save the last consistent update boundary at or below each cap and record actual exposure. Qualification fixtures use separate small nonresearch budgets.
- Default inference retains 20 visible terms, 100 total terms, eight charged portfolio attempts, 60 seconds per target portfolio and the repaired drain policy. Raw/rescue stay one-proposal diagnostics. Budget changes require a new comparison.
- The challenge defaults to 128 parameter and 128 structural rows for each separate selection/assessment role, subject to bounded feasibility checking and frozen quotas. Size changes create new manifests; these counts are planning defaults, not existing achieved coverage.
- Completion floors, replay caps, milestones and promotion thresholds are engineering defaults, not literature-established effects. Different rules must be frozen before assessment results are observed.
- Non-goals: model architecture changes, new language/library primitives, pretrained teachers, learned correctness verifiers, new continuous/HNF/recurrence solvers, MCTS/grammar-alignment redesign, policy-gradient/GRPO training, and automatic longer unchanged runs. Each requires separate justification.
- This deliverable is a specification. Implementation planning is next; research execution remains subject to existing qualification and run-governance gates.
