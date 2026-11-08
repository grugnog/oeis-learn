# Feature Specification: Progressive Mathematical Discovery

**Feature Branch**: `013-progressive-mathematical-discovery`

**Created**: 2026-11-08

**Status**: Draft — requirements complete; implementation, qualification and research outcomes pending

**Input**: Following the merge of PR #63, specify the entire five-stage pathway: establish capability and language coverage; diversify verified training; tune learning settings and model size; progressively scale training and introduce verified OEIS self-learning; learn reusable abstractions and assess mathematical discoveries. Multiple training tasks of up to 24 hours each are permitted.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Know what the foundation can learn and express (Priority: P1)

As the researcher, I can distinguish dependable execution, learned capability, limits of the available mathematical language, and missing evidence before investing in larger experiments.

**Why this priority**: Prior engineering repairs passed qualification, but structural gains are concentrated, templates remain competitive, and longer historical training showed memorization. A reliable executable result is necessary but does not establish broad learning.

**Independent Test**: Evaluate retained checkpoints and conventional baselines on a frozen capability panel; replay independent language witnesses and controlled failures. This delivers a useful readiness report without any new training.

**Acceptance Scenarios**:

1. **Given** qualified retained D weights, **when** the panel is evaluated, **then** the report separately shows parameter transfer, composition transfer, unseen program structures, OEIS transfer, completion and cost, with templates and conditioning controls under declared budgets.
2. **Given** representative harder tasks with independently supplied solution witnesses, **when** language coverage is audited, **then** each task is classified as expressible within the declared limits, expressible only outside them, or unresolved; failed neural search alone never establishes inexpressibility.
3. **Given** an extension motivated by a coverage gap, **when** its acceptance runs, **then** exact semantics, independent positive/negative examples and unchanged-language regressions pass before it enters a learning experiment.
4. **Given** an already exposed sequence, related program structure or shifted duplicate, **when** a fresh assessment cohort is assembled, **then** the related group is excluded or explicitly marked exposed; a new seed does not restore freshness.

---

### User Story 2 - Learn from new mathematical structures (Priority: P1)

As the researcher, I can train on a reproducible supply of diverse verified problems, compare it with the existing static corpus, and see whether new experience improves transfer and preserves earlier skills.

**Why this priority**: Increasing exposure to nearly identical examples may increase confidence without increasing mathematical capability.

**Independent Test**: Generate and independently verify a bounded corpus, recover its exact continuation after interruption, and run a matched static-versus-refreshed pilot with the same model and allocation policy.

**Acceptance Scenarios**:

1. **Given** the same generation commitment and spent-work journal, **when** preparation resumes, **then** accepted examples, rejected draws and subsequent ordering reproduce without restoring consumed work.
2. **Given** a large set of new constants but few new structures, **when** diversity is reported, **then** structural diversity remains explicitly small and the run cannot claim expanded structural support.
3. **Given** matched static and refreshed runs, **when** their pilot completes, **then** data diversity is the declared intervention; initialization, language, allocation, scientific evaluation budget and token accounting are matched, and preparation costs remain visible.
4. **Given** frontier tasks repeatedly fail, **when** the curriculum advances, **then** the frozen exploration and retention policy applies without importing development or assessment solutions into training.

---

### User Story 3 - Choose learning settings and model size using evidence (Priority: P2)

As the operator, I can run a bounded tuning study, stop unpromising trials, extend promising ones and select a practical model using held-out capability and workstation cost.

**Why this priority**: Many existing settings are initial choices, but tuning against a small repeatedly observed panel would produce misleading winners.

**Independent Test**: Execute a small study containing completed, pruned, failed and resumed trials, then reproduce its selection from the recorded results and rules. Fixtures demonstrate orchestration; actual learning trials establish research evidence.

**Acceptance Scenarios**:

1. **Given** frozen data and development panels, **when** learning settings are searched, **then** every trial records its effective settings, initialization, exposure, cost and termination reason, including failed and pruned trials.
2. **Given** model sizes with different training speeds, **when** candidates are ranked, **then** equal-token capability and equal-wall-time utility are reported separately, and model-load, inference and verification costs are disclosed.
3. **Given** a slow-starting configuration, **when** early pruning is considered, **then** a declared grace period and reserved unpruned control trials protect the study against untested short-run ranking assumptions.
4. **Given** finalists with conflicting seed results or inadequate completion, **when** selection runs, **then** it returns an inconclusive decision or follows a predeclared bounded tie-break; it never silently launches unlimited replacement trials.

---

### User Story 4 - Progress through longer recoverable training (Priority: P2)

As the workstation operator, I can run a sequence of longer learning tasks, each bounded by 24 hours, with automatic evidence-based progression inside a frozen campaign and clear recovery when interrupted.

**Why this priority**: Longer runs are useful only when gains, total cost and the ability to recover are measured.

**Independent Test**: A shortened production-path rehearsal crosses a checkpoint, interruption, continuation, milestone decision and final export; a real long-task acceptance separately measures sustained device behavior.

**Acceptance Scenarios**:

1. **Given** an eligible configuration and a declared campaign, **when** the next stage begins, **then** it uses the remaining campaign allowance and its registered checkpoint lineage, rather than silently starting a fresh budget.
2. **Given** a task approaching its 24-hour ceiling, **when** training must stop, **then** reserved time permits checkpointing and reporting; a shortfall against its token milestone is reported as partial, not completed research.
3. **Given** a valid checkpoint and a crash, **when** recovery occurs, **then** model, optimizer, allocation, generator, replay, random state and spent/discarded exposure reconcile before any further update.
4. **Given** improving training loss but stagnant or degrading held-out capability, **when** a milestone is reviewed, **then** the declared rule stops or redirects the campaign; it does not automatically buy another longer run.
5. **Given** an inconclusive gain and intact integrity, **when** a pre-reserved diagnostic branch is eligible, **then** it consumes only its declared allowance, records the uncertainty it investigates and leaves the original gate unpassed; without that branch it stops with a proposal.

---

### User Story 5 - Learn from verified OEIS discoveries (Priority: P2)

As the researcher, I can measure the yield of searching an OEIS training pool, learn from useful verified candidates, and test whether this improves solutions on other problems.

**Why this priority**: Real-sequence learning requires trustworthy targets and enough discoveries to sustain learning; known-term fitting alone is insufficient.

**Independent Test**: On a bounded designated training pool, reject prefix-only overfits, admit independently verified candidates, complete one search-and-learning cycle, and compare with the starting checkpoint and continued synthetic training on separate evaluation tasks.

**Acceptance Scenarios**:

1. **Given** a candidate matching the visible prefix but failing reserved training continuation terms, **when** admission runs, **then** it is rejected and the failed search remains charged.
2. **Given** accepted discoveries, **when** the learner updates, **then** every target has source, verification and membership evidence; no development or assessment solution enters the replay collection.
3. **Given** insufficient verified discoveries, **when** the pilot gate runs, **then** it reports measured yield and the deficit and stops OEIS self-learning expansion instead of recycling a handful of examples as new diversity.
4. **Given** repeated verification against the same training continuation terms, **when** results are reported, **then** those terms are identified as training feedback, and transfer is assessed on different sequences.
5. **Given** a successful search-and-learning cycle, **when** the next cycle is considered, **then** promotion depends on gains on other problems and retention, not solely an increasing number of solutions in the training archive.

---

### User Story 6 - Accumulate reusable methods and assess discoveries (Priority: P3)

As the researcher, I can extract reusable subprograms from verified training solutions, test whether they help solve harder unseen tasks, and review candidate discoveries with honest evidence and novelty labels.

**Why this priority**: Compositional knowledge and useful search may enable progress beyond repeatedly enlarging a fixed model.

**Independent Test**: Extract a component from a controlled training archive, independently validate it, compare bounded search with and without it, and produce a discovery dossier including a deliberately refuted conjecture.

**Acceptance Scenarios**:

1. **Given** a recurring construction in training solutions, **when** a reusable component is proposed, **then** its assumptions, numeric semantics, dependency lineage and expanded behavior are checked before adoption.
2. **Given** component-assisted successes, **when** improvement is measured, **then** an otherwise matched baseline receives comparable total search resources, and hidden expansion/execution costs are charged.
3. **Given** a program agreeing on all tested terms, **when** it is published for review, **then** it is labeled finite agreement unless stronger evidence exists; neither model generation nor a different source string establishes novelty.
4. **Given** a claimed theorem or algorithmic improvement, **when** review occurs, **then** the claim has explicit scope, comparison with known work, independent checks and, for a proved label, a replayable sound certificate or independently checked proof.

---

### User Story 7 - Add external mathematical data deliberately (Priority: P3)

As the researcher, I can decide whether an additional mathematical database provides useful new tasks, introduce a bounded source pilot, and measure its marginal benefit before expanding acquisition or training.

**Why this priority**: More records are valuable only when they add meaningful, verifiable problems compatible with the selected learning task.

**Independent Test**: Admit a small source snapshot, preserve its exact values and provenance, transform it into well-defined tasks, and compare a matched training mixture with and without that source.

**Acceptance Scenarios**:

1. **Given** combinatorial object–statistic records, **when** counting/distribution tasks are constructed, **then** the indexing, transformation and independent term generator are recorded; arbitrary file order is not treated as a mathematical sequence.
2. **Given** non-integral coefficients, approximate zero locations or finite polynomial data, **when** ingestion runs, **then** the source is rejected for the integer-continuation profile or assigned a separately declared compatible task; values are never silently rounded, flattened or padded into apparent discoveries.
3. **Given** related objects appearing in different databases, **when** splits are frozen, **then** detected cross-source relationships and duplicates remain grouped and their detection limits are disclosed.

### Edge Cases

- A useful result misses a numerical promotion threshold or shows cheaper execution without an accuracy gain: preserve the original verdict; only a predeclared utility or exploratory route can authorize a different action.
- Sparse fresh support cannot satisfy frozen strata: report availability and stop scoring; any revised cohort is a new prospective version, never a silent reseed or quota reduction.
- Multiple rules fit the visible prefix: retain ambiguity; verify finite continuation and do not assert uniqueness.
- Generation never terminates, arithmetic exceeds its numeric profile, fitting times out, or the service observes a result after its deadline: preserve distinct outcomes and full denominators; do not convert unknowns into mathematical UNSAT.
- A new primitive changes the set of representable programs or solver support: create a new language profile and disclose the changed prior and unsupported fitting cases.
- Refresh produces mostly duplicates, trivial programs, or expensive rejections: charge all generation work and stop at the frozen admission budget with actual diversity counts.
- Trials overfit frequently reused development data: preserve exposure history and use an independent confirmation set before final assessment.
- A 24-hour interruption occurs during discovery admission or library publication: recover an atomic consistent model/data/library version without double admission or erased search costs.
- A changed architecture or language makes a checkpoint incompatible: reject continuation; explicitly declared weights-only initialization is a new experiment, not exact resume.
- A learned component looks shorter only because it hides a large program: account for expanded complexity and execution cost as well as surface length.
- A source has too few continuation terms, invalid offsets, unavailable licensing terms or a missing immutable snapshot: classify it ineligible with a reason before training.
- A refuted candidate has already entered training: invalidate affected evidence, identify dependent targets/components/checkpoints, quarantine the lineage and restart from the last clean compatible state or label it contaminated.
- Hardware sensors are unavailable or device/resource settings change: report missing measurements; require compatible qualification for execution rather than inventing utilization or reusing unrelated evidence.

## Requirements *(mandatory)*

### Functional Requirements

#### Scope, research decisions and information boundaries

- **FR-001**: The feature MUST implement all five stages, including usable OEIS self-learning, reusable-component learning and discovery review. Completing the initial tuning pilot alone MUST NOT close the feature. Each later stage MUST have bounded acceptance independent of achieving a positive research result.
- **FR-002**: Every study MUST freeze its question, parent evidence, data and exposure identities, language, model/initialization, baselines, seeds, budgets, metrics, selection/stopping rules and permitted transitions before outcomes are observed. Missing required fields MUST block research execution.
- **FR-003**: D MUST be the initial working allocation recipe and retained checkpoint reference, without an asserted universal win. A full new B/D replication MUST NOT be a prerequisite. Uniform allocation MAY be included only as a separately budgeted control answering a concrete question.
- **FR-004**: Studies MUST distinguish synthetic-only learning from OEIS-derived or externally assisted learning. The synthetic-only track retains random initialization and mechanical demonstrations. Later OEIS learning MUST have a distinct profile, checkpoint lineage and reports; its results MUST NOT be relabeled strict synthetic-only evidence.
- **FR-005**: Learning, tuning/development, independent confirmation and final assessment memberships MUST be distinct and versioned. Detectable structural, duplicate, prefix, shift and cross-source relation groups MUST stay together as appropriate to each endpoint. Previously exposed cohorts, including spec-012 fresh-002, MUST remain exposed. Reports MUST state residual leakage-detection limitations.
- **FR-006**: Assessment truth and results MUST remain unavailable to training, candidate generation, admission, curriculum allocation, tuning and checkpoint/library selection. Assessment execution MUST require the designated final model and frozen evaluation contract. Any later reuse MUST be labeled exposed.

#### Stage 1 — Capability and representation

- **FR-007**: The capability suite MUST separately measure new parameters, new compositions, unseen program structures and OEIS transfer; it MUST stratify harder tasks by structural complexity, required operations and execution demands. Strata, counts, feasibility and uncertainty procedures MUST be frozen before scoring. Unavailable strata MUST remain unavailable rather than disappear from aggregates.
- **FR-008**: Evaluation MUST use actual identified checkpoints, independently executed exact values, fixed visible inputs, scoring horizons and search allowances. It MUST report all-task verified yield, family/stratum aggregates, paired outcomes, completion/censoring, token-cap and service failures, program complexity and measured costs. Canonical-target token accuracy alone MUST NOT establish learning.
- **FR-009**: Baselines MUST include the starting checkpoint, generic templates, shuffled/absent conditioning and declared conventional methods relevant to each stratum. Initialization or earlier-checkpoint controls MUST isolate learned contribution where compatible. A learned-plus-template portfolio MUST freeze its visible-only selection and total budget before evaluation; observed success unions are diagnostic only.
- **FR-010**: A language-coverage audit MUST use independent constructive witnesses and classify representation and budget limitations separately from search failure. Proposed extensions MUST be motivated by documented gaps, have exact edge-case semantics, and preserve independently checkable acceptance. Division/remainder, bounded memory and reusable calls are candidates for audit, not silently adopted defaults.
- **FR-011**: New language or primitive profiles MUST qualify numerical behavior, grammar completion, transformations, fitting compatibility, containment and actual device learning before production use. Unsupported fitting MUST remain explicit. The qualified drain schedule is the initial execution baseline; alternative scheduling MUST demonstrate completion and behavior effects as well as speed before adoption.

#### Stage 2 — Diverse verified training

- **FR-012**: Training preparation MUST support both the retained static support and reproducible refreshed support, with immutable draw commitments, independent validation, rejected-draw accounting, consumed-work records and recovery of generation order.
- **FR-013**: Diversity reporting MUST distinguish numeric variants, canonical structures, behavioral equivalence under tested inputs, compositions, operation coverage and difficulty bands. Finite behavioral agreement MUST NOT be presented as all-domain equivalence. Increasing only constant variants MUST NOT count as new structural diversity.
- **FR-014**: Curriculum rules MUST maintain declared familiar-task retention, frontier exploration and diversity floors using training-only feedback. Definitions of difficulty, allocation, plateau, retention and admission limits MUST be fixed before a study; harder tasks MUST NOT be defined solely by integer magnitude.
- **FR-015**: A matched static-versus-refreshed pilot MUST hold initialization, model, language, allocation recipe and evaluation allowances fixed at matched charged-token milestones. Reports MUST additionally disclose generation/verification costs and equal-wall-time utility. If another factor changes, it MUST be declared as a separate intervention.
- **FR-016**: Training targets MUST be independently valid programs with exact source values, declared semantics and membership provenance. Where multiple verified targets are retained, their sampling and weighting MUST be declared and preserve diversity without importing evaluation-derived alternatives.

#### Stage 3 — Bounded tuning and sizing

- **FR-017**: Tuning MUST support separately attributable studies of learning settings and model size, with finite search spaces, trial counts, per-trial resources and total study allowance. The baseline MUST remain an explicit candidate. Data distribution and development memberships MUST be fixed within a comparison.
- **FR-018**: Learning-setting candidates MUST support learning rate/schedule, effective batch size, regularization and dropout; sizing candidates MUST include a smaller configuration and changes in width/depth. Architectural changes MUST use declared fresh initializations and compatible qualification; they MUST NOT masquerade as continued D training.
- **FR-019**: Trial objectives MUST prioritize held-out executable capability subject to completion/retention requirements. Reports MUST separate equal-token results from equal-wall-time utility and retain training, search, verification, startup and inference costs. Faster censoring MUST NOT improve the objective.
- **FR-020**: Pruning MUST use a frozen grace period, resource coordinate and decision rule. At least one baseline and one preselected non-baseline configuration MUST reach the full comparison allowance without performance-based pruning, except for technical failure, to assess early-ranking validity. Pruned, failed, infeasible and completed trials MUST remain visible.
- **FR-021**: Finalists MUST be re-evaluated at the declared longer allowance using at least two independent training seeds and a confirmation panel unused for adaptive tuning. Additional tie-break runs MUST be included in the original finite allowance or require a new study version. Adaptive development results MUST NOT be reported as unbiased confirmation.

#### Stage 4 — Progressive scale and real-sequence learning

- **FR-022**: A campaign MUST support several training tasks of up to 24 hours each, including each task's startup, online search/verification, checkpoint and finalization time. Planned pauses may exclude inactive operator waiting, but resumed execution and uncommitted crash work MUST remain charged. Preparation and qualification have separate explicit finite allowances; the campaign MUST reserve all stages and a finite aggregate budget before launch.
- **FR-023**: Token milestones and task boundaries MUST be independent. A milestone may span predeclared continuation tasks; reaching a task ceiling MUST yield a recoverable partial result. Neither restarting nor creating a sibling output directory may reset campaign tokens, search attempts, work or active elapsed time.
- **FR-024**: Recovery MUST restore complete compatible learning, allocation, generator, sampler, replay, library, random and optimizer state at declared boundaries, reconcile discarded work, and preserve the previous valid checkpoint during publication. Completed historical runs MUST NOT be reopened with silently enlarged budgets.
- **FR-025**: Progress MUST be emitted at least once per 60 seconds during active training or explicitly report unavailable/stalled progress. Each milestone MUST publish learning/transfer/retention curves, completion and cost, and separate machine-checkable evidence and next-action decisions against the frozen rules, including any pre-reserved exploratory action.
- **FR-026**: Progression MUST depend on positive held-out executable gains, acceptable retention and completion, not training loss alone. Confirmed capability adoption and its automatic scaling path MUST use confirmation across at least two training seeds. A separately predeclared, budgeted exploratory continuation MAY investigate an inconclusive result when integrity remains intact and an additional held-out/runtime signal justifies the question; it MUST NOT count as a passed learning gate. All gates MUST declare a minimum meaningful gain, uncertainty method, minimum support and retention tolerance before execution; defaults are specified below. Failed gates MUST stop their confirmed-expansion path while preserving diagnostic evidence. A frozen contingency may retain a valid baseline, defer the branch or run one bounded diagnostic episode; it cannot bypass integrity, consume unreserved resources, use assessment feedback or relabel the original result.
- **FR-027**: OEIS adaptation MUST begin with a bounded yield audit on a designated training pool, measuring independently verified discoveries, diversity, rejection reasons and cost per accepted target. The study MUST freeze minimum yield/diversity and maximum discovery cost thresholds before search; inadequate yield MUST stop expansion with an explicit deficit.
- **FR-028**: Real-sequence tasks MUST separate visible candidate inputs from additional training-only admission terms and from evaluation sequences. Training continuation checks MAY admit targets but MUST be counted as training feedback. A visible-prefix fit, imported formula or model assertion MUST NOT alone establish target validity. Candidate programs MUST remain subject to independent execution and bounded verification.
- **FR-029**: The initial OEIS loop MUST alternate bounded search, verified target admission and supervised updates mixed with synthetic retention examples. Search population/diversity, replay mixture, archive capacity, freshness and weighting MUST be frozen. Reinforcement-learning objectives are a separate intervention, not a hidden addition to this initial loop.
- **FR-030**: Self-learning MUST be compared with its starting checkpoint and a matched continued-synthetic baseline. Report learning exposure and total search-plus-training cost separately. Advancement MUST require transfer to other sequences, adequate completion and retention, not merely growth of the discovery archive.

#### Stage 5 — Reusable knowledge and honest discovery

- **FR-031**: The feature MUST support proposing, independently validating, versioning and using reusable components extracted solely from permitted training solutions. Each component MUST retain exact semantics, assumptions, dependencies, expanded representation, verification scope and origin. Invalid or cyclic/unbounded dependencies MUST be rejected under the selected language profile.
- **FR-032**: Library adoption MUST be tested against a matched no-library baseline on held-out problems, charging component discovery, validation, expanded execution and search costs. Surface compression alone MUST NOT establish improved capability or efficiency. Library updates MUST occur at declared recoverable boundaries and qualify changed execution scope.
- **FR-033**: Every discovery dossier MUST distinguish finite match, conjecture, bounded verification, proved statement and novelty review. It MUST include source/offsets, visible and tested ranges, candidate provenance, counterexample checks, assumptions, related known methods and actual verification limitations. Novelty review MAY conclude known, uncertain or potentially novel.
- **FR-034**: Algorithmic improvement claims MUST compare verified behavior and measured work over a declared range against relevant known methods. All-domain, asymptotic or theorem claims MUST require independently checked reasoning with explicit assumptions; finite tests alone cannot promote them.
- **FR-035**: The system MUST retain refutations and invalidate dependent target/component evidence transitively. Affected checkpoints MUST be identified and quarantined for the relevant claim; recovery MUST use clean lineage or explicitly report contamination rather than delete adverse evidence.

#### External data, operation and evidence

- **FR-036**: External-source admission MUST freeze source revision, availability/usage conditions, exact values, indices, object relationships, task transformation and verification method. A FindStat/Sage-derived combinatorial pilot MUST be supported; LMFDB/Cremona and KnotInfo/Dartmouth MUST receive documented suitability decisions before any optional ingestion. Bulk acquisition of all sources is not required.
- **FR-037**: Non-integral, approximate, finite-support or object-valued data MUST enter only an explicitly compatible task profile. Polynomial zero padding and arbitrary ordering MUST NOT manufacture apparent extrapolation successes. Source expansion MUST demonstrate added structural coverage or matched-budget transfer benefit before larger acquisition/training is adopted.
- **FR-038**: All production stages MUST operate through the existing portable workstation workflow with bounded isolated roles, no new host services or driver changes, and actual accelerator execution for selected device tests. Qualification MUST match source, language, model family, numerical precision and runtime scope; compatible receipts MAY be reused without repeating unrelated gates.
- **FR-039**: The operator MUST be able to prepare/freeze, inspect, execute, pause, resume and review the whole staged campaign with documented invocations. A single invocation MAY advance through already frozen eligible stages. It MUST stop at failed/inconclusive gates without unplanned research expansion or repeated requests for already granted scope.
- **FR-040**: Each campaign MUST collect one bounded diagnostic package on success, partial completion or failure containing manifests, receipts, actual outcomes, first producer error/traceback, checkpoints' identities, accounting and available timing/queue/memory/device observations. Unavailable metrics and omitted files MUST be explicit. Independent review MUST be able to reconcile reported aggregates and gate decisions from included evidence; required evidence omissions prevent verified status.
- **FR-041**: Repository publication MUST contain concise findings, reproducible scope and limitations. Raw datasets, complete experimental outputs and model weights remain workstation artifacts. Historical plans, scores, INCOMPLETE results and qualification scope MUST retain their original meaning.

### Research progression and scope

Software acceptance and scientific advancement are separate. Negative or inconclusive research is a valid completed experiment, but does not satisfy a positive learning gate or authorize the confirmed-scaling path. A separate prospectively frozen contingency may authorize bounded investigation or continuation of a valid baseline, with the failed gate preserved. All five stages require implemented and tested capability even if live research stops early.

| Stage | Required research decision | Evidence permitting dependent work |
| --- | --- | --- |
| 1. Foundation | Are execution, evaluation and language coverage adequate for the selected tasks? | Current qualification; replayable benchmark and coverage report; sufficient fresh support; no unclassified acceptance failures |
| 2. Diversity | Does refreshed structural support improve transfer at useful cost? | Matched static/refreshed comparison and retention/completion measurements; select the baseline if refresh does not help |
| 3. Tuning | Which settings and size justify longer compute? | Bounded study, completed controls, independent confirmation and repeat-seed evidence; retain baseline or stop when inconclusive |
| 4. Scaling | Do further tokens and then verified real-sequence learning improve other-task performance? | Positive milestone decisions; for OEIS, adequate discovery yield and improvement against continued synthetic training |
| 5. Abstraction | Do reusable methods improve held-out capability or cost, and what discovery claims are supported? | Matched component ablation, independent checks and scoped discovery dossiers; novelty and proofs receive separate decisions |

Learning gates use all-task family-macro verified yield as the default primary metric, with unsolved/censored tasks retained in the denominator and missing outcomes reported separately. The proposed default minimum meaningful gain is **2 percentage points**, at least **95% measurement completion** in each compared primary stratum, and no more than **2 percentage points loss** on the declared familiar-task retention aggregate. Confirmation requires the same improvement direction in at least two training seeds and a positive lower bound under a predeclared paired, family-aware 95% uncertainty procedure; task resampling alone MUST NOT be claimed to estimate training-seed variability. These are prospective feature-013 defaults, not changes to feature-012 thresholds. Planning MUST define minimum panel support and distinguish descriptive from inferential uncertainty; infeasible support yields INCOMPLETE. Any alternative thresholds must be frozen with rationale before outcomes, never selected to rescue a failed run. These numerical research targets are provisional before freeze and binding for the selected claim afterwards; exact correctness, isolation and accounting remain unconditional. Study sizes, token milestones and curriculum fractions are tunable design choices, not quotas to consume. The plan MUST classify targets and define bounded responses to misses, including an explicitly separate efficiency/noninferiority claim when predeclared. Neither useful efficiency nor a justified exploratory action may be presented as confirmed accuracy improvement.

The initial synthetic-only studies may use OEIS development scores for declared model selection, but never OEIS values as target construction or allocation feedback. The later OEIS-derived track is explicitly in scope under the user's request and must be identified as a separate experimental track under constitution V. This specification does not amend the constitution or assert that its runtime adoption/qualification is already complete.

Out of scope: unrestricted autonomous research budgets; automatic host reconfiguration; imported OEIS/LODA solutions or pretrained external model teachers; an unbounded hyperparameter sweep; all-database bulk ingestion; numerical approximation presented as exact integer reasoning; and automatic claims of mathematical novelty. Those are not prerequisites for the five-stage pathway.

### Key Entities *(include if feature involves data)*

- **Capability contract**: Task strata, grouping/exposure history, baselines, numerical and language profile, visible/scoring ranges, budgets and uncertainty/advancement rules.
- **Language coverage record**: Representative task, independent witness, expressibility/budget classification, unsupported operations and qualification scope.
- **Training stream commitment**: Generator/source identities, ordered draws, acceptance/rejection evidence, structural diversity, splits and spent-work continuation state.
- **Tuning study and trial**: Finite search space, effective settings, seed, resource coordinates, pruning history, outcomes and independently confirmed selection.
- **Campaign and training task**: Ordered stages, aggregate budgets, per-task ceilings, checkpoint lineage, milestone decisions and remaining allowances.
- **Verified target archive**: Candidate programs, training membership, exact acceptance evidence, diversity/weighting, search costs and refutation dependencies.
- **Reusable component library**: Versioned validated components, dependency graph, expanded semantics/cost, origin and compatible model/runtime scope.
- **Discovery dossier**: Candidate method, comparisons, tested domain, counterexamples, claim category, proof evidence and novelty-review disposition.
- **External-source admission**: Immutable source, usage terms, task transformation, exact values/indices, related-object grouping and suitability result.
- **Evidence package**: Bounded independently reconcilable receipts, measurements, decisions, failures and explicit omissions for one campaign.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One readiness report covers all four transfer categories and distinguishes learning, language limits and operational failure; every scored task has an outcome and declared denominator, and all acceptance counterexamples receive the expected classification (FR-005–011).
- **SC-002**: Static and refreshed pilots reproduce their declared inputs and consumed exposure after interruption and report structural diversity separately from parameter variants. The researcher can determine the matched-token and measured-cost effects without undocumented differences (FR-012–016).
- **SC-003**: One bounded tuning study accounts for 100% of launched trials, includes the required unpruned controls and confirms finalists across at least two seeds on an unused confirmation panel. Replaying the recorded rules reproduces its selection or inconclusive result (FR-017–021).
- **SC-004**: A production acceptance sequence demonstrates at least two linked training tasks with interruption and recovery; no task exceeds its declared ceiling of at most 24 hours, no resume resets spent budget, and all checkpoint/cleanup reserves are reconciled. Short rehearsals establish mechanics; only actual sustained runs establish long-duration performance (FR-022–025).
- **SC-005**: Every attempted scale increase produces a reproducible advance, stop or inconclusive decision. Positive advancement meets the frozen learning, completion, retention and uncertainty thresholds; improving training loss alone never passes this gate. An exploratory action must reproduce its separate frozen trigger, information goal and remaining resource reservation while preserving the original evidence verdict (FR-025–026).
- **SC-006**: A bounded OEIS pilot reports accepted distinct targets, structural coverage and cost, rejects known prefix-only overfits, and either completes a verified search/update/evaluation cycle or publishes the precise yield deficit. Every accepted target belongs to the permitted training pool (FR-027–030).
- **SC-007**: At least one reusable-component acceptance case validates extraction, independent semantics, recovery and matched evaluation, including a negative/refuted case. Research reports separately determine whether components improve unseen-task capability or measured efficiency (FR-031–032, FR-035).
- **SC-008**: Every discovery report includes its claim category and evidence limits; unsupported proof/novelty labels are rejected, and a deliberate refutation identifies all dependent targets, components and checkpoint lineages (FR-033–035).
- **SC-009**: A bounded combinatorial-source pilot and suitability reports for the other proposed source groups are delivered. All admitted tasks preserve source meaning, exact values, index conventions and cross-source exclusions; unsuitable samples receive explicit rejection reasons (FR-036–037).
- **SC-010**: Actual accelerator qualification is available for every research execution scope, progress gaps exceeding 60 seconds are identified, and success/partial/failure paths each produce a bounded package with the original failure details and independently reproducible aggregate/gate calculations (FR-038–040).
- **SC-011**: A final feature report records implementation acceptance and actual research disposition for all five stages. Unexecuted or blocked stages remain explicit; completed software acceptance never implies established learning, novelty or proof (FR-001–004, FR-041).

## Assumptions

- The target remains the user's Ryzen AI Max+ 395 workstation with 128 GB shared RAM, existing supported accelerator runtime and the repository's portable container workflow. Inference memory measurements do not establish training-memory capacity; each model size needs measured feasibility.
- PR #63 is merged at `9ea2d12dab91ed8b975dabb1dad11c28af33da1d`. Existing qualification and retained artifacts may support compatible scopes; changes require scoped requalification. The existing drain schedule is sufficient to start; repairing overlap is optional throughput work, not a blocker for all research.
- The user authorized specification of the full pathway and multiple training tasks of up to 24 hours. This turn creates requirements, not runtime implementation or a launch of those tasks. Future execution may proceed through a concretely frozen authorized campaign without asking again at every internal step.
- Initial planning defaults are 16 short tuning trials, extension of at most three finalists, and at least two confirmation seeds. Exact trial resources and a finite total campaign allowance must be frozen after feasibility measurement. These are proposed defaults, not measured optimums or a promise that every experiment uses its full allowance.
- Proposed cumulative scaling milestones are 25M, 100M and 400M charged training tokens. Architecture/data changes may justify different prospectively frozen milestones. A milestone may require multiple predeclared tasks; this is not permission to run any individual task longer than 24 hours or to conceal earlier initialization/training cost.
- Model sizing and learning-setting searches are separate from changes to training distributions, language primitives and search policy. A new model size starts afresh; continuation of retained D requires an explicitly compatible child experiment and provenance for all prior exposure.
- FindStat/Sage-derived combinatorial tasks are the first external-source pilot. Other databases are staged by suitability; no claimed quantity of usable sequences or access method is assumed before admission.
- The default discovery input remains exact sequence terms. Object-conditioned mathematical tasks require an explicitly declared input/profile extension and separate comparisons. Imported formulas, names or metadata must not silently solve a terms-only task.
- Exact finite continuation is the initial capability endpoint. New mathematical approaches are a research goal, not a guaranteed delivery outcome. A known method rediscovered independently remains valuable learning evidence but is not a novelty claim.
- Implementation planning must cover the complete five-stage dependency graph and acceptance tasks, including negative paths, rather than defer stages 4–5 indefinitely behind a pilot-only implementation. Detailed algorithms, library choices and source layouts belong to that next phase; Optuna is a candidate implementation for the required bounded tuning capability.

### Prior evidence and research basis

- [Spec 012 original research plan](../012-parameterized-adaptive-curriculum/validation/research-plan.md): historical scope and budget choices; not authority to relaunch completed experiments.
- [Fresh structural findings](../012-parameterized-adaptive-curriculum/validation/fresh-002-review.md): D's concentrated advantage, template complementarity and preserved INCOMPLETE result.
- [Qualified reliability and scheduling review](../012-parameterized-adaptive-curriculum/validation/reliability-results.md): current probe gate, actual device evidence and overlap failure.
- [Historical long-run analysis](../../docs/pilot-analysis-20261028.md): memorization and limitations of the previous distribution; not proof of a model capacity ceiling.
- [Deep symbolic regression for recurrence prediction](https://proceedings.mlr.press/v162/d-ascoli22a.html): synthetic training and OEIS transfer precedent.
- [Learning Program Synthesis for Integer Sequences from Scratch](https://arxiv.org/abs/2202.11908): learned search and sequence self-learning precedent.
- [DreamCoder](https://arxiv.org/abs/2006.08381): reusable abstractions and search/learning precedent.
- [FunSearch](https://www.nature.com/articles/s41586-023-06924-6): independently evaluated program search as a discovery route; different pretrained-model assumptions.
- [FindStat](https://www.findstat.org/), [Sage interface](https://doc.sagemath.org/html/en/reference/databases/sage/databases/findstat.html), [Cremona data](https://johncremona.github.io/ecdata/), [LMFDB](https://www.lmfdb.org/), [KnotInfo downloads](https://knotinfo.org/homelinks/database_download.php), [Dartmouth knot data](https://knots.dartmouth.edu/): source suitability requires task-level admission, not merely availability of files.
