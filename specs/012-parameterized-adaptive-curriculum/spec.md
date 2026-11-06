# Feature Specification: Parameterized Targets and Adaptive Curriculum

**Feature Branch**: `012-parameterized-adaptive-curriculum`

**Created**: 2026-11-04

**Status**: A–D implemented and all mandatory Phase E software/evidence tasks verified, including the supplemental actual HIP crash/debit proof. Optional T073 research analysis remains deferred; see [closeout](validation/closeout.md).

**Input**: Build on merged 011 / PR #62 to compare parameterized program targets with literal targets, and adaptive family allocation plus spaced review with a fixed family mixture. Complete structural-only evaluation prerequisites, concentrate essential complex work in dedicated phases, minimize operator HIP invocations and prevent recurrence of 011 reliability failures.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish a feasible structural comparison (Priority: P1)

As a researcher, I want a fresh, frozen structural evaluation and trustworthy controls before spending hours training or inferring on an unusable cohort.

**Why this priority**: Scientific comparisons require valid denominators and lineage.

**Independent Test**: Prepare a small structural cohort, reject an independently constructed exclusion/quota failure, and evaluate actual retained checkpoints without a learner.

**Acceptance Scenarios**:

1. **Given** the accepted 011 structural allocation, **when** preparation succeeds, **then** exactly 128 tasks from 27 families are frozen: 68 certified, 26 ambiguous and 34 unresolved; parameter evaluation is explicitly deferred.
2. **Given** a quota shortfall, missing original input or native/reference disagreement, **when** admission runs, **then** no research learner/generator starts and exported evidence identifies affected families, original failure and restart boundary.
3. **Given** an old literal checkpoint, **when** reuse is considered, **then** initialization, ordered data, model/optimizer, exposure and execution compatibility are established or the checkpoint is labeled an unmatched reference.

### User Story 2 - Learn parameterized programs (Priority: P1)

As a researcher, I want to remove unnecessary numeric-literal prediction while preserving program structure, control flow and exact grounding.

**Why this priority**: Parameterized targets are the selected next learning direction, but their benefit must be measured separately from allocation changes.

**Independent Test**: Build paired views of concrete programs, independently verify grounding, perform small matched learner updates and evaluate with identical fitting availability.

**Acceptance Scenarios**:

1. **Given** supported numeric parameters, **when** targets are prepared, **then** grounding reproduces the original concrete program and both views retain identical underlying examples/splits.
2. **Given** control-sensitive or unsupported parameters, **when** views are prepared, **then** the example retains a counted literal fallback rather than disappearing or becoming an unsafe freely fitted target.
3. **Given** shorter parameter targets, **when** results are compared, **then** tokens, examples, updates and elapsed cost are reported separately; token matching is not called equal-data or equal-time matching.

### User Story 3 - Focus practice and schedule retention (Priority: P1)

As a researcher, I want practice to follow useful learning progress and measured forgetting across training families, while maintaining exploration and revisiting acquired skills.

**Why this priority**: 011 compared repeated versus fresh instances under a fixed mixture; it did not test adaptive allocation.

**Independent Test**: Supply controlled observations for improving, stable, deteriorating, noisy and unmeasured families, then verify bounded priorities, review opportunities and deterministic decisions.

**Acceptance Scenarios**:

1. **Given** a learnable improving family and a persistently high-error family, **when** allocation updates, **then** high error alone cannot monopolize training and every family retains exploration.
2. **Given** previously learned examples, **when** independent reviews succeed repeatedly, **then** each reviewed example advances through bounded review intervals on its own target-recall evidence; failure returns it to frequent review, and scheduled practice performs an actual charged learner update. A separate family probe cannot promote that example.
3. **Given** unavailable or timed-out probes, **when** feedback is processed, **then** no success, failure, mastery or forgetting is invented.
4. **Given** different family lengths/costs, **when** exposure is audited, **then** probabilities, selection reasons, examples, tokens and elapsed cost explain allocation.

### User Story 4 - Recover without changing the experiment (Priority: P1)

As an operator, I want recoverable multi-hour work, useful progress and shareable evidence without manual diagnostic loops or repeating unrelated successful stages.

**Why this priority**: 011 failures repeatedly involved budgets, provenance, checkpoint overhead and missing evidence rather than learning itself.

**Independent Test**: Interrupt preparation, batch reservation, feedback, review, checkpoint publication and export; compare resumed fixtures with uninterrupted execution and verify all terminal archives.

**Acceptance Scenarios**:

1. **Given** a complete compatible checkpoint, **when** adaptive training resumes, **then** model, scheduler, review, pending work and charged budgets restore together without duplicate feedback or refunded crash work.
2. **Given** completed compatible preparation/evaluation, **when** a dependent stage retries, **then** original evidence is validated and reused; incompatibility names the smallest safe rerun.
3. **Given** a primary failure followed by export failure, **when** finalization runs, **then** both errors survive in bounded independently verifiable evidence or an explicit last-resort diagnostic location.

### User Story 5 - Attribute gains and diagnose limitations (Priority: P2)

As a researcher, I want structural outcomes, retention curves and before/after program examples that distinguish representation gains, allocation gains and diagnostic overhead.

**Why this priority**: Equal final scores can hide different learning speeds, proposals and fitting behavior.

**Independent Test**: Reconcile fixtures with equal totals but different family errors, retention and costs, including missing measurements.

**Acceptance Scenarios**:

1. **Given** literal-fixed A, parameterized-fixed B and parameterized bandit-plus-review C, **when** comparisons are reported, **then** B–A addresses representation and C–B addresses the combined package. Only a separately frozen optional bandit-only D supports C–D attribution to replacing ordinary bandit practice with scheduled review; no review-only benefit is claimed without it.
2. **Given** positive, negative or inconclusive single-seed findings, **when** reporting finishes, **then** no automatic winner, extra seeds or assessment-based policy change occurs.
3. **Given** available passing/failing attempts, **when** cases are selected for review, **then** original/parameterized/fitted sources and reasons are retained, with unavailable categories explicitly stated.

### Edge Cases

- Cold-start/equal weights; selection-time versus later probabilities; incorrect first sharing update; reward sign loss; misattributed cross-family trends; noisy or stale feedback; persistent errors without progress; cheap families dominating; review demand exceeding its reserved capacity.
- Initial acquisition mistaken for retained recall; incidental practice omitted from review clocks; a grade survives a failed optimizer update; simultaneous due anchors exceed service capacity.
- Memorized examples remain easy while unseen instances from the same training family deteriorate; unavailable diagnostic pools; ambiguous continuation or solver timeout mistaken for incompetence.
- Repeated parameters, protected guards, nonlinear targets, parameter-count limits, unsafe grounding and semantically equivalent programs with different target text.
- Historical initialization/order evidence missing; source/image/backend mismatch; weights-only data presented as resumable state.
- Reference feasibility passes but original-data exclusions remove a whole family; parameter exclusions remain relevant despite deferred parameter evaluation.
- Slow startup/checkpoint/export, insufficient space, worker death, unacknowledged cleanup, double-applied feedback, stale cached artifacts or partial publication.
- Missing nested results, a verifier's COMPLETE status omitted by a generic projector, unsafe evidence paths, or a second error masking the first.

## Requirements *(mandatory)*

### Functional Requirements

#### Controlled comparisons and inherited evaluation

- **FR-001**: Register a distinct experiment policy for fresh parameterized fixed/adaptive comparisons, preserving existing 011/P3 recipe semantics and historical records.
- **FR-002**: Freeze arms, changing factors, initialization, data/support, representation, scheduler, diagnostics, checkpoints, budgets, metrics and stopping rules before outcome-dependent decisions. B/C MUST differ only in declared allocation/review policy; matched A/B only in target representation and its disclosed consequences.
- **FR-003**: A MUST be a verified compatible retained control or a separately frozen fresh matched control. Otherwise label it unmatched and withhold causal B–A claims; B–C validity remains separately assessable. Compatibility includes ordered concrete inputs, initial weights, model/optimizer, exposure, diagnostic intervention and execution policy.
- **FR-004**: B/C MUST use the same finite 32-family training support and underlying concrete pool. No online generation, support expansion, different numeric law, reinforcement learning, search-discovered demonstrations or hidden refresher corpus belongs to this first intervention.
- **FR-005**: Complete 011 T177–T180's structural-only prerequisites and T181's control audit without relabeling T174 complete. Historical v1 selection contains 128 tasks/27 families (68 certified/18 families). The operator-authorized v2 amendment contains 128 tasks/23 families, with 68 certified tasks/15 families as primary, 26 ambiguous/3 families and 34 unresolved/5 families separately reported. The populations MUST remain separately identified.
- **FR-006**: Preserve reviewed structural draw identities, explicitly versioned frozen quotas, original exclusions, 20 visible/100 scoring terms, eight attempts, 60-second matched portfolios, seven controls and bank seeds. All 16 original exclusion inputs remain required; parameter evaluation reads DEFERRED.
- **FR-007**: Every frozen quota and native agreement check MUST pass before research learning/inference. Export per-family availability before/after exclusion, rejection counts, deficits, timings and input identities without protected terms. Never silently reseed, add draws, relax quotas or substitute families.

#### Target and feedback isolation

- **FR-008**: Paired learning views MUST retain concrete identities, exact independent grounding, declared sharing and supported syntax. Protect control-dependent literals and count safe literal fallback; never silently filter unsupported rows.
- **FR-009**: Compared representations MUST share fitter, final-source validation and inference budgets/parameter limits. Placeholders MUST NOT be described as repairing wrong structure or revealing unobserved independent coefficients.
- **FR-010**: Allocation feedback, review and diagnostic panels MUST derive only from mechanically generated training support. Structural selection/assessment targets, scores or family-specific failures MUST NOT enter automatic allocation, review, target construction or support expansion.
- **FR-011**: Competence probes MUST be distinguished from gradient examples and final evaluation, include independent within-training-family instances, and disclose probe reuse/adaptation. Loss on the just-trained example alone is insufficient evidence of family progress.

#### Adaptive practice and review

- **FR-012**: Use a named, faithful published EXP3.S variant for family allocation, with selection-time importance weighting, explicit exploration, fixed-share updates and numerical conformance fixtures. Freeze reward definition, scaling, cost proxy and every departure in the surrounding training/review policy; a custom priority score or legacy class name MUST NOT be presented as that algorithm.
- **FR-013**: Every family MUST retain a declared exploration probability floor. State the resulting concentration bound and actual exposure separately; do not silently clip probabilities into a different algorithm. Review anchors MUST have a finite service-opportunity bound, including synchronized deadlines, initial acquisition and terminal-budget truncation; reject impossible schedules before training.
- **FR-014**: Maintain bounded per-example review state for representative existing training examples, using a declared Leitner-style adaptation. Each acquired example's own pre-update recall observation may promote its queue; a completed failure returns it to the shortest interval. Missing/censored grades cannot promote mastery. Record incidental committed exposure and acquisition separately; retain independent family probes to measure transfer.
- **FR-015**: Review choices MUST perform charged updates within the same learner allowance, not only create events or add unreported tokens. Repeated-example review and other practice MUST be separately counted.
- **FR-016**: Measure a chosen family's learning-progress reward on the same separate training probe before and after an identified bounded contiguous training block, with no intervening other-family or review updates. Preserve signed progress and observed counts. All-family trends are diagnostics, not retrospective credit to the most recently selected arm. Do not run full generation/fitting per batch; target recall, loss, semantic success and execution completion remain distinct.
- **FR-017**: Diagnostics and reward observation MUST have explicit work/time ceilings and missing-feedback rules. A censored reward is not an observed zero: preserve completed training and debit, suppress its reward update, and report missing coverage. Persistent failure pauses or marks the adaptive result incomplete under a frozen rule; no silent successful adaptation or automatic feedback-policy switch.
- **FR-018**: Compared new arms MUST share common monitoring definitions, panels and cadence for comparable retention. Policy-required reward/grade observation counts may differ, especially in optional D, and MUST be separately disclosed and charged. Selected-family reward observations remain policy-conditioned and must not be called matched evaluation. Charge all controller/diagnostic costs, expose distribution reweighting and preserve fixed evaluation denominators.

#### Recovery, operations and evidence

- **FR-019**: Resume MUST bind model/optimizer, random state, data order, bandit logweights and valid-update/action counters, reward-scaling history, selection-time probabilities, pre/post probe and pending block state, per-anchor acquisition/queue/due/exposure state, feedback IDs, pending batch and token/time ledgers. Reject incompatible policies, versions, inventories and execution environments.
- **FR-020**: Reservation, optimizer update, reward processing and review advancement MUST have explicit commit ordering. A partial block cannot receive a fabricated complete reward, and an unsuccessful optimizer update cannot promote review state. Restore the last complete checkpoint without duplicated actions/feedback; spent-but-lost work remains charged. Never restore a newer policy journal onto older model weights.
- **FR-021**: Normally persist full recovery snapshots about hourly, plus initial/final and declared milestone pins, at update boundaries. Keep frequent lightweight progress. Slow atomic publication has its own bounded allowance; previous complete state remains usable on failure.
- **FR-022**: Preflight MUST verify exact input locations/inventories, control compatibility, finite data/probe capacity, target/model limits, source/images/runtime/resources, peak simultaneous disk use and nested deadlines before costly stages. Bounded discovery failure names an explicit root/inventory requirement.
- **FR-023**: Reuse original compatible producer evidence and immutable completed stages only after dependency validation. Production changes invalidate affected qualification; temporary-helper changes bind new execution plans separately. Old failed evidence stays failed.
- **FR-024**: Record each stage's command/configuration identities, start/progress/terminal status, elapsed/remaining time, failure and cleanup acknowledgement. A heartbeat alone MUST NOT be reported as measured progress.
- **FR-025**: Terminal handling MUST preserve the first error plus later cleanup/export errors, bounded role logs/ledgers, quota/progress data, original producer references and omissions. Sharing excludes private weights/state/truth; provide a verified minimal fallback and read-only evidence-recovery path.
- **FR-026**: Recovery MUST identify the smallest unfinished/incompatible dependency. A restarted wrapper must not automatically repeat validated completed stages; unverified or failed outputs never qualify for successful reuse.

#### Decisions and delivery

- **FR-027**: Report structural family-macro success, task counts, paired discordances, censoring, raw/fitted behavior, family retention and learning curves against tokens, examples, updates and elapsed cost. Distinguish training-probe curves from frozen final structural scores; diagnostic opportunity and any unavailable historical curves remain explicit. Do not treat tasks as independent training seeds.
- **FR-028**: Primary structural interpretation requires at least 65/68 certified tasks completed in each compared arm. Distinguish classification ambiguity from unresolved execution. Single-seed findings remain exploratory; adaptive exposure does not alter evaluation denominators.
- **FR-029**: Deterministically review at least 12 available case/arm records spanning shared/arm-exclusive successes, continuation errors, no-fit, timeouts/decode failures and simple-control wins; retain before/after sources/assignments where available and state missing categories/data.
- **FR-030**: Small software acceptance MUST exercise actual finite accelerator updates, actual checkpoint evaluation, paired targets, adaptive feedback, real review updates, compatible exact pause/resume and failure export. Neither full research training nor a positive learning result is required for acceptance.
- **FR-031**: Research execution requires a separately frozen operator choice. The default screen remains B/C plus a qualified or labeled A reference. An optional bandit-only D requires explicit arm selection, matching initialization/data, added complete resource reservations and distinct interpretation. One invocation may qualify then run the already authorized scope; no automatic ablation, extra seeds, winner adoption or assessment exposure.

### Non-Functional Requirements

- **NFR-001 — Minimal critical phases**: Isolate highly complex work in at most two dedicated implementation phases: algorithm/state invariants and essential correctness-critical execution/provenance/resume integration. Only invariant-bearing work belongs there. Keep scaffolding, routine wiring, docs and broad test execution separate; independent counterexamples belong with their critical work. One combined critical phase is acceptable if reviewable. Additional critical phases require explicit justification and a revised plan rather than scattered algorithm edits.
- **NFR-002 — Fewest operator runs**: Normal unchanged-compatible acceptance MUST require one operator invocation, serially composing canonical project operations and automatically packaging evidence; a temporary script is permitted. Include CPU/native feasibility, representative startup/work/checkpoint calibration and small HIP proofs before expensive research. Qualify once per required compatible scope; any repeat records its cause and smallest safe rerun.
- **NFR-003 — Reliability before hardware**: Every reviewed 011 failure class MUST map to a prevention gate, regression/injected failure, evidence field and recovery behavior before hardware handoff. Hardware must not be the first test of closed contracts, quotas, packaging paths, deadline graphs or checkpoint transactions.
- **NFR-004 — Generous finite limits**: Enforce nested wall/work/memory/disk bounds including startup, synchronization, checkpoint I/O, cleanup and reporting. Forecast full scale from representative cold/warm/worst-observed cases with explicit margin. Multi-hour training is acceptable; no inherited five/six-hour campaign limit may silently truncate justified longer work. Operational margins do not enlarge scientific per-task budgets.
- **NFR-005 — Isolation and portability**: Preserve the single-learner container workflow, exact integers, strict mechanical provenance and role isolation. No host driver/service changes, external teachers or silent CPU fallback.
- **NFR-006 — Compatibility**: Existing 011 endpoints/studies/qualification retain their meaning or receive an explicit new version. Freeze identities for every scientific-policy and reuse decision; missing measurements remain unavailable.
- **NFR-007 — Efficient observability**: Avoid full snapshots every update. Bound logs, queues, caches and progress records while retaining whole-run throughput/ETA/retention evidence. Only deterministic bounded CPU preparation may run in parallel; accelerator roles remain serial.
- **NFR-008 — Honest efficiency**: Charge and separately report preparation, diagnostics, review, checkpoint and total costs. Accuracy gains or speedups are hypotheses, not acceptance requirements.

### Key Entities *(include if feature involves data)*

- **Experiment contract**: Frozen arms, policy versions, source/environment, control/data lineage, budgets and decision rules.
- **Target pair**: Concrete program, paired views, grounding proof, parameter sharing and fallback disposition.
- **Family inventory**: Allowed training families, examples, independent probes and exclusion commitments.
- **Family observation**: Selected block/probability, matched pre/post probe, signed reward and scaling provenance, diagnostic counts, completion, cost and immutable event identity.
- **Review state**: Per-anchor acquisition, queue level, due/last-exposure update, grade identity and practice/review history; target recall remains distinct from semantic competence.
- **Adaptive state**: Published-kernel weights/counters, reward-scaling state, review queue, random/cursor state and complete pending block/observation state.
- **Control receipt**: Evidence establishing or rejecting a matched historical comparison, distinct from weights-only reuse.
- **Run evidence**: Original stage references, results or incomplete reasons, costs, logs and recovery boundaries.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 128 structural tasks and category/family totals reconcile, or research never starts and every shortfall is reported.
- **SC-002**: Every target pair passes independent grounding; every unsupported row retains a counted safe disposition without denominator loss.
- **SC-003**: Independent numerical fixtures reproduce the selected EXP3.S variant, and controlled scenarios demonstrate exploration, bounded per-example review and actual charged retention updates; noisy/unavailable feedback cannot fabricate competence.
- **SC-004**: Qualified pause/resume fixtures match uninterrupted model/optimizer, next examples, adaptive/review state and charged exposure with zero duplicate feedback IDs for clean boundary pauses; crash recovery additionally accounts for discarded work.
- **SC-005**: One normal operator invocation completes small acceptance and verifiable evidence without intermediate manual commands; every injected failure has a truthful bounded recovery record.
- **SC-006**: 100% of reported arm/family aggregates and exposure/cost totals are reconstructible from exported evidence, with explicit missing measurements.
- **SC-007**: Reports separate B–A representation and C–B allocation/review effects, matched/unmatched controls and positive/negative/inconclusive findings without requiring improvement.
- **SC-008**: Active supervised stages emit progress or explicitly unavailable/stalled progress at most 60 seconds apart; forecasts include measured throughput and checkpoint/export reserves.
- **SC-009**: Each 011 reliability class has an owner, prevention gate, failure test and evidence field before hardware delivery.
- **SC-010**: The plan maps all invariant-bearing work to no more than two bounded critical phases and uses one consolidated operator-validation phase, with routine work separate.

## Assumptions

- Merged 011 at `efce05d64294cb013d7075802e44560ce1d45b05` supplies the foundation. Six unchecked follow-ups are inherited, not retroactively completed.
- The initial screen uses one paired seed and 25M target tokens per new arm. Several hours per learner are acceptable; bandit parameters, block size, review cadence and operational limits are explicit planning defaults to freeze, not measured optima.
- B/C start from identical fresh random initialization. A reuse requires complete matching evidence; otherwise a fresh A or reference-only path is chosen before research, not improvised after outcomes.
- The first policy combines the Graves et al. EXP3.S fixed-share kernel with a bounded per-example Leitner adaptation on a fixed finite pool. The optional bandit-only ablation is supported but off by default. RbF/MIR, review-only ablations, support/refresh changes and multi-seed confirmation are deferred; no theorem establishes optimal final accuracy for this combined policy.
- Original private inputs/checkpoints and compatible images remain on the workstation. Missing originals require a specific preflight action or separately frozen replacement, not invented provenance.
- Existing repository evidence and prior public archives suffice for design; no chat/file resend is needed now. Actual control compatibility and performance calibration require operator-side originals later.
- This specification/design does not run training/HIP, establish efficiency gains or automatically authorize a research campaign.


## Operator-authorized feasibility amendment (2026-11-05)

After the actual original-input admission failed, the operator requested a fix.
The frozen [structural amendment](validation/structural-amendment.md) supersedes
only the structural population/quota parts of FR-005/FR-006 and their 27/18-family
success-criterion references for new v2 runs. The current population is 128 tasks,
23 families: 68 certified/15 families, 26 ambiguous/3 and 34 unresolved/5. All 2,288
original draw identities, numeric law and all sixteen original exclusions remain
unchanged. Quotas are frozen from outcome-blind conservative distinct-source
capacity; no runtime reseed, reallocation, biased relabeling of old results or
exclusion relaxation is permitted. Training and 65/68 completion requirements do
not change. Claims are conditional on this disclosed feasible population; historical
v1 comparisons retain their original denominators and failures.
