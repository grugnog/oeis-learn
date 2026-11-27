# Feature Specification: Round028 — Learning to Select and Generate Sequence Programs

**Feature Branch** `015-learned-program-search`

**Created** 2026-11-25

**Status** Draft for research planning, stacked on feature 013 (PR #72).

**Input** Use scarce GPU time to find out whether a model can learn to choose or propose better executable sequence programs. Reuse the existing campaign and trial infrastructure. Add a hard limit or timeout only where critical.

**Research decision.** Given the first 20 terms and their original indices, can a learned method choose one executable program that predicts the next 80 terms exactly more often than the registered incumbent, under the same search budget? A useful round can also establish that the next investment belongs in fitting constants, generating candidates, or closing a gap in the evidence.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Find where round027 lost answers (Priority: P1)

As a researcher, I want to know whether the right program was proposed, fitted and selected before I spend GPU time training another model.

**Why this priority** Trial 05 solved 18 of 128 development problems at 1M tokens. Retained model D solved 23 on the same cases. They used the same verifier for successes and different rules for unresolved attempts. The overlap is still unknown. All eight unresolved trial 05 cases had unfinished fitting, including six composition cases. Trial 00 solved 11 at both 1M and 2M, gaining four cases and losing four.

**Independent Test** A reviewer can trace an individual case from the original prompt and attempted programs to the published verdict, including the first step where the evidence runs out.

**Acceptance Scenarios**

1. **Given** the 128 historical cases, **when** the operator requests the audit, **then** it shows paired gains, losses and overlaps, starting with trial 05 versus D, and identifies all six unfinished composition cases. It accounts for the six 1M trials, trial 00 at 2M, the controls and each comparable saved 25M result.
2. **Given** a sealed set of candidates containing an independently verified solution, **when** the original selector chose a different program that fails, **then** the report identifies a selection opportunity. A truncated attempt record yields an unknown candidate set.
3. **Given** an unfinished fit, **when** the original visible inputs, program and resource profile are available, **then** the operator can replay that fit on CPU and see both outcomes and costs. A changed fitter produces separately identified evidence. Round027 keeps its original decisions.

### User Story 2 - Test whether a model can choose better (Priority: P1)

As a researcher, I want to compare the learned and existing selectors on the very same candidate programs. If both selectors get a better candidate list, I still need to know which part of the improvement came from learning.

**Why this priority** The earlier structural comparison found six completed cases solved by D alone against templates and five solved by templates alone against D. Candidate selection is a plausible, inexpensive first experiment. [Earlier paired results](../012-parameterized-adaptive-curriculum/validation/fresh-002-review.md)

**Independent Test** Give both selectors the same saved candidate set for each fresh development case. An independent scorer can reproduce their exact continuations from the programs they committed to.

**Acceptance Scenarios**

1. **Given** mechanically generated training families, **when** selectors are trained, **then** every positive and negative label is traceable to a program and an exact continuation on a training case. The learner can see examples of plausible programs that match the visible terms and fail later.
2. **Given** a frozen candidate list and a fixed model, **when** the evaluation compares the existing selector, the learned selector and the learned selector supplied a different case's visible terms, **then** all three receive identical eligible candidates. Only the learned conditioning input changes in the shuffled comparison. Each selection is fixed before the scorer receives the hidden terms.
3. **Given** an apparent same-list advantage, **when** the complete search is evaluated, **then** the candidate and incumbent each get the same total allowance, including generation, fitting and selection. The report shows model-only, conventional-only and combined results where applicable. It identifies additional solves attributable to the learned choice.

### User Story 3 - Train a generator when proposals are the bottleneck (Priority: P2)

As a researcher, I want a matched training comparison when the existing system routinely fails to propose a solution to a problem it can represent and execute.

**Why this priority** Training longer helps only if it improves the accompanying evidence. The saved round027 diversity runs already contain two fixed-collection and two refreshed-collection checkpoints of approximately 25M tokens each.

**Independent Test** Starting from the same eligible initialization within each seed pair, compare ordinary training with one intervention that learns from verified training programs. Evaluate both on the same development problems at the same exposure.

**Acceptance Scenarios**

1. **Given** complete failed searches and independent working programs for cases in the same difficulty range, **when** the preregistered routing rule selects generation, **then** one baseline and one intervention receive matched training and inference budgets. The intervention and its training sources can be identified from the registration.
2. **Given** a generator trial is feasible, **when** it reaches 0, 1M and 4M additional training tokens, **then** the operator sees the baseline and intervention at the registered checkpoints, the actual full-update token counts, and the full-budget comparison at the decision point.
3. **Given** 30 of 32 cases in a primary group have a verdict at an intermediate look, **when** execution and evidence integrity are sound and the reserved work fits, **then** the run records the incomplete measurement and can continue to its registered exploratory endpoint. Promotion requires at least 31 of 32 measured verdicts in this example. A verified solve and a completed miss both count as measured.

### User Story 4 - Spend a GPU shift and recover the evidence (Priority: P1)

As the workstation operator, I want to know what is running, which expensive work can be reused, and whether stopping or restarting will preserve both results and costs.

**Why this priority** The additional 1M tokens in the trial 00 follow-up took 8.6 recorded worker minutes. Development and retention generation and scoring took 43.2. These are elapsed worker measurements. GPU occupancy needs its own measurement.

**Independent Test** Interrupt an accepted workstation rehearsal after a completed evaluation batch and during a learner update. Resume the same registration and independently verify the terminal report, completed work and resource journal.

**Acceptance Scenarios**

1. **Given** a restart, **when** the operator repeats the same Git-pinned command, **then** it reuses independently verified candidate, fit and score batches, restores full training state at a valid update boundary, preserves failed-attempt charges, and shows the next unfinished unit. A second concurrent launcher cannot claim the same work.
2. **Given** a large valid artifact, a cold start or a slow but progressing worker, **when** production-scale acceptance exercises that path, **then** the operator sees progress or delayed telemetry. A binding hard limit identifies the resource, its justification, measured usage, preserved evidence and recovery point.
3. **Given** a genuine shortage or changed runtime, **when** preflight checks capacity and identity, **then** the affected work pauses before expensive dispatch. After capacity is restored or the changed component is qualified, the normal continuation verifies applicable prior evidence and resumes eligible work.

### Edge Cases

- Two different programs may fit all 20 visible terms. Only the sealed final choice earns credit for predicting all 80 held-out terms. Alternative candidates scored afterwards measure the opportunity available to the selector.
- A case that supplies both a verified solution and an unfinished search carries both facts. A complete search with zero solves, a proven spent unsuccessful attempt and an interrupted or timed-out fit have distinct meanings.
- A 25M checkpoint may have been scored on a different roster, with another budget, or on confirmation data. Report its compatibility and coverage. An absent or incompatible score earns no paired result. Prior confirmation outcomes cannot select the new intervention.
- An empty or singleton candidate list provides no choice between selectors. Count it in the all-case denominator and report its frequency. A lack of fresh families or eligible training negatives is a measured deficit, with the affected branch deferred.
- If the device, runtime, candidate set, exposure or resource accounting differs between arms, publish the affected comparison as incomplete or invalid and preserve unaffected evidence.

## Requirements *(mandatory)*

### Functional Requirements

**Diagnosis and routing**

- **FR-001** The audit MUST account for every historical case by stable problem, family, stratum, checkpoint, scoring rule and search budget. It MUST distinguish verified solve, completed miss, unresolved work and unavailable evidence, and show gains and losses on common cases. Carry forward the recorded D and random controls, conventional solvers, six 1M results, trial 00 at 2M and availability of each of the four 25M checkpoints.
- **FR-002** For each supported diagnosis, identify the sealed proposal, fit and selection evidence, whether any admitted program could predict the continuation, and whether the selected program did. An independent working example can demonstrate expressibility. A search miss alone cannot establish that a problem is inexpressible.
- **FR-003** Before inspecting additional case outcomes, register the diagnostic order, comparable denominator, rule for sufficient evidence, allocation when causes coexist, and terminal action. Address a reproducible fitting or verification failure first on affected cases. Give a same-list CPU selector test priority when verified alternatives were passed over. Commission the matched generator experiment if the measured proposal bottleneck exceeds the recoverable selection opportunities on comparable cases and independent working examples demonstrate headroom. Count each case once per diagnosis and disclose overlap. Ties favor the cheaper sufficient diagnostic. If evidence cannot distinguish the choices, spend only on a preregistered discriminating replay when its result could change the action. Otherwise report the gap. A new inference pass must name the missing comparison and its budget.

**Learning, panels and fair comparisons**

- **FR-004** On the strict track, use eligible random initial weights and mechanically generated training demonstrations. Verified program targets and hard negatives come exclusively from training families. CPU search may construct positive examples and plausible, prefix-matching negatives on training families. At evaluation, candidate generation, fitting and selection may use only information available from the visible prefix. Exclude hidden terms, benchmark solutions, case identifiers as predictors, and OEIS metadata from learned choices. Verify the final executable program independently under the declared exact-integer semantics and original indices.
- **FR-005** Preserve the 128 exposed round027 problems as the historical benchmark. Freeze a separate development panel and difficulty bands before comparing candidates. Calibrate difficulty on disjoint synthetic calibration families using prospectively chosen properties such as program complexity, known executable solutions and conventional-solver performance. Include held-out parameter variants, new combinations of familiar operations, unseen structures and a separately reported OEIS transfer measure. Record detectable duplicates, shifted windows, family and composition links across roles using feature 013's separation rules.
- **FR-006** Freeze the incumbent and conventional controls using prior exposed evidence and a declared tie rule. Compare selectors on identical candidate identities and budgets, and include a reproducible shuffled-input control. At the full-budget decision use feature 013's registered reference profile of eight attempts, 60 seconds per portfolio, four seconds per fit and 1,024 generated tokens per attempt. Both arms share the same input, search opportunities and accounting. Hybrid methods divide a single total allowance among their components. These values describe the scientific comparison, with any change handled prospectively under FR-014. A union of retrospectively best outputs is a diagnostic upper bound.
- **FR-007** If generator training is selected, freeze a single intervention and its ordinary-training baseline, common architecture, initialization per seed pair, data exposure rules, optimizer settings and token accounting. The proposed common learning settings are trial 05's batch size 32, zero weight decay, constant baseline learning rate and zero dropout. Use inexpensive, prospectively fixed probes at 0 and 1M additional tokens and a full eight-attempt comparison at 4M. Preserve full-update overshoot, incomplete observations and all attempted work.
- **FR-008** Freeze family membership, seeds, number of looks, baselines and the following spending decisions before comparing outcomes. Keep development, independent confirmation, familiar-task retention and final assessment distinct. Use the same frozen fresh-family replication panel for both seed pairs and freeze it before the first replication score is read. When unresolved cases could reverse a spending decision, apply the replay rule first.

  | Observed exploratory endpoint | Authorized next action |
  | --- | --- |
  | Positive paired primary gain at the full search budget, comparable costs and no demonstrated retention regression beyond feature 013's two-percentage-point allowance | Use the reserved independent training seed to repeat the same comparison. For selection, also require a same-list gain over the existing selector and better observed yield with correct inputs than with shuffled inputs. |
  | The sign or eligibility depends on genuinely unresolved cases | Use the registered discriminating replay if it can settle that question. Otherwise record an inconclusive result and retain the incumbent. |
  | Both paired seeds show a positive gain, with acceptable conservative retention on fresh reserved families | Recommend separately budgeted 25M-scale confirmation. Keep the confirmation panel untouched until the design and candidate are fixed. |
  | Measured gain is absent, the second seed reverses direction, or retention regresses | Report the paired effects and retain the incumbent. Publish an inconclusive verdict when the evidence cannot support a stronger one. |
- **FR-009** For a formal gain, inherit feature 013's paired, family-aware decision. For the primary synthetic measure, first average verified solves within each family, then average families within each group. Give composition and unseen structure equal weight. Require a mean gain of at least two percentage points, a positive effect for each of two independent training seeds, at least 95% completion per compared arm, seed and primary stratum, at most two percentage points of conservative familiar-task retention loss per seed, and a positive multiplicity-adjusted conservative lower bound. Keep all cases in the denominator. Report parameter and OEIS transfer separately. The lower bound accounts for missing outcomes, family dependence and the number of registered comparisons. An independent 25M-scale confirmation requires its own prospective budget and authorization. [Experiment and advancement contract](../013-progressive-mathematical-discovery/contracts/experiment.md)

**Workstation operation and evidence**

- **FR-010** Use feature 013's campaign and trial ownership, qualification, freeze, acceptance, progress, accounting, checkpoint restoration and report verification. Identify the source and any host-side additions. Verify source, image, inputs, exposure and producer compatibility before reusing qualification, calibration, repaired panels, preparation or checkpoints. Qualify changed runtime behavior on the workstation and require round028's own freeze and acceptance before research dispatch. Provide exact Git and run commands that preserve the qualified donor checkout.
- **FR-011** Save independently verifiable candidate sets, completed fit and score batches, trial decisions and full-state training checkpoints at recoverable boundaries. A retry MUST establish the same inputs, policy, model, seed, costs and ownership before crediting saved work. Preserve original failure receipts, settled and outstanding charges, consumed tokens, optimizer state, random state and data position. Publish aggregates and evidence identifiers for sharing. Keep private sequences, programs and per-case traces on the workstation.
- **FR-012** Report CPU fitting and search time, GPU decoding and training time, scoring, waiting, elapsed worker time and device occupancy with their measurement basis. Distinguish useful work, startup/cleanup, repeated work and historical costs. Identify unavailable readings and overlap. Before an expensive stage, check its remaining disk, inode, checkpoint, projection and report needs against actual capacity, with credit for verified allocations already paid for.
- **FR-013** Add a hard limit or timeout only to protect a named critical invariant, workstation resource, or the agreed scientific comparison. Check inherited guards along the full path as well. Reuse a suitable guard and correct a demonstrably undersized one at its source. If observation suffices, report the measurement without a hard cutoff. For a new or tightened operational guard, record the harm it prevents, units and scope, evidence available, worst valid observation, margin and action at exhaustion. Use applicable actual production-test measurements when available, covering the intended workstation and qualified runtime, real artifact sizes, cold starts and slow valid cases. Where such measurements are unavailable and the guard is critical, use a documented, generously provisioned provisional bound based on the strongest available evidence, including the device or filesystem capacity. Mark unmeasured quantities explicitly and name the production check that will validate or revise the bound. Targets, typical timings and missing heartbeats are observations. None alone authorizes termination.
- **FR-014** Before a long run, exercise the critical boundaries on production-shaped data through publication and independent verification, including a large valid producer, slow progress, an actual shortage within a disposable reservation, and interruption and recovery. Demonstrate that the largest supported valid input and measured slow work can finish with headroom, that a genuine exhaustion stops only the affected work, and that the same entry point resumes it. Record measured margins. If a scientific time or search allowance changes, register a new matched comparison and retain its predecessor's result.
- **FR-015** Publish a mechanically verifiable report containing the evidence verdict, next action, paired counts and costs, input and checkpoint identities, and the effect of missing or unresolved cases. Report a negative, incomplete or inconclusive result when warranted. Scaling eligibility and software acceptance have distinct verdicts. [Decision and contingency contract](../013-progressive-mathematical-discovery/contracts/contingencies.md)

### Key Entities

| Entity | What it means for the researcher |
| --- | --- |
| Case and family | An indexed sequence, its permitted role, difficulty, transfer group and known exposure. |
| Candidate set | The exact programs available for a case, their origins, fit status, selection and cost at the moment of commitment. |
| Matched trial pair | Baseline and intervention with a common starting condition and prospectively matched budgets. A second training seed provides independent replication. |
| Run registration and recovery record | The frozen question, inputs, inference policy, permissions, resource allowances, completed work and evidence needed to reproduce or resume it. |
| Evidence verdict and next action | The strength of the observed comparison and the separately justified choice to replay, train, confirm, retain the incumbent or stop. |

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** A reviewer can account for 128 of 128 historical cases, all seven screening/follow-up measurements and the disposition of all four 25M checkpoints. The six unfinished trial 05 composition cases and the D/trial 05 paired differences can each be traced to their evidence or an explicit missing input.
- **SC-002** On the prospective selector panel, 100% of paired selections have identical eligible candidate sets, identical search allowances and reproducible normal and shuffled comparisons. An independent scorer can reproduce the selected programs' exact continuation verdicts. A deliberately different candidate list invalidates the paired claim.
- **SC-003** For each supported routing outcome, a reviewer can reproduce the bottleneck diagnosis, the registered next action and the affected resource allocation. In the generator branch both arms publish the registered looks, actual token exposure and paired full-budget endpoint, or a recoverable interruption or terminal deficit.
- **SC-004** A reviewer can reproduce family-weighted yields, paired gains and losses, completion, conservative retention, independent-seed effects and the applicable uncertainty bound from the frozen rosters. Only a candidate meeting FR-009 earns a confirmed-gain verdict. Finishing an exploratory trial with an inconclusive result is a valid outcome.
- **SC-005** In a workstation interruption rehearsal, repeating the launch credits every valid completed batch once, restores the last complete training state and reconciles every resource charge. An altered checkpoint or producer is detected before dependent work, and the final report independently verifies.
- **SC-006** Every new or tightened hard guard has the justification and evidence in FR-013. The operator can identify which guards have been production-tested and which rely on generous provisional bounds, with the intended validation case for each provisional value. The workstation rehearsal accepts representative slow and large valid work and publishes usable diagnostics on exhaustion. The operator can distinguish measured GPU occupancy from worker elapsed time in the final report.

## Assumptions

- Round028 is a new prospective study. Round027's six formal trial outcomes, trial 00's exploratory `STOP_AT_2M`, the corrected panels, the fixed/refreshed comparison and their original charges are historical evidence. The current reference is PR #72 at `2fb9954b01b14db021e0ef9b7bebf73f933ddc0d`. The qualified donor checkout is `77667e67176ae32fa16457d52565bae1dbe41afc`. [Round027 follow-up rules and workstation checks](../013-progressive-mathematical-discovery/validation/trial00-followup/README.md)
- One shared workstation GPU is available. The initial planning allocation is about one eight-hour GPU shift after qualification, with at least half targeted for evaluation and replication. Actual production measurements govern the reservation. If the comparison cannot fit, the operator sees the attainable scope and cost before dispatch. The shift carries no per-worker timeout.
- Round028 covers diagnosis, one matched learning intervention, rehearsal, evidence publication and a conditional recommendation for longer confirmation. Broader hyperparameter or architecture sweeps, large-scale OEIS training, automatic artifact deletion and final assessment belong to separately authorized work. Finite agreement on 80 held-out terms supports a scoped prediction result.
