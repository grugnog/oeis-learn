# Feature Specification: Remove Unused Repository Content

**Feature Branch** `014-repository-cleanup`

**Created** 2026-11-24

**Status** Draft

**Input** Remove unused code, scripts, configuration, documentation and other tracked files where removal has no effect on supported work. Deprecate entangled code when removing it would require a risky refactor.

**Stack base** `013-progressive-mathematical-discovery` at `a2618bbd`. [PR #72](https://github.com/grugnog/oeis-learn/pull/72)

**Four operations must survive independently.** An operator can start a fresh spec 013 campaign, resume an accepted campaign from saved work, monitor it, and independently verify its terminal report. The last recorded round027 handoff had the report watcher waiting and research still awaiting resume. Screening, confirmation and final assessment are required even though that handoff had yet to reach them.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Remove an obsolete workflow with confidence (Priority: P1)

A maintainer can remove an old experiment and its exclusive files, explain the decision to a reviewer, and keep a shared dependency operational. Git supplies the historical copy.

**Why this priority** The payoff is a smaller working tree with fewer misleading commands and documents. The number of lines deleted says little about whether a change is safe.

**Independent test** Review the old `scripts/run_hpo_sweep.py` experiment, its `configs/hpo_search_space.yaml` and the two `reports/hpo_sweep_results.*` outputs as one candidate group. Check who still uses each item, including any relationship to current screening. For an early run summary such as `runs/001_baseline_cold_start/reports/summary.md`, check whether a current claim still cites it.

**Acceptance Scenarios**

1. **Given** a retired workflow whose remaining references are confined to its own instructions, tests and outputs, **when** a maintainer establishes that the group has no current or specifically required future use, **then** the group can be removed together, surviving instructions resolve, and the affected neighboring workflows pass their baseline checks.
2. **Given** an apparently unused helper, **when** a saved run, installed command, CI job or documented operator action selects it, **then** the review identifies that consumer and preserves the operation. If obsolete behavior shares code with that operation and extraction would risk a regression, a discoverable deprecation note identifies the boundary and why the code stays.
3. **Given** two guides for the same operator command, **when** one contains only superseded instructions and the other covers all supported use, **then** the maintainer can delete the stale guide, update incoming links and follow the surviving instructions successfully.

### User Story 2 - Start a fresh campaign (Priority: P1)

An operator can prepare the pinned inputs in a clean workspace, qualify the environment, freeze the campaign, obtain acceptance and start the first eligible research task.

**Why this priority** A resumed workspace can mask missing packaged files, build inputs and setup commands.

**Independent test** Exercise the documented workstation and installed foundation paths using a clean workspace and bounded inputs. Observe the preparation gates and the first permitted dispatch.

**Acceptance Scenarios**

1. **Given** valid registered inputs and qualification for the selected source and device, **when** the operator prepares and starts the campaign, **then** the expected roster freezes, acceptance binds to that campaign, and the first eligible task receives its declared inputs.
2. **Given** missing or mismatched qualification, **when** the operator requests `--run-after-prepare`, **then** the failed gate is visible and no research task starts.

### User Story 3 - Resume saved work and reach the next decision (Priority: P1)

An operator can recover an interrupted, accepted campaign using its original evidence and carry its completed work forward exactly once.

**Why this priority** Losing a checkpoint or charging a finished task twice can change which trials the campaign can afford and invalidate its comparison.

**Independent test** Use a copy of the round027 recovery evidence, or a representative saved-state fixture with the same decisions and counts. Compare task identities, checkpoints and charges before and after recovery, including a second continuation attempt.

**Acceptance Scenarios**

1. **Given** the stopped diversity comparison, 47 completed task outputs and four full-state checkpoints, **when** the approved continuation verifies the producers and records the comparison, **then** its disposition is `INCOMPLETE / RETAIN_BASELINE`, prior work and charges are preserved, and `trial-00-propose` is the next eligible task. Repeating the request cannot duplicate publication, charging or dispatch.
2. **Given** a mismatched producer, checkpoint, owner or source identity, **when** continuity is checked, **then** the operator sees the failed prerequisite before further work can run against that evidence.
3. **Given** saved outcomes for screening, pruning and finalist selection, **when** each eligible path advances, **then** the original decision rules select the next trial or confirmation segment and final assessment uses the locked selection. A fixture can exercise these outcomes independently of the live run.

### User Story 4 - Monitor a campaign without taking it over (Priority: P1)

An operator can inspect an existing round once or watch it while another process owns the run.

**Why this priority** A waiting watcher and a running research invocation are different operational states. Mistaking one for the other can leave a campaign idle.

**Independent test** Observe copied preparation, interrupted and active-screening records in one-shot and interval modes. Compare the display with those records and check the run journal and budget before and after observation.

**Acceptance Scenarios**

1. **Given** a selected root and round, **when** the operator checks progress, **then** the display shows the recorded freeze and acceptance gates, task transitions, screening status and available learner milestones, including the last recoverable checkpoint where recorded.
2. **Given** a waiting or interrupted run, **when** a watcher starts, **then** it reports the available state and waits or diagnoses the interruption as appropriate. Observing the run starts zero tasks and adds zero charges.

### User Story 5 - Verify what a terminal report actually establishes (Priority: P1)

A reviewer can use the published bundle to check the scientific result independently of the process that produced it, including when the run ended incomplete.

**Why this priority** A verified candidate, a completed search and a completed scientific comparison support different claims.

**Independent test** Publish and verify a representative full-sized terminal bundle. Repeat with an incomplete comparison and with a required producer removed or altered. Compare the evidence, accounting and verdict with the baseline.

**Acceptance Scenarios**

1. **Given** a terminal invocation and its original producers, **when** the operator collects the report and a reviewer independently verifies the bundle, **then** both can trace the claims and costs to those producers, and the verdict preserves the distinction between candidate verification, search completion and comparison completion.
2. **Given** incomplete or altered evidence, **when** the reviewer verifies the bundle, **then** a missing required producer is identified and claims that depend on it remain unverified. Substituting a producer with different bytes fails the integrity check. Private scoring inputs stay outside the public report.

### Edge Cases

- A reference from a superseded runbook may disappear with its workflow. A filename in a CI partition, an operator command or a saved preparation proof is an active dependency even when a text or coverage search finds no conventional caller.
- A later round can authenticate an earlier helper by its exact bytes. Editing its docstring to mark it deprecated could invalidate recovery. Place the note outside the authenticated file in that case.
- A source change can leave calculations identical while invalidating a qualification binding. The accepted run uses its pinned source. A fresh run on changed source follows the documented qualification transition.
- A test may be the only independent check on rejection of invalid evidence. Keep that check for as long as the behavior is supported. A test exclusive to a retired workflow can be reviewed with it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** Review tracked code at file and symbol level, scripts, configuration, docs, tests and fixtures, CI/build/packaging assets and checked-in outputs. For each proposed deletion or related group, record the paths or symbols, former purpose, evidence checked for current and required future use, disposition and affected workflow check. Check direct callers and indirect use through installed interfaces, CI, build or package manifests, configuration, saved state, operator instructions and provenance records. Related files may share one rationale. An unresolved consumer keeps the affected candidate open for investigation.
- **FR-002** Remove an item when the evidence establishes it serves no supported workflow or concrete future requirement. For duplicate material, identify the surviving source and confirm it covers every still-needed instruction or use. Leave superseded material to Git history. Base the size of the cleanup on demonstrated safe removals.
- **FR-003** Retain demonstrably used behavior. Mark an obsolete but entangled portion deprecated when its safe removal would require refactoring a live path. State what is deprecated, the live dependency at risk, any supported replacement and the condition under which removal can be reconsidered. The marker must preserve existing invocation, output and authenticated bytes.
- **FR-004** Group deletions so a reviewer can validate and revert each group independently. Include directly affected references and exclusive tests. For supported workflows across specs, compare the affected entry points, CI checks, installed resources and operator instructions before and after. Withhold a group if its safety check fails or the evidence is unavailable. Independent groups can proceed.
- **FR-005** Preserve the four campaign operations and their negative cases in Stories 2–5. Both the installed `oeis-learn` interface and the separate workstation/foundation commands are consumers. Preserve Docker and Podman operation, the qualification and GPU build targets, installed assets and the test paths the existing qualification partitions invoke.
- **FR-006** Preserve the frozen research design and the operations chosen by saved state or configuration. The 352-task design includes capability and diversity baselines, four model-sizing trials, 12 learning-setting trials, pruning, selection, linked confirmation and final assessment. In the streamlined design, sizing evaluates only at 4M tokens, learning settings retain intermediate evaluations, and the second 12.5M-token confirmation segment continues the first from full state. Preserve role-specific input access, selected-device execution and exact numerical semantics throughout.
- **FR-007** Preserve recovery from accepted state, including source and image bindings, preparation proofs, task ownership and leases, full checkpoint state, reservations and committed charges, and disk byte and inode accounting. Preserve gated reuse of the original calibration, repaired panels and previously completed work. A new source identity requires its own applicable qualification. Historical receipts keep their original scope and status.
- **FR-008** Preserve the existing progress record formats, root and round selection, snapshot and interval monitoring, and independent terminal watcher. The resume helper acts only on an explicitly approved continuation. Monitoring and collection can operate separately from the qualified training checkout.
- **FR-009** Preserve independent publication and verification of historical and current report records at campaign scale. Exercise a roughly 49 MB calibration input, four large comparison projections and a journal of roughly 290,000 budget events through publication **and** verification. For any affected limit, record the resource it protects, observed full-sized demand and the outcome at the boundary. Preserve provenance, private input boundaries and the original interpretation of incomplete evidence.

### Baseline dependencies to check during review

These examples identify specific dependencies on the stack base. Review neighboring files on their own evidence.

| Existing consumer | What the cleanup must preserve |
| --- | --- |
| Workstation and foundation commands | Input registration, rehearsal, preparation, service/training/runtime measurements, research, assessment and collection. Preserve donor-session options, `--expanded-runtime-budgets`, `--streamlined-research`, `--run-after-prepare`, and the foundation `discovery` lifecycle including pause, resume, report and `verify-bundle`. Operators also fetch the pinned corpus with `data fetch --config configs/foundation/oeis_corpus.json`. [Workstation launcher](../../scripts/discovery_hip_handoff.py) · [Foundation runner](../../scripts/foundation_run.py) · [Command inventory](../../src/oeis_learn/pipeline_validation/progressive_cli.py) |
| Qualification, installed workers and shared execution | The handoff loads `qualify_bigint_phase5.py` by filename. CI invokes phase 2 and progressive qualification partitions. Frozen tasks select installed worker operations by name, with declared input channels and GPU assignments. These paths consume packaged policies and schemas, native crates, locks, build fixtures, `docker/dev/`, shared training/checkpoint code and BigInt generation, fitting and scoring. [Handoff](../../src/oeis_learn/pipeline_validation/progressive_handoff.py) · [Foundation CI](../../.github/workflows/foundation.yml) · [Role registry](../../src/oeis_learn/pipeline_validation/progressive_roles.py) · [Package data](../../pyproject.toml) |
| Retained inputs and predecessor evidence | Spec 012 checkpoint D supplies the starting model and B the control, with original producers in `pac-012-bcd-r1-002`. Spec 013 uses registration 003, HIP rehearsal 007, census/preparation 002, the 1,296-case calibration 009, corrected panels 016 and subsequent recovery receipts. Operators must be able to establish this chain while keeping private panels separate from public diagnostics. [Retained research inputs](../013-progressive-mathematical-discovery/validation/research-inputs.md) · [Panel adoption](../013-progressive-mathematical-discovery/validation/round016-panel-campaign.md) |
| Round-specific recovery | Round016 panel and reservation repair, round019 storage inspection versus continuation and crash recovery, round020/023/024 rehearsals, and round024/025 launches have distinct purposes. Preparation-owner versions select successors bearing 019, 020, 022, 023, 024 or 025 in their names. The round025 successor handles both round026 and the round027 retry. [Validation helpers](../013-progressive-mathematical-discovery/validation/) · [Successor selection](../../src/oeis_learn/pipeline_validation/progressive_research_launch.py) |
| Round027 replay, resume and reporting | After replay, `git show` retrieves `continue-round027-publication.py` from `9030a2bc`. Its `--continue` action uses the qualified checkout at `77667e67`. Separately, `git archive` supplies `report-round027.py` **and its matching `src/` tree** from `a2618bbd`. Its `--wait` action can collect and verify without switching the training checkout. [Comparison replay](../013-progressive-mathematical-discovery/validation/replay-round027-comparison.py) · [Pinned resume helper](https://github.com/grugnog/oeis-learn/blob/9030a2bcaefc083a2686850cd635a054da8bfa43/specs/013-progressive-mathematical-discovery/validation/continue-round027-publication.py) · [Pinned reporter](https://github.com/grugnog/oeis-learn/blob/a2618bbd1ff908a1d720f47268edf4234852c5d8/specs/013-progressive-mathematical-discovery/validation/report-round027.py) |
| Monitoring and diagnostics | The progress reader accepts `--root`, `--round`, `--once` and `--interval`. Operators also inventory disk use, retire qualified diagnostic tensors with receipts, and recover timing from existing outputs. [Progress reader](../013-progressive-mathematical-discovery/validation/research-progress.py) · [Disk inventory](../../scripts/discovery_cleanup.py) · [Retirement](../../src/oeis_learn/pipeline_validation/qualification_retirement.py) · [Timing collector](../../tools/research_timing_diagnostics/collect.py) |

### Key Entities

- **Cleanup candidate** — a symbol, file or related group and the evidence for removing, retaining or deprecating it. A consumer can be a person following a supported procedure, a build, a test, a named operation or a saved run.
- **Accepted campaign** — a frozen task roster and its bound inputs, qualification, resource limits, ownership and checkpoints. These identify which work may resume and which costs have already been incurred.
- **Evidence chain** — original producers, identities, receipts and the public report derived from them. It lets a reviewer establish both the result and the limits of the claim.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** The working tree contains fewer obsolete files or symbols than the baseline. Every removal maps to a reviewed disposition and an affected-workflow check. Reviewers can identify the net reduction, follow all surviving references touched by each group and reverse that group independently. Every deprecation identifies a specific dependency and a reconsideration condition.
- **SC-002** Four independently repeatable demonstrations pass for fresh start, accepted-run resume, monitoring and independent report verification. Fresh start reaches the first permitted task on valid inputs. A missing prerequisite allows zero research dispatches.
- **SC-003** Recovery accounts for 47 of 47 saved outputs and four of four full-state checkpoints in the round027 case. Previously completed tasks are rerun zero times and their original charges counted once. The next screening proposal, linked confirmation and final assessment each have a reachable, evidence-backed case.
- **SC-004** Both monitoring modes report the state recorded in their inputs. Starting either monitor changes zero task transitions, reservations or charges. The terminal watcher can wait while research is idle and collect after a terminal invocation.
- **SC-005** At the recorded campaign sizes, publication and independent verification agree with baseline claim scope and accounting. An incomplete comparison stays incomplete. Altering a required producer fails verification, and losing it leaves the dependent claim unverified.
- **SC-006** Each delivered cleanup group introduces zero new failures in the affected baseline checks and zero broken supported command, package-resource or documentation references. Readers can distinguish fixture, CPU/native and actual HIP evidence and identify any outstanding qualification.

## Assumptions

- The review covers the whole tracked repository on the PR #72 baseline, including earlier specs and checked-in reports. A documented current consumer or an accepted future stage justifies retention. Git provides access to superseded material. The shared constitution governs qualification, numerical semantics and interpretation of evidence. [Project constitution](../../.specify/memory/constitution.md)
- This is a maintenance decision. Research redesign, changed budgets or scoring rules, migration of accepted campaigns, refactoring to enable deletion, and disposal of live workstation data are outside its scope. Copied evidence or bounded fixtures can exercise the specified decisions without spending the live campaign’s budget. Record which checks used fixtures and which require real hardware or private originals.
