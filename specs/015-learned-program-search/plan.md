# Implementation plan for round028 learned program search

**Branch** `015-learned-program-search` · **Date** 2026-11-25 · **Parent** feature 013, PR #72 · **Specification** [Round028](spec.md)

## Summary

The first decision costs no new training. On the same 128 historical problems, determine whether the search produced a correct executable program and chose a different one, whether no correct program entered a complete search, or whether a failed fit or missing record prevents either conclusion. Register the decision rule before reading further private outcomes. Count missing cases explicitly.

If the selector overlooked verified answers, train a small CPU ranker and compare three choices from **exactly the same programs**. If complete searches lacked independently witnessed solutions more often, train a matched pair of generators. The proposed generator treatment rewards the token that begins a verified solution over the token that begins a plausible alternative known to fail later, at the point where those two programs first differ. Both arms see the same training tokens and updates. The 0 and 1M looks are inexpensive observations. The 4M full-search endpoint determines whether to spend the reserved independent seed. A result that justifies retaining the current system is a completed research outcome.

Implement this as a new registered study on the existing campaign, trial, qualification, freeze, acceptance, accounting and reporting infrastructure. Round027 is an immutable donor. The plans for [scientific decisions](contracts/experiment.md) and [workstation operations](contracts/operations.md) specify the two kinds of acceptance independently.

## Technical context

| Area | Design |
| --- | --- |
| Language and dependencies | Python ≥3.11 project, standard-library host CLI, pinned container builds from the root Dockerfile and committed locks. Existing NumPy/PyTorch, native bigint worker and independent reference semantics. Current GPU base pins ROCm 7.2.1, Python 3.12 and PyTorch 2.9.1. Optuna 4.5.0 remains the 013 tuning dependency. The paired study uses frozen arms. No new host or model-service dependency. |
| Platform and state | One shared AMD Ryzen AI Max+ 395 workstation, 128 GB shared RAM, NixOS, Podman and ext4. Record actual device, filesystem, engine, driver, image IDs and available capacity. Use the existing physical workspace state/lock root and a separate verified candidate source snapshot. |
| Persistence | Existing content-addressed private files, freeze progress SQLite, RunStore, trial and budget journals, atomic output bindings and full-state checkpoints. Keep large producers by reference, order batch shards explicitly and stream verification. Release aggregates and sufficient statistics in a separately verifiable ZIP. |
| Scale and scope | 128 historical cases, seven trial measurements, recorded controls and four prior 25M checkpoints. Fresh development and replication families, separate familiar retention, one conditional learning intervention and two independent seed pairs if earned. Future 25M confirmation is a separate research decision. |
| Feasibility | Aim for one eight-hour GPU shift after qualification. Reserve at least half the planning allocation for evaluation and replication. Measure lease time and occupancy separately, including qualification and any historical charges as separately identified costs. Earlier training and evaluation worker times are useful priors for the rehearsal. |
| Scientific and operational constraints | 20 visible terms, original indices, one committed choice, 80 exact hidden terms. Full comparison uses the qualified eight-attempt profile. Inherited containment, storage, projection and reporting limits are audited end to end. Only add or tighten a hard guard for a critical invariant, on production evidence or a documented generous provisional basis. |
| Validation | Focused pytest contract and integration cases, independent arithmetic/scoring counterexamples, disclosure and archive-only verification, plus physical CPU/native and HIP rehearsal with the real pipeline. Distinguish document validation, software acceptance and scientific evidence in all reports. |

## Constitution check

The adopted [constitution v2.0.0](../../.specify/memory/constitution.md) governs this feature. The existing [feature 013 plan](../013-progressive-mathematical-discovery/plan.md) and [host contract](../../AGENTS.md) document the portable Docker/Podman implementation of workstation container isolation.

| Principle | Before design | After design and implementation evidence required |
| --- | --- | --- |
| I. Exact data and declared numerical semantics | Carry original indices, exact integers and qualified profile. | Design passes. A scorer executes final grounded bytes on the registered 20+80 inputs. Test large integers, offsets and numerical exceptions against independent expected results. |
| II. Language and independent acceptance | Preserve the declared grammar, operand encoding, solver semantics and one chosen executable. | Design passes. Bind token, source, grounding and verifier identities. Test a plausible prefix match that fails later and an unselected genuine solve. |
| III. Containment and recovery | Keep bounded native execution, external deadlines, journaled budgets and durable recovery. | Design passes. Exercise cold/slow work, a real shortage, interruption, original service allowance and publication recovery. Verify cost preservation and actual headroom. |
| IV. Workstation feasibility | Containerized dependencies, a single shared GPU lease, qualified images, real parameter update and full-state restore. | Design passes. Pin source separately from the donor and require a new campaign freeze/acceptance on the selected engine and device. The hardware evidence is produced during implementation. |
| V. Strict provenance and isolation | Strict random-start training on generic mechanical demonstrations. Isolate old exposed data, fresh panels, training, retention and final assessment. | Design passes. Test cross-family/shift audit, forbidden mounts and still-hidden truth at selection time. Censoring, incomplete searches and all-case denominators are explicit. |
| VI. Honest discovery and scoped proof | Report finite continuation and uncertainty. Keep verdict distinct from spending authority. | Design passes. Recompute paired effects, adjusted bounds and actions independently. The report records the supported finite horizon. |

No constitutional exception is needed. A passed planning gate carries no device, qualification, freeze or research authority.

## Project structure

| Feature artifact | Use |
| --- | --- |
| [research.md](research.md) | Phase 0 decisions, alternatives and measured production evidence. |
| [data-model.md](data-model.md) | Private/public records, identities, roles, state transitions and retry semantics. |
| [contracts/experiment.md](contracts/experiment.md) | Routing table, training interventions, family sampling, budgets and statistical decisions. |
| [contracts/operations.md](contracts/operations.md) | Planned CLI, admission, limits, durability, reporting and production acceptance. |
| [quickstart.md](quickstart.md) | Git-pinned operator commands and end-to-end validation scenarios. |

### Integration in the existing package

Paths below are relative to the repository. Extend an existing owner where it already authenticates a decision. Add a small study-specific implementation for new learning or routing logic adjacent to that owner. Runtime packaging must include and authenticate the new 015 contracts and effective configuration, in addition to the launcher bytes. Changes to installed code and included contracts participate in the new source identity.

| Responsibility | Existing owners and intended change |
| --- | --- |
| Portable dispatch and qualification | `scripts/foundation_run.py`, `scripts/foundation_local.py`, `src/oeis_learn/pipeline_validation/progressive_cli.py`, `progressive_commands.py`, `progressive_handoff.py`, `progressive_device_acceptance.py` and `experiments/source_snapshot.py`. Add the round028 operation and explicit source/state roots. Reuse image qualification, device receipt and workstation owner. Snapshot the 015 contracts for both source-tree and installed-wheel verification. |
| Frozen study and accounting | `src/oeis_learn/generalization/progressive/{campaign,graph,lifecycle,trials,trial_evidence,budgets,freeze,freeze_progress,periodic_recovery}.py`, `configs/foundation/progressive/` and `pipeline_validation/progressive_study.py`. Introduce a closed, versioned paired-study graph and exploratory observation. Keep native formal `COMPLETE` eligibility, legacy Optuna ordinals and recovery replay intact. |
| Historical join, data and panels | `specs/013-progressive-mathematical-discovery/validation/report-round027.py`, `audit-round027-screening.py` and `trial00-followup/` provide authenticated donor adapters. Implement the new read-side join against those immutable records. Reuse `progressive_research_census.py`, `progressive_preparation.py`, `generalization/progressive/{complexity,exposure,prepared_panels}.py` and `generalization/selector_audit.py`. |
| Candidate capture and selection | `src/oeis_learn/fresh/{bigint_diagnostics,candidate_archive,synthesis_policy}.py`, `generalization/progressive/baselines.py`, `pipeline_validation/{progressive_evaluation,progressive_role,progressive_roles,progressive_outputs}.py`. Reuse the complete-candidate retention option and exact admission. Add versioned case-boundary output, CPU ranker, shuffled-choice role and a hidden scorer. |
| Matched generator training | `pipeline_validation/{progressive_learner,progressive_training}.py`, `generalization/progressive/{models,training_rows,periodic_recovery}.py`. Extend the existing update with a versioned training-only objective, matched row schedule and full optimizer/RNG/data-state restoration. No change to historical objective dispatch. |
| Decision, report and observability | `generalization/progressive/{metrics,decisions,public_comparisons,report,report_transport,verify}.py`, `pipeline_validation/{memory,progressive_report_collect,progressive_workstation_state}.py`. Supply anonymous sufficient statistics, an independent reducer/verifier path, and nonterminating phase-aware telemetry. Keep original-format archives readable. |

### Implementation sequence and acceptance coverage

| Order | Deliverable through the normal entry point | Completion evidence |
| --- | --- | --- |
| 1 | Source/state split, phase-specific donor import, sealed diagnostic policy, private case matrix and reproducible routing. Inventory reused calibration, repaired panel and preparation closures. Reuse requires applicable source, sensitivity/producer compatibility, exposure and original costs. The 1,296-case historical calibration cannot supply fresh difficulty families. | Account for 128/128, seven looks, D/random/conventional controls, four 25M statuses and each of the six unfinished composition cases. Show missing data, both policies and each possible route. FR-001–003, FR-010, SC-001/003. |
| 2 | Build disjoint rosters and training pairs, freeze controls and difficulty, extend candidate/fit/choice/score roles and run the selector on a fixed bank. | Independent hidden scorer, shuffled donor and bank identity tests, witness/miss/unknown distinction and equal-budget live comparison. FR-004–006, FR-011, SC-002. |
| 3 | Add the gated paired generator, common token journal and versioned pilot/replication states. Wire cross-arm checkpoints and controls into the existing formal reducer. | Independently restore a real update, bind the 0/1M/4M looks, reject mismatched exposure, retain incomplete intermediate looks, and show the two seeds on one fresh replication panel. FR-007–009, SC-003/004. |
| 4 | Complete the Git launcher, stored freeze progress, durable unit recovery, resource admission, telemetry and private/public reporting. Rehearse the entire chosen path on the candidate before admitting research. | Cross-root owner contention, interruption at each handoff, actual shortage, large and slow valid inputs, independent archive replay and privacy test. Report measured or provisional margins. FR-010–015, SC-005/006. |
| 5 | Execute only the registered, feasible branch after candidate qualification, prospective freeze and campaign acceptance. | Verified terminal report with costs, remaining uncertainty, retention, next action and separate evidence and software verdicts. FR-008/009/015, SC-003/004. |

The full production acceptance matrix and precise spending decisions are in the contracts. A workstation measurement can block or narrow the proposed spend prospectively. It cannot erase a failed attempt, relax the scientific comparison after results or authorize a longer study.
