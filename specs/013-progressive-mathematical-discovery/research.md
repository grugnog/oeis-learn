# Research decisions — progressive mathematical discovery

Reviewed 2026-11-08 against the merged 012 code and spec 013. Decisions below resolve design questions; actual workstation counts, source bytes and performance are measured inputs to freeze, not guessed facts. No research run was executed.

## R1 — New authority over existing execution

**Decision**: Introduce discovery-campaign/runtime/task/v1 and a separate discovery CLI group. Preserve historical validation and share installed computation/projection/supervision through version dispatch.

**Rationale**: `generalization/execution.py::_pac_policy` fixes 25M tokens and fresh initialization; `pipeline_validation/generalization_training.py::run_torch` rejects PAC replay/initializers and generic durations over 21,600s. `generalization_roles.py::invoke` has a six-hour default and a specific PAC envelope exception. The new goals need a different scope, not deleted assertions.

**Alternatives**: A temporary shell workflow cannot establish cumulative authority or safely recover. Rewriting a second trainer loses proven exact arithmetic and recovery behavior. Changing old schemas would corrupt historical meaning.

## R2 — Reproducible structural refresh with fixed allocation bins

**Decision**: A counter-keyed typed-AST generator emits finite verified shards. Freeze operator/depth/control-flow bins and retain D-derived EXP3.S eight-update allocation in those bins with explicit familiar/frontier floors. Both static/refreshed arms share this new policy.

**Rationale**: `generalization/refresh.py::RefreshPlan` only changes coefficients. `data/bigint_programs.py::BigintSampler` has four structural patterns; its canonical structure, rendering and independent evaluation are reusable, its distribution is not sufficient. `generalization_curriculum.py::PacController` binds a fixed family inventory, and PAC preparation fixes 32,768 rows/32 families.

**Alternatives**: Adding arms dynamically changes the kernel/probability clock and recovery. More constants alone do not test structural diversity. Unbounded online generation makes rejected work and exact continuation hard to account for. The new bins are explicitly D-derived, not a claim of identical historical D curriculum.

## R3 — Canonical optimizer/update semantics

**Decision**: Refactor a common logical optimizer-update primitive for the new profile, keeping legacy calls unchanged. Weight accumulated microbatch per-program means by their example counts; clip, optimizer-step, LR-step and curriculum clock advance once per complete logical batch. Checkpoint only at those boundaries.

**Rationale**: `rl/sft_trainer.py::supervised_step` zeroes and steps every call. Repeating it is not gradient accumulation. `run_torch` constructs AdamW with only LR explicit. Scheduler/dropout/regularization and RNG must be represented in state and qualification.

**Alternatives**: Batch clipping changes token equality; equal microbatch averaging misweights a partial microbatch. Treating interrupted gradients as committed work makes recovery scientifically inconsistent.

## R4 — Research model constructors and modest tuning

**Decision**: New research profile resolves the real retained recipe, validates dimensions, and supports declared dropout. Search 12 learning trials plus four discrete sizes, then confirm at most three nonbaseline finalists plus baseline using two seeds.

**Rationale**: `pipeline_validation/model_sizes.py` and `experiments/bootstrap_models.py::construct_sizing_models` are hardware-only profiles. Their encoder/dropout restrictions and forced decoder dropout zero cannot silently stand in for research choices. Actual parameter counts and finite HIP updates must be checked.

**Alternatives**: A large architecture search is disproportionate before diverse-data evidence. Warm-starting different shapes confounds sizing and lacks compatible optimizer state. Equal tokens and equal wall time answer different questions; report both.

## R5 — Optuna in a CPU proposal role

**Decision**: Use pinned Optuna ask/tell, serial proposals, a single scalar capability objective and explicit eligibility gates. Four learning trials form the startup tranche, including two enqueued controls, then seeded TPE proposes the remaining eight. Size candidates are enumerated. Median pruning is based on matched token coordinates; baseline/control remain unpruned.

**Rationale**: The official [ask/tell interface](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html) separates proposals from execution. [Storage documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) warns that ordinary study storage does not preserve sampler state. The host must not import Optuna.

Persist ordered canonical proposal/result events before dispatch/after scoring. For each new ordinal reconstruct an in-memory study from committed history, seed the sampler from study identity + ordinal, then ask once. Seal the full parameter proposal before launch. After interruption reuse that proposal rather than ask again. Exact Optuna version/distributions/history order are part of compatibility; test byte-identical proposal replay. In-memory reconstruction is a deliberate adapter policy, not a claim that database resume automatically restores RNG.

**Alternatives**: No external database/server or parallel GPU trials. Avoid opaque untrusted sampler pickle as evidence. Multiobjective built-in pruning is not assumed; costs remain separate reported axes. Pin the implementation-time qualified release in locks instead of relying on a floating docs version.

## R6 — Conditional uncertainty and exposure accounting

**Decision**: Two seeds must agree in direction. Family-cluster paired bootstrap is conditional on those seed realizations; report seed effects separately. Use conservative censoring and predeclared simultaneous bounds for finite finalist/milestone comparisons. Separate confirmation panels by decision stage where support permits.

**Rationale**: `generalization/metrics.py` already has censoring/family metrics and deliberately requires >=3 seeds for seed uncertainty. Do not weaken it. `cohorts.py`/`exclusions.py` only encode three roles and two holdout categories; use new membership contracts for development/confirmation and composition.

**Alternatives**: A task bootstrap cannot estimate training-seed population variation. Repeatedly probing one supposedly untouched panel introduces adaptive selection. A giant B/D replication is not required; targeted matched baselines remain necessary for new interventions.

## R7 — Immutable compatibility, mutable entitlements

**Decision**: Separate learning compatibility, campaign reservations and invocation receipts. Extend full state with stream/replay/library/admission cuts and cost cursors. Reuse Ledger, GeneralizationCheckpointStore and supervised crash accounting through new closed versions.

**Rationale**: Existing training compatibility hashes the complete config, including time allowance. Altering seconds cannot currently be exact resume. New task entitlement needs its own identity and no budget reset. Unknown crash elapsed time conservatively consumes the unproven remaining reservation.

**Alternatives**: Increasing a completed run's cap, merely creating a sibling output directory, or restoring weights without optimizer/RNG are not valid continuation. A 24-hour task cap is not a total campaign cap. The large maximum reservation in the execution contract must be visible before freeze; initial work may select only early stages.

## R8 — OEIS admission without imported teachers

**Decision**: A separate admission role verifies generated concrete programs against designated training-only continuation terms using native and independent reference execution. Use safe paired-target conversion afterwards, with literal fallback. Start with one bounded search/update cycle, task-balanced replay and a matched continued-synthetic control.

**Rationale**: Existing `generalization/replay_admission.py::admit` expects a mechanical oracle, 100 terms and a strictly shorter concrete target; `ReplayPool` and round planning are closed literal-only contracts. OEIS has no supplied mechanical target to shorten. New ancestry types are needed; never mark those records mechanical.

**Alternatives**: Prefix-only fitting gives spurious targets. Importing OEIS formulas violates the selected experimental track. Full RL adds an objective change before a supervised self-learning yield has been established.

## R9 — Useful components through transparent expansion first

**Decision**: Mine typed pure expression fragments across distinct verified training structures; guide bounded program search with them, expand hygienically into the existing source language before fitting/execution, and distill successful expanded programs into the learner. Freeze component IDs at state boundaries. No decoder vocabulary change is required for this initial complete implementation.

**Rationale**: `discovery/macro_mining.py` and `program_archive.py` target legacy logical IR/WAT, not the current bigint language. Reuse their bounds/design, not their qualification. Transparent expansion provides actual reusable-method search while avoiding token-dictionary migration as a prerequisite.

**Alternatives**: Mere offline compression does not implement stage 5. Opaque free calls hide complexity. A compact component-token decoder can be a later separate profile with explicit vocabulary migration and HIP qualification; it is not silently added here.

## R10 — Narrow proof support and refutation

**Decision**: Begin proof promotion with a separately bound checker for pure polynomial identities. Preserve exact assumptions/domain and distinguish mathematical-integer identity from checked-machine range safety. Other claims retain finite/conjectural status. Append dependency-aware quarantine records for refutations.

**Rationale**: Existing `discovery/certificates.py` and `certificate_checker.py` serve legacy IR and explicitly limit compiler linkage; they are not production bigint theorem authority. Existing novelty analysis can inform a sealed dossier, but its outputs remain training-ineligible.

**Alternatives**: No general automatic theorem-proving promise; no novelty from a changed source string. Supported narrow certificates and explicit unsupported states deliver honest useful review now.

## R11 — A feasible combinatorial pilot, not fake long sequences

**Decision**: Use a separate assisted combinatorial-short/v1 profile: exact totals or low-order moments of an allowlisted permutation statistic over complete collections for sizes 0..8; visible sizes 0..3, held-out training admission or evaluation sizes 4..8. Source-related tasks stay grouped. Freeze at most eight statistics and moment orders 1 and 2, subject to duplicate/semantic checks. Independently implement/check the bounded enumeration; stop with a deficit if too few distinct tasks support a learning comparison.

**Rationale**: [FindStat](https://www.findstat.org/) stores object/statistic maps; its [Sage interface](https://doc.sagemath.org/html/en/reference/databases/sage/databases/findstat.html) does not provide unlimited independently labeled 100-term sequence tasks. Full permutation enumeration at size 8 is bounded; arbitrary size-100 enumeration is not practical. A short-horizon pilot measures source feasibility, not the main 20/100 endpoint.

**Alternatives**: Padding coefficients with zeros or flattening arbitrary object order invents an easy extrapolation problem. LMFDB/Cremona and knot data initially receive suitability reports addressing integral values, object relations, finite support, access terms and verification cost. Acquisition scripts are allowlisted; source-provided code is not executed automatically.

## Unresolved runtime facts, not unresolved design

Actual retained checkpoint/recipe hashes, available fresh groups, source permissions/snapshot bytes, qualified package versions and measured resource envelopes are mandatory freeze inputs. Their absence blocks a run with a named deficit. No experiment defaults may substitute guessed values. Design decisions and interfaces are resolved sufficiently for task generation; no NEEDS CLARIFICATION remains.
