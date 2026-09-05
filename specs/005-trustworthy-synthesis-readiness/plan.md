# Implementation Plan: Trustworthy Synthesis Readiness

**Branch**: `005-trustworthy-synthesis-readiness` | **Date**: 2026-09-04 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from [spec.md](spec.md)

## Summary

This feature replaces disconnected demonstration and benchmark paths with one reproducible checkpoint-to-verdict workflow. A versioned evaluation protocol and frozen benchmark manifest drive deterministic candidate generation, constant resolution, canonical deduplication, native sandbox execution, exact 20-term fitting, exact 100-term extrapolation, compactness checks, and candidate-level failure attribution. The same result object powers the CLI, benchmark reports, readiness gates, and bounded ablations.

The plan also wires the already implemented adaptive curriculum into production orchestration, adds auditable run qualification and diagnostic overrides, introduces a bounded four-limb result profile for exact recurrence values beyond 64 bits, and separates mathematical discovery candidates, numerical conjectures, and symbolic proofs. Another production-length training run is explicitly deferred until the bounded experiments and constitutional readiness gates pass.

## Technical Context

**Language/Version**: Python 3.11+; Rust 2021 edition

**Primary Dependencies**: PyTorch 2.2+ in strict FP32; NumPy; DuckDB; PyArrow; PyYAML; SymPy 1.12+; mpmath 1.3+; explicit `z3-solver`; Wasmtime 20+, `wat`, Rayon, and PyO3 in the native evaluator; `jsonschema` in development dependencies for contract tests

**Storage**: Immutable versioned JSON manifests and reports for qualification evidence; existing DuckDB files for mutable sequence indexing; versioned `.pt` model checkpoints; Markdown generated only as a human-readable projection of JSON

**Testing**: pytest unit, contract, and integration suites; pytest-benchmark for latency and throughput checks; cargo test for native evaluator behavior; progressive preflight and a 1,000-candidate qualification smoke run

**Target Platform**: Linux x86_64 Tier 1 workstation with 4 CPU cores / 8 threads, up to 64 GB host RAM, and up to 4 GB GPU VRAM; generated programs target sandboxed WebAssembly

**Project Type**: Hybrid Python/Rust research library and command-line application with native sandbox execution

**Performance Goals**:

- identical fixed-seed semantic results across interactive and benchmark entry points;
- 100% candidate lineage coverage across an audit of at least 1,000 candidates;
- 100% syntactic assembly validity for grammar-constrained qualification candidates;
- candidate runtime-trap rate at or below 15%;
- no paired experiment trial longer than 4 hours and no more than 24 Tier 1 workstation-hours for the complete pre-production decision cycle;
- at least 10 percentage points improvement for resolved best-of-8 Stage 1 extrapolation over unresolved single-candidate inference;
- Stage 1 exact pass rate at least 80%, competence at least 0.85, minimum coverage at least 0.50, variance at most 0.01, and retained-task accuracy at least 95% before production promotion.

**Constraints**:

- all neural operations remain FP32 and within the Tier 1 4 GB VRAM profile;
- every generated candidate remains grammar constrained and uses a declared `i64_scalar_v1` or `i256x4_v1` WAT result profile;
- each `compute(n)` invocation receives at most 10,000 fuel and 16 MiB linear memory;
- qualification requires the native evaluator until fallback execution enforces equivalent limits;
- exact equality is required for 20 observed and 100 disjoint authoritative unseen terms; horizons cannot be shortened silently;
- candidate budgets are limited to 1, 8, or 16 and paired experiments use at least three seeds;
- no new production-length training run is part of this feature.

**Scale/Scope**:

- one frozen 32-48 sequence Stage 1 ablation cohort plus the constitutional Stage 1 promotion cohort;
- a 524-sequence mutable training pool;
- 16 cached candidates per prompt/seed for inference ablations;
- six short fixed/adaptive training trials of 500 allocation decisions each, 32 active rollouts per decision, and two replay examples;
- recurrence qualification on Fibonacci, Lucas, Pell, powers of two, and at least one leakage-free held-out linear recurrence;
- canonical pointwise integer linear discovery claims with affine index transforms and reviewed closed-form symbolic definitions.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle / Gate | Plan Requirement | Pre-Design Status | Evidence |
| :--- | :--- | :--- | :--- |
| I. Exact representation and FP32 | Preserve the configured tri-stream encoder, reject incompatible checkpoints, and run neural inference/training in FP32. | PASS | Checkpoint v2 records precision and complete encoder construction; this feature does not replace the encoder. |
| II. Sound grammar-guided WAT | Generate only through the environment-indexed masker; extend tracked result arity and recurrence phases without permitting free-form code. | PASS | Both result profiles remain WAT; qualification requires 100% assembly validity and retains No-Ghost checks. |
| III. Deterministic bounded sandbox | Use native GIL-free execution, reset at most 10,000 fuel per invocation, enforce 16 MiB memory, and classify every trap. | PASS | The protocol records backend and limits; non-equivalent fallback results are unqualified. |
| IV. Workstation-first feasibility | Bound candidate budgets, trial duration, total rollout counts, model size, and pre-production compute on Tier 1. | PASS | Paired trials use 500 decisions and equal 32-rollout budgets; the full decision cycle is capped at 24 workstation-hours. |
| V. Curriculum and anti-memorization | Require exact 20+100 verification, MDL <= 1.20, Stage 1 competence >= 0.85, coverage >= 0.50, variance <= 0.01, and pass rate >= 80%. | PASS | Frozen manifests prevent horizon truncation and leakage; the readiness policy blocks promotion below any threshold. |
| VI. Credit assignment and discovery | Preserve EGCA-GRPO and require latent search, PSLQ confirmation, exact held-out validation, and genuine symbolic reduction before proof status. | PASS | Adaptive orchestration calls the existing trainer; discovery evidence is separated into immutable promotion stages. |
| TDD and subsystem gates | Add failing unit/contract/integration fixtures before each implementation slice and retain Rust sandbox tests. | PASS | Test surfaces are defined below and in [quickstart.md](quickstart.md). |

No constitutional violation requires an exception or amendment.

## Project Structure

### Documentation (this feature)

```text
specs/005-trustworthy-synthesis-readiness/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/
│   └── requirements.md
└── contracts/
    ├── benchmark-manifest.schema.json
    ├── checkpoint-metadata.schema.json
    ├── cli-interface.md
    ├── discovery-report.schema.json
    ├── evaluation-protocol.schema.json
    ├── experiment-manifest.schema.json
    ├── readiness-report.schema.json
    ├── symbolic-definitions.schema.json
    └── synthesis-evaluation.schema.json
```

### Source Code (repository root)

```text
configs/
├── readiness_tier1_v1.json              # Versioned preflight and promotion thresholds
├── discovery_protocol_v1.json           # Frozen search/validation/proof settings
└── experiments/
    ├── trustworthy_inference_v1.json    # Solver and candidate-budget pairs
    └── trustworthy_curriculum_v1.json   # Fixed versus adaptive training pairs

src/oeis_learn/
├── cli/
│   ├── main.py                         # Thin command adapters and exit-code mapping
│   └── reporting.py                    # JSON authority and Markdown projections
├── curriculum/
│   ├── orchestrator.py                 # EXP3.S, Ada-G, replay, and trainer coordination
│   ├── scheduler.py                    # Constitutional competence and coverage metrics
│   ├── symple_bandit.py                # Task probabilities and budget-filled allocation
│   ├── extrapolation.py                # Detailed exact-horizon assessment
│   └── mdl_verifier.py                 # Post-assembly compactness evidence
├── data/
│   ├── benchmark.py                    # Frozen manifest loading and leakage checks
│   ├── models.py                       # Shared protocol, result, gate, and event entities
│   ├── synthetic_generator.py          # Diverse recurrence demonstrations
│   └── symbolic_definitions.py         # Versioned reviewed formula registry
├── decoder/
│   ├── sampler.py                      # Candidate-local generators and real top-p filtering
│   ├── constant_solver.py              # Unified solver dispatch and precise outcomes
│   ├── environment_tracker.py          # Result-profile and recurrence-frame constraints
│   └── wat_grammar.py                  # Scalar and four-limb WAT profiles
├── discovery/
│   ├── pipeline.py                     # Shared checkpoint-to-claim orchestration
│   ├── relation_identity.py            # Primitive canonical relations and triviality checks
│   ├── numerical_validator.py          # Exact partitioned relation validation
│   ├── pslq_solver.py                  # Coefficient confirmation only
│   ├── symbolic_prover.py              # Evidence-returning exact proof verifier
│   └── vector_search.py                # Latent candidate proposal only
├── evaluation/
│   ├── __init__.py
│   ├── checkpoint.py                   # Checkpoint v2 loading and legacy conversion
│   ├── protocol.py                     # Immutable protocols, hashes, and seed derivation
│   ├── synthesis.py                    # Sole candidate and cohort evaluation workflow
│   ├── readiness.py                    # Pure gate policy evaluation and overrides
│   └── experiments.py                  # Paired evaluation/training ablation runner
├── rl/
│   ├── trainer.py                      # Explicit per-prompt group execution
│   ├── progressive.py                  # Evidence-producing preflight tiers
│   ├── telemetry.py                    # Candidate and curriculum event telemetry
│   └── elite_buffer.py                 # Separate active/replay visits and retention
├── sandbox/
│   ├── runner.py                       # Typed result profiles and native limit forwarding
│   ├── optimizer.py                    # Truthfully named normalization/backend provenance
│   └── fallback_runner.py              # Diagnostic-only until limit parity exists
└── tracking/
    └── run_manager.py                  # Qualification lifecycle and immutable report paths

crates/oeis_wasm_evaluator/
└── src/
    ├── lib.rs                          # Typed scalar/four-limb Python boundary
    └── sandbox.rs                      # Per-invocation fuel, memory, and trap evidence

scripts/
├── build_benchmark_manifest.py         # Offline frozen 120-term cohort builder
├── run_long_e2e_benchmark.py           # Thin production-training/evaluation adapter
├── run_progressive_validation.py       # Versioned readiness policy runner
└── run_trustworthy_ablations.py         # Manifest-driven paired experiments

data/benchmarks/
├── trustworthy_synthesis_v1.json       # Frozen target terms and provenance
└── symbolic_definitions_v1.json         # Reviewed formula definitions and domains

tests/
├── contract/
│   ├── test_benchmark_manifest_contract.py
│   ├── test_discovery_report_contract.py
│   ├── test_experiment_manifest_contract.py
│   ├── test_readiness_report_contract.py
│   └── test_synthesis_evaluation_contract.py
├── integration/
│   ├── test_synthesis_entrypoint_parity.py
│   ├── test_readiness_override.py
│   ├── test_ablation_fairness.py
│   ├── test_recurrence_qualification.py
│   └── test_discovery_pipeline.py
└── unit/
    ├── test_checkpoint_loader.py
    ├── test_evaluation_protocol.py
    ├── test_synthesis_pipeline.py
    ├── test_readiness_policy.py
    ├── test_curriculum_orchestrator.py
    ├── test_recurrence_tracker.py
    ├── test_wide_result_profile.py
    ├── test_relation_identity.py
    ├── test_numerical_validator.py
    └── test_symbolic_prover.py
```

**Structure Decision**: Extend the existing monolithic Python package and native Rust evaluator rather than introduce another service. Cross-entry-point evidence belongs under a new `evaluation` package; training allocation remains under `curriculum`; mathematical claim verification remains under `discovery`. Scripts and CLI handlers are adapters over these package APIs.

## Implementation Strategy

### Phase A: Evidence Foundation

1. Define protocol, benchmark, synthesis-result, readiness, experiment, and discovery contracts.
2. Add immutable domain models, canonical JSON hashing, checkpoint v2, and explicit legacy conversion.
3. Build and validate the frozen 120-term benchmark plus leakage fingerprints.
4. Add deterministic candidate-local seed derivation and checkpoint/runtime provenance.

### Phase B: Unified Synthesis and Qualification

1. Implement the shared candidate state machine and detailed verifier results.
2. Route CLI synthesis and benchmark evaluation through the shared pipeline.
3. Add scalar and four-limb result decoding, per-invocation fuel reset, and observable memory-limit evidence.
4. Replace permissive preflight decisions with the versioned readiness policy and unqualified override lifecycle.
5. Verify entry-point parity and the 1,000-candidate syntax/runtime gate before further training work.

### Phase C: Controlled Curriculum Experiments

1. Refactor the trainer to execute an explicitly sized prompt group and return structured outcomes.
2. Add the curriculum orchestrator and wire EXP3.S, budget-filled Ada-G, feedback, dormancy replay, and append-only events.
3. Implement manifest-driven paired inference and short-training comparisons.
4. Require complete seed/variant pairs and readiness-policy success before producing an authorization decision.

### Phase D: Recurrence Readiness

1. Add the recurrence transition frame and scalar/four-limb grammar profiles.
2. Generate varied, grammar-emittable, leakage-screened recurrence demonstrations.
3. Validate complete old-state-to-new-state rotations and bounded progress before loop backedges.
4. Qualify the four named canaries and one held-out recurrence under the shared pipeline.

### Phase E: Discovery Integrity

1. Add canonical primitive relation identity and deterministic deduplication.
2. Separate latent proposal, exact numerical validation, and symbolic verification.
3. Add the reviewed definition registry and explicit claim state transitions.
4. Route CLI and benchmark discovery through one pipeline and derive Markdown from contract-valid JSON.

## Test and Validation Plan

- Write unit tests first for each new pure model, hash, transition, gate, allocator, relation, and proof rule.
- Add contract tests before producers emit each new JSON artifact.
- Add CLI/benchmark parity and failure-injection integration tests before removing hard-coded paths.
- Run focused Python tests after each implementation slice and native Rust tests after ABI or sandbox changes.
- Run the existing full Python and Rust suites after integration; unrelated pre-existing failures are reported but not repaired within this feature.
- Finish with the progressive readiness suite, 1,000-candidate qualification smoke test, and bounded three-seed ablation manifest. A production-length run remains blocked unless every mandatory gate passes.

## Post-Design Constitution Re-check

| Principle / Gate | Design Verification | Status |
| :--- | :--- | :--- |
| Exact representation and FP32 | Protocol and checkpoint schemas fix precision and reject incompatible construction; four-limb output preserves integer exactness. | PASS |
| Sound grammar-guided WAT | Both result profiles and recurrence phases are represented in the environment-indexed grammar and tested for 100% assembly. | PASS |
| Deterministic bounded sandbox | Native contracts carry memory/fuel settings and classify each invocation; fallback cannot qualify without parity. | PASS |
| Workstation-first feasibility | Cached candidates, fixed budgets, bounded trials, and explicit 24-hour ceiling prevent uncontrolled compute expansion. | PASS |
| Curriculum and anti-memorization | Frozen data, leakage checks, exact horizons, MDL, competence, coverage, variance, pass-rate, and retention gates are all explicit. | PASS |
| Credit assignment and discovery | Existing EGCA remains in training; discovery requires PSLQ confirmation and exact symbolic evidence without fallback promotion. | PASS |
| TDD and quality gates | Unit, contract, integration, native, progressive, and smoke validations map directly to feature requirements. | PASS |

No complexity exception is required. The new packages separate existing ownership concerns without introducing a new deployable service or bypassing constitutional architecture.
