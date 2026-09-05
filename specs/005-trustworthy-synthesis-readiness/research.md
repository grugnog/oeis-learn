# Research: Trustworthy Synthesis Readiness

**Feature**: [spec.md](spec.md)  
**Branch**: `005-trustworthy-synthesis-readiness`  
**Date**: 2026-09-04

## 1. Shared Checkpoint-to-Verdict Workflow

### Decision

Create one side-effect-free synthesis evaluation pipeline under `src/oeis_learn/evaluation/`. It owns checkpoint loading, sequence encoding, deterministic candidate generation, optional constant resolution, structural canonicalization and deduplication, assembly, bounded execution, exact observed-term comparison, exact unseen-term comparison, compactness assessment, classification, and aggregation.

The interactive CLI, production benchmark, readiness evaluator, and ablation runner call this pipeline. Training continues to own gradient updates and may reuse lower-level candidate evaluation helpers, but it does not become the production inference API because it mutates curriculum and replay state.

### Rationale

The current CLI returns a hard-coded program, the benchmark has an independent partial inference path, and training is the only path that invokes constant resolution. A single workflow makes entry-point parity testable and prevents future stages from being skipped in one caller.

The public boundary is:

- `InferenceBundleLoader.load(...) -> InferenceBundle`
- `SynthesisPipeline.evaluate_target(target, protocol) -> SynthesisEvaluationResult`
- `SynthesisPipeline.evaluate_cohort(cohort, protocol) -> CohortEvaluationResult`

### Alternatives considered

- **Reuse `EgcaGrpoTrainer.train_step_for_prompt` directly**: Rejected because it performs optimization, scheduler updates, replay mutation, and reward construction.
- **Keep separate CLI and benchmark implementations**: Rejected because this caused the Run 007 mismatch and cannot guarantee semantic parity.
- **Put orchestration in the CLI module**: Rejected because readiness, experiments, and tests also need a non-command interface.

## 2. Immutable Protocols, Manifests, and Artifacts

### Decision

Use versioned JSON manifests as the authoritative source for qualification inputs and outputs. DuckDB remains the mutable training/index store; it is not authoritative for a frozen evaluation or discovery claim.

Each evaluation receives an immutable `EvaluationProtocol` containing checkpoint digest, vocabulary digest, benchmark manifest digest, observed and unseen horizons, result ABI, candidate budget, seed, sampling settings, enabled processing stages, solver limits, sandbox limits, compactness threshold, and deterministic-runtime metadata. Canonical JSON serialization produces a `protocol_id` SHA-256 digest.

Candidate and cohort reports are append-only JSON artifacts validated against contracts. Human-readable Markdown is generated from the JSON and never treated as the source of truth.

### Rationale

Run 007 artifacts do not retain enough information to reconstruct candidate lineage or determine which configured Phase 4 mechanisms were active. Immutable protocol and result objects make comparisons auditable and prevent configuration drift during a paired experiment.

### Alternatives considered

- **Add all evidence directly to existing DuckDB tables**: Rejected for this phase because existing databases have no migration framework and mutable rows are poor provenance records.
- **Treat Markdown reports as authoritative**: Rejected because Markdown omits machine-verifiable fields and is easy to overstate.
- **Store only aggregate metrics**: Rejected because failure-stage attribution and reproducibility require candidate-level lineage.

## 3. Checkpoint Compatibility and Deterministic Generation

### Decision

Introduce checkpoint format version 2 for qualified evaluation. It contains model constructor parameters, encoder and decoder state dictionaries, vocabulary SHA-256, precision, producer version, epoch, and optional training-state metadata. Qualified inference loads weights only, validates all metadata and tensor shapes strictly, and switches both models to evaluation mode.

Run 007 is migrated once through an explicit conversion command using its archived configuration. Legacy checkpoints are not loaded implicitly for qualification. Ambiguous legacy metadata is an error.

Derive each candidate seed from SHA-256 of `(base_seed, protocol_id, sequence_id, candidate_index)`. Candidate generation receives a local random generator rather than consuming global Python, NumPy, or neural-runtime state. Candidates are generated in stable index order. Budget 1 is therefore a prefix of budget 8, which is a prefix of budget 16.

Exact reproducibility is guaranteed within the same recorded runtime environment. Cross-device or cross-version comparisons are semantic comparisons of classifications and canonical programs, not promises of byte-identical floating-point logits.

### Rationale

Existing RL and SFT checkpoints omit a complete architecture and vocabulary contract. Global random state and active dropout also make the current benchmark non-reproducible. Versioned construction metadata and candidate-local seeds make failures diagnosable and paired budgets fair.

### Alternatives considered

- **Infer architecture from state-dictionary shapes**: Rejected because optional encoder modes and vocabulary semantics remain ambiguous.
- **Keep a permanent permissive legacy loader**: Rejected because silent defaults would undermine qualified evidence.
- **Reset global seeds before each command**: Rejected because caller order and unrelated random operations would still affect results.

## 4. Frozen Benchmark Cohort and Term Partitions

### Decision

Build `data/benchmarks/trustworthy_synthesis_v1.json` from versioned OEIS stripped data and authoritative b-files. The manifest records source revision, retrieval date, OEIS offset, signed exact integer terms, sequence family, curriculum stage, term digest, formula reference where available, and leakage fingerprints.

A target qualifies only when it has 20 observed terms and 100 disjoint unseen terms. Missing authoritative terms produce `INSUFFICIENT_TRUTH` and exclude the target from qualified aggregate denominators while retaining it in a separate insufficiency count. Verification never shortens a requested horizon silently.

For discovery, the manifest declares three disjoint partitions:

- search terms used to propose coefficients;
- validation terms used to reject accidental relations;
- unseen terms opened only after coefficients and support are frozen.

The existing local databases contain at most 50 terms and therefore cannot supply qualification evidence by themselves.

### Rationale

The current extrapolation verifier silently checks fewer than 100 unseen values when fewer terms are present. A frozen source revision, explicit offset, and exact partition ranges are required for reproducible anti-memorization and discovery evidence.

### Alternatives considered

- **Use current DuckDB rows directly**: Rejected because term counts range from 14 to 50 and offsets/provenance are incomplete.
- **Generate unseen terms from model-facing formulas**: Rejected for general qualification because this can couple the oracle to training data and excludes sequences without trusted formulas.
- **Allow shorter horizons and relabel them as 100-term checks**: Rejected as false evidence.

## 5. Exact Integer Result Profiles

### Decision

Support two declared WAT result profiles:

1. `i64_scalar_v1`: the existing `(param i32) (result i64)` profile for targets whose complete qualification horizon fits signed 64-bit integers.
2. `i256x4_v1`: `(param i32) (result i64 i64 i64 i64)`, representing one signed two's-complement 256-bit integer as four little-endian 64-bit limbs.

The host reconstructs an exact integer as

$$
U = \sum_{j=0}^{3} (\ell_j \bmod 2^{64})2^{64j}, \qquad
Y = \begin{cases}U-2^{256} & U \ge 2^{255}\\U & U<2^{255}.\end{cases}
$$

Each benchmark target declares the narrowest compatible profile. Qualification rejects any expected or generated value outside its declared range. The named recurrence canaries require at most 150 bits through index 119, so `i256x4_v1` covers them exactly.

Fuel is reset for every `compute(n)` invocation. Every invocation must consume at most 10,000 instructions; reports also record maximum and total fuel across the 120 calls. The 16 MiB memory ceiling remains per module instance.

### Rationale

Fibonacci at index 119 requires 82 bits, Lucas 83 bits, powers of two 120 bits, and Pell 150 bits. Modular 64-bit arithmetic or a shorter horizon would violate exact-integer and 100-term requirements. A fixed four-limb profile is bounded, uses the existing WAT target, avoids dynamic memory, and covers the specified canaries.

### Alternatives considered

- **Modulo-$2^{64}$ verification**: Rejected because modular equality is not exact OEIS equality.
- **Reduce the unseen horizon**: Rejected because it weakens the specification and anti-memorization gate.
- **Unbounded linear-memory big integers**: Deferred because they expand grammar and runtime complexity beyond the named canaries.
- **Host-side arbitrary-integer reinterpretation of overflowing scalar WAT**: Rejected because it would not reflect executable program semantics.

## 6. Candidate Identity, Stage Ordering, and Failure Taxonomy

### Decision

Candidate evaluation follows one fixed state machine:

`GENERATED -> RESOLVED_OR_NOT_REQUIRED -> CANONICALIZED -> ASSEMBLED -> EXECUTED -> OBSERVED_MATCH -> UNSEEN_MATCH -> COMPACT -> QUALIFIED`.

A failure ends qualification at one primary stage while preserving secondary diagnostics. Primary stages are `GENERATION`, `CONSTANT_RESOLUTION`, `CANONICALIZATION`, `ASSEMBLY`, `EXECUTION`, `OBSERVED_MATCH`, `EXTRAPOLATION`, and `COMPACTNESS`.

Canonical identity is the SHA-256 of a deterministic normalized WAT token stream after constant resolution. Compilation bytes and optimizer provenance are retained separately. Formatting-only duplicates execute once and point to the first canonical candidate through `duplicate_of`.

The existing regex optimizer is labeled `NORMALIZED_WAT_REWRITES`, not `wasm-opt` or compiler DCE. A qualification artifact may claim Binaryen optimization only when a recorded Binaryen backend actually ran. MDL size is computed only after successful assembly.

### Rationale

This ordering prevents an assembly success from being reported as mathematical success, prevents duplicate candidate budgets from overstating search, and removes stronger labels than the current implementation supports.

### Alternatives considered

- **Hash raw WAT**: Rejected because whitespace and harmless formatting would appear unique.
- **Use compiled bytes as the only identity**: Rejected because toolchain metadata can affect bytes and evidence still needs readable canonical WAT.
- **Assign multiple primary failures**: Rejected because operators need a stable funnel; secondary details preserve additional context.

## 7. Readiness Policy and Diagnostic Overrides

### Decision

Create a versioned `ReadinessPolicy` and a pure gate evaluator. Progressive tests collect evidence; they do not contain ad hoc promotion logic. Threshold records include comparator, value, unit, source, policy version, and whether the threshold is constitutionally non-relaxable.

Quick preflight gates validate mechanisms and reject known failure fixtures. They include nonzero exact synthesis in the single-prompt tier, complete candidate failure accounting, active adaptive-curriculum events, and no host instability.

Production promotion is a conjunction of:

- syntactic assembly validity = 100%;
- candidate runtime-trap rate <= 15%;
- Stage 1 rolling competence >= 0.85;
- minimum task coverage >= 0.50;
- competence variance <= 0.01;
- exact Stage 1 synthesis pass rate >= 80%;
- verified-task retention after 500 intervening steps >= 95%;
- all required paired experiments complete;
- no unresolved constitutional gate.

An explicit diagnostic override requires operator, timestamp, reason, intended diagnostic purpose, failed gate IDs, and policy hash. It moves a run to `OVERRIDDEN_UNQUALIFIED`; that run cannot graduate a curriculum stage, update a best-run pointer, or be reported as qualified.

Runtime trap rate is measured per candidate and separated into fuel exhaustion, execution trap, and resource-limit failure. Parse, assembly, and environment failures are separate counters. The current group-level `compiler_trapped` boolean is retired from qualification calculations.

### Rationale

Run 007 passed preflight with zero exact Tier 2 success and very low Tier 3 competence. Its final group-level trap measure reached 70%. Pure, versioned gates and explicit unqualified overrides make authorization behavior testable.

### Alternatives considered

- **Retain `--skip-preflight`**: Rejected because it loses intent and allows artifacts to appear qualified.
- **Gate only on loss, entropy, or reward variance**: Rejected because these are diagnostics, not proof of exact synthesis.
- **Use mean metrics alone**: Rejected because zero minimum coverage can be hidden by a favorable average.

## 8. Bounded Paired Experiments

### Decision

Run evaluation-only and short-training experiments from immutable manifests.

Evaluation ablations use the converted Run 007 epoch-60 checkpoint, a frozen 32-48 sequence Stage 1 cohort, and seeds `17`, `42`, and `101`. For each prompt/seed, generate and cache 16 ordered raw candidates once. Constant-resolution on/off variants consume identical raw candidates. Candidate budget variants use prefixes of that list.

Fixed versus adaptive training starts every trial from identical model, reference-model, optimizer, and replay snapshots. Each variant receives 500 allocation decisions, two active prompts, exactly 32 active rollouts per decision, and two replay examples. Fixed allocation uses seeded uniform prompt selection, 16 rollouts per prompt, and seeded uniform replay. Adaptive allocation uses EXP3.S selection, deterministic budget-filled Ada-G allocation, and dormancy replay.

Evaluate at decisions 0, 100, 200, 300, 400, and 500. Report every seed, paired differences, bootstrap 95% confidence intervals, worst seed, candidate evaluations, solver time, and wall time. Incomplete seed/variant pairs remain `PARTIAL` and never enter promotion aggregates.

### Rationale

This isolates one factor at a time, prevents best-seed selection, and is far cheaper than another 60-epoch run. Based on Run 007 throughput, the planned short-training trials remain comfortably within the 24-workstation-hour decision budget.

### Alternatives considered

- **Generate fresh candidates for each candidate budget**: Rejected because the comparison would confound budget with sampling noise.
- **Compare adaptive and fixed runs with different rollout totals**: Rejected because compute, not allocation strategy, could explain the result.
- **Proceed directly to another production run**: Rejected because Run 007 plateaued and intended mechanisms were inactive.

## 9. Active Adaptive Curriculum Semantics

### Decision

Add a curriculum orchestrator that owns task selection, group allocation, per-prompt trainer calls, feedback, and replay. The trainer accepts an explicit prompt and group size and returns a structured rollout result; it does not choose tasks.

Use normalized zone-of-proximal-development feedback

$$
r_i = 4\hat p_i(1-\hat p_i) + |\Delta C_i| + 2\max(0,-\Delta C_i).
$$

The factor 4 keeps the dispersion component in $[0,1]$ and matches the implemented scheduler. Ada-G first computes bounded ideal groups and then deterministically assigns any remaining rollout budget to prompts with the largest target-hit-probability deficit, preserving exactly 32 active rollouts.

Replay selection updates a distinct replay-visit timestamp. Selection probabilities, allocated groups, feedback components, replay choices, and retention checks are emitted as append-only events.

### Rationale

The Phase 4 scheduler, allocator, and dormancy functions exist but did not control Run 007. A single owner prevents configuration from being accepted while behavior remains inactive and guarantees fair compute accounting.

### Alternatives considered

- **Let the benchmark script coordinate each mechanism**: Rejected because scripts should be thin adapters and are difficult to unit test.
- **Keep group size as trainer-wide mutable state**: Rejected because Ada-G requires per-prompt allocation.
- **Leave unused budget unallocated**: Rejected because fixed/adaptive comparisons require equal rollout totals.

## 10. Recurrence State Transitions and Training Leakage

### Decision

Use a canonical unfolded recurrence form with current-state locals, next-state temporaries, and a progress counter. The environment tracker maintains a recurrence frame with phases:

`GUARD -> COMPUTE_NEXT -> COMMIT_ALL -> ADVANCE -> BACKEDGE`.

During `COMPUTE_NEXT`, current-state locals are read-only. All next-state values must be computed before any current-state commit. A backedge is legal only after every required state assignment and the progress update. This prevents accumulator swapping from destroying values that later assignments still need.

Extend the maximum generation length to at least 192 tokens for recurrence profiles. Generate diverse order-1 and order-2 demonstrations with varied seeds, signed small coefficients, offsets, and both scalar and four-limb result profiles. Canonicalize demonstration programs into grammar-emittable unfolded syntax before supervised training.

Freeze evaluation canaries before generating training data. Exclude records sharing a 120-term digest or canonical program hash with an evaluation target. Fibonacci, Lucas, Pell, powers of two, and the selected held-out recurrence are not seeded into the training replay buffer.

### Rationale

The current generator contains one Fibonacci template and a geometric template, while default replay includes the named canaries. The grammar tracks stack and scope but not transaction-like state updates. Explicit recurrence phases address the observed register-swap failure without adding new identifier tokens.

### Alternatives considered

- **Rely on more examples without grammar state**: Rejected because invalid partial commits remain admissible and receive execution feedback late.
- **Add a dedicated high-level recurrence DSL**: Rejected because direct WAT synthesis is a constitutional requirement.
- **Keep exact canary programs in replay**: Rejected because it invalidates generalization evidence.

## 11. Canonical Mathematical Relations and Numerical Evidence

### Decision

Represent each operand as `(oeis_id, index_scale, index_shift)`. For an integer linear relation:

1. reject missing operands, non-integer coefficients, all-zero coefficients, zero coefficients, repeated complete operands, and known aliases;
2. sort coefficient/operand pairs lexicographically by the complete operand reference;
3. divide coefficients by their absolute greatest common divisor;
4. flip all signs if the first coefficient is negative;
5. serialize canonical JSON and derive `claim_id` from SHA-256.

Permutation, global sign, and common scaling therefore map to one claim. Duplicate latent observations attach evidence to the existing claim rather than creating another claim.

Use exact multi-row integer nullspace computation over search terms to propose primitive coefficients. To satisfy the independent integer-relation requirement, PSLQ must reproduce the same primitive relation on deterministic 500-digit projections of the search rows. Freeze support and coefficients, then require exact zero residual at every validation and unseen index. No refitting after unseen access is allowed.

A support is nontrivial only when it is minimal: the full evidence matrix has nullity one for the proposed support and no proper operand subset has a nonzero null relation. Finite-term equality alone does not establish that differently named sequences are aliases; alias rejection requires a curated alias registry or identical authoritative definitions.

### Rationale

Current discovery infers coefficients from one term, checks only three indices, accepts zero coefficients, and counts traversal-order duplicates. Canonical primitive relations plus exact disjoint validation eliminate these failure modes.

### Alternatives considered

- **Use latent distance as relation evidence**: Rejected because distance only proposes candidates.
- **Use one-index PSLQ as verification**: Rejected because accidental scalar relations are common.
- **Treat equal finite prefixes as global aliases**: Rejected because finite agreement does not prove sequence identity.
- **Decompose reducible relations automatically**: Deferred; the first release rejects them with explicit evidence.

## 12. Claim Status and Symbolic Proof Policy

### Decision

Use this only promotion path:

`LATENT_CANDIDATE -> NUMERICALLY_VERIFIED_CONJECTURE -> SYMBOLICALLY_PROVEN_IDENTITY`.

`REJECTED` is terminal for a frozen claim version. Missing terms or definitions produce `INSUFFICIENT_EVIDENCE` without promotion.

Symbolic promotion initially supports reviewed exact closed-form definitions from a versioned local registry. Each definition includes source revision, source hash, index transform, domain, assumptions, structured expression, and parser version. The prover substitutes the canonical operand transforms and requires exact reduction of the full identity to structural zero over the intersection of declared domains. Missing definitions, parser failure, unsupported recurrences, or inconclusive simplification leave a claim numerical; there is no fallback proof text.

The authoritative discovery JSON records latent, numerical, and symbolic evidence separately. Markdown groups claims by status and cannot rename a conjecture as a theorem. Summary metrics count unique canonical claims, duplicates, conjectures, symbolic identities, rejections, and insufficient-evidence outcomes.

### Rationale

The current symbolic prover labels PSLQ-only fallback text as `PROVEN`, and Run 007 counts two permutations of one trivial relation. Evidence-separated immutable states prevent this overclaim.

### Alternatives considered

- **Treat exact finite validation as proof**: Rejected because a finite prefix cannot establish a general identity.
- **Accept free-form formula strings without provenance**: Rejected because index offsets, assumptions, and source revisions are not auditable.
- **Support recurrence and generating-function proofs immediately**: Deferred until dedicated initial-condition and domain proof methods are specified.

## 13. Technology and Validation Strategy

### Decision

Retain the existing Python 3.11+ and Rust 2021 hybrid architecture. Use PyTorch in strict FP32, DuckDB for mutable sequence indexing, JSON Schema contracts for external artifacts, SymPy exact matrices and symbolic reduction, mpmath PSLQ at 500 digits, Wasmtime/WAT/Rayon through the native evaluator, and pytest/cargo test for validation.

Add `z3-solver` as an explicit project dependency because constant resolution already imports it. Add no database migration in this feature. Any real Binaryen integration is separately capability-detected and recorded; fallback normalization cannot claim Binaryen passes. Qualification requires the native evaluator because the current fallback does not enforce the memory ceiling observably.

Testing is layered:

- unit tests for deterministic seeds, protocol hashes, checkpoint rejection, stage transitions, gate truth tables, allocation budgets, recurrence phases, relation canonicalization, and proof status;
- contract tests for manifests and JSON reports;
- integration tests for CLI/benchmark parity, diagnostic override behavior, paired experiment fairness, recurrence canaries, and discovery parity;
- Rust tests for scalar and four-limb ABI decoding, per-invocation fuel reset, memory limits, and trap classification;
- a 1,000-candidate qualification smoke run before bounded experiments.

### Rationale

These choices reuse the repository's installed stack, preserve Tier 1 operation, and turn existing claims into executable checks. Explicit dependency and backend provenance prevent capabilities from being inferred from names alone.

### Alternatives considered

- **Introduce a web service or distributed experiment platform**: Rejected as outside scope and unnecessary on Tier 1.
- **Use a new persistence framework**: Rejected because immutable JSON artifacts and existing run directories satisfy the evidence requirements.
- **Permit fallback execution for qualification**: Rejected until it enforces the same memory and result-ABI contracts as the native backend.

## Resolution Summary

All planning unknowns are resolved. The design does not weaken exactness, extrapolation, compactness, grammar guidance, sandbox isolation, or Tier 1 constraints. The four-limb result profile is the bounded compatibility extension required to make the named 120-term recurrence qualification mathematically possible.
