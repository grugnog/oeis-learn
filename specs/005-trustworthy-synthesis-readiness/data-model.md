# Data Model: Trustworthy Synthesis Readiness

**Feature**: [spec.md](spec.md)  
**Branch**: `005-trustworthy-synthesis-readiness`  
**Date**: 2026-09-04

## Conventions

- Internal exact sequence values use arbitrary-precision integers.
- JSON artifacts encode exact integers as canonical decimal strings matching `^-?(0|[1-9][0-9]*)$` so consumers cannot lose precision.
- All hashes are lowercase SHA-256 values prefixed with `sha256:`.
- Timestamps are UTC RFC 3339 strings.
- Durations are non-negative milliseconds.
- Enumerated state and reason values are uppercase strings.
- Immutable entities are replaced by a new version rather than edited after qualification begins.

## 1. Evaluation Evidence

### `CheckpointProvenance`

Identifies an inference-compatible checkpoint and the model contract required to reconstruct it.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `format_version` | String | Yes | `2.0` for qualified evaluation |
| `checkpoint_sha256` | String | Yes | SHA-256 digest of checkpoint bytes |
| `producer_version` | String | Yes | Non-empty repository/package revision |
| `epoch` | Integer | Yes | `>= 0` |
| `precision` | String | Yes | Must be `fp32` |
| `encoder_config` | Object | Yes | Complete constructor values; no inferred defaults |
| `decoder_config` | Object | Yes | Complete constructor values; no inferred defaults |
| `vocabulary_sha256` | String | Yes | Digest of ordered token vocabulary |
| `source_checkpoint_sha256` | String or null | No | Set when produced by explicit legacy conversion |
| `runtime_environment` | Object | Yes | Python, neural runtime, device, OS, and native evaluator versions |

**Rules**:

- Loading fails before candidate generation if the digest, precision, vocabulary, constructor metadata, or tensor shapes do not match.
- Optimizer state is not loaded for inference.
- A legacy checkpoint can become qualified only through an explicit conversion that writes a version 2 artifact and records the source digest.

### `BenchmarkSource`

Describes where authoritative sequence truth originated.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `name` | String | Yes | Source collection name |
| `revision` | String | Yes | Immutable release, commit, or snapshot identifier |
| `retrieved_at` | Date-time | Yes | UTC timestamp |
| `content_sha256` | String | Yes | Digest covering source inputs |
| `license_notice` | String or null | No | Source attribution reference |

### `BenchmarkTarget`

A frozen sequence target with enough truth for qualification.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `oeis_id` | String | Yes | `A` plus six digits |
| `name` | String | Yes | Non-empty |
| `offset` | Integer | Yes | OEIS index corresponding to the first stored term |
| `family` | String | Yes | Stable family label |
| `curriculum_stage` | Integer | Yes | `1..5` |
| `observed_terms` | List[decimal string] | Yes | Exactly 20 terms |
| `unseen_terms` | List[decimal string] | Yes | Exactly 100 terms immediately following observed terms |
| `result_profile` | Enum | Yes | `i64_scalar_v1` or `i256x4_v1` |
| `terms_sha256` | String | Yes | Digest over offset and all 120 terms |
| `formula_definition_id` | String or null | No | Reviewed symbolic definition reference |
| `term_fingerprint` | String | Yes | Leakage comparison key |
| `program_fingerprints` | List[String] | Yes | Canonical known-program hashes; may be empty |
| `tags` | List[String] | Yes | Normalized tags |

**Rules**:

- The result profile must represent every observed and unseen term exactly.
- Search or training data with the same `term_fingerprint` or a matching `program_fingerprint` is excluded from held-out evaluation.
- Targets lacking 120 authoritative terms are listed as manifest exclusions, not padded or shortened.

### `BenchmarkCohort`

A versioned collection of benchmark targets.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `schema_version` | String | Yes | Contract version |
| `cohort_id` | String | Yes | Stable human-readable identifier |
| `manifest_sha256` | String | Yes | Digest of canonical manifest excluding this field |
| `source` | `BenchmarkSource` | Yes | One frozen source snapshot |
| `observed_horizon` | Integer | Yes | Exactly 20 for qualification v1 |
| `unseen_horizon` | Integer | Yes | Exactly 100 for qualification v1 |
| `targets` | List[`BenchmarkTarget`] | Yes | Unique `oeis_id`; non-empty |
| `exclusions` | List[Object] | Yes | Sequence ID plus stable exclusion reason |

### `EvaluationProtocol`

Immutable instructions for one reproducible synthesis evaluation.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `schema_version` | String | Yes | Contract version |
| `protocol_id` | String | Yes | SHA-256 over canonical protocol fields |
| `checkpoint_sha256` | String | Yes | Must match loaded checkpoint |
| `benchmark_manifest_sha256` | String | Yes | Must match loaded cohort |
| `observed_horizon` | Integer | Yes | 20 |
| `unseen_horizon` | Integer | Yes | 100 |
| `candidate_budget` | Integer | Yes | One of `1`, `8`, `16` |
| `base_seed` | Integer | Yes | Signed 64-bit integer |
| `temperature` | Number | Yes | `>= 0` |
| `top_p` | Number | Yes | `(0, 1]` |
| `max_tokens` | Integer | Yes | `>= 1`; recurrence profile uses at least 192 |
| `constant_resolution` | Boolean | Yes | Explicit, never inferred |
| `solver_timeout_ms` | Integer | Yes | `>= 1` |
| `max_placeholders` | Integer | Yes | `>= 0` |
| `fuel_per_invocation` | Integer | Yes | `1..10000` |
| `memory_limit_mib` | Integer | Yes | `1..16` |
| `mdl_ratio_max` | Number | Yes | `<= 1.20` for qualification |
| `native_evaluator_required` | Boolean | Yes | Must be true for qualification |
| `code_revision` | String | Yes | Revision plus dirty-worktree marker |
| `environment_fingerprint` | String | Yes | Digest of relevant runtime versions and deterministic settings |

**Identity**:

`protocol_id = SHA256(canonical JSON of all fields except protocol_id)`.

### `StageRecord`

Evidence for one candidate-processing stage.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `stage` | Enum | Yes | `GENERATION`, `CONSTANT_RESOLUTION`, `CANONICALIZATION`, `ASSEMBLY`, `EXECUTION`, `OBSERVED_MATCH`, `EXTRAPOLATION`, `COMPACTNESS` |
| `status` | Enum | Yes | `NOT_RUN`, `NOT_REQUIRED`, `PASSED`, `FAILED`, `TIMEOUT` |
| `duration_ms` | Number | Yes | `>= 0` |
| `reason_code` | String or null | No | Stable uppercase code when not passed |
| `message` | String or null | No | Human-readable detail |
| `evidence` | Object | Yes | Stage-specific values; exact integers encoded as strings in JSON |

### `SynthesisCandidateRecord`

Complete lineage for one requested candidate index.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `candidate_id` | String | Yes | Deterministic from evaluation ID and index |
| `candidate_index` | Integer | Yes | Zero-based; `< candidate_budget` |
| `candidate_seed` | Integer | Yes | Derived from protocol and target identity |
| `raw_token_ids` | List[Integer] | Yes | Generated ordered tokens |
| `raw_wat` | String | Yes | Exact decoded candidate |
| `resolved_wat` | String or null | No | Present after successful constant resolution or copied when not required |
| `resolved_constants` | List[decimal string] | Yes | Empty when not required or unresolved |
| `canonical_wat` | String or null | No | Deterministically normalized grounded candidate |
| `canonical_sha256` | String or null | No | Present after canonicalization |
| `duplicate_of` | String or null | No | Earlier candidate ID with the same canonical digest |
| `stage_records` | List[`StageRecord`] | Yes | One record per stage in fixed order |
| `outputs` | List[decimal string] | Yes | Up to 120 reconstructed exact values |
| `max_fuel` | Integer or null | No | Maximum across invocations; `<= 10000` when executed |
| `total_fuel` | Integer or null | No | Sum across invocations |
| `peak_memory_mib` | Number or null | No | `<= 16` when qualified |
| `first_observed_divergence` | Integer or null | No | Relative index `0..19` |
| `first_unseen_divergence` | Integer or null | No | Relative index `0..99` |
| `byte_size` | Integer or null | No | Available only after successful assembly |
| `mdl_ratio` | Number or null | No | Available only after successful assembly |
| `classification` | Enum | Yes | `EXTRAPOLATING_SUCCESS`, `FAILED`, `DUPLICATE` |
| `primary_failure_stage` | Stage enum or null | Yes | Exactly one for `FAILED`; null otherwise |
| `secondary_diagnostics` | List[Object] | Yes | Additional non-primary observations |

**Candidate invariants**:

- Stage records are ordered and later stages are `NOT_RUN` after the primary failure.
- A duplicate references an earlier candidate and reuses its evaluation evidence without a second sandbox execution.
- `EXTRAPOLATING_SUCCESS` requires all eight stages to pass or be legitimately not required, exactly 120 outputs, exact equality, all per-call limits, and MDL ratio `<= 1.20`.

### `SynthesisEvaluationResult`

One target's candidate set and aggregate outcome.

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `schema_version` | String | Yes | Contract version |
| `evaluation_id` | String | Yes | Deterministic protocol/target identity plus execution attempt |
| `created_at` | Date-time | Yes | UTC |
| `protocol` | `EvaluationProtocol` | Yes | Embedded immutable protocol |
| `checkpoint` | `CheckpointProvenance` | Yes | Loaded checkpoint evidence |
| `target` | `BenchmarkTarget` | Yes | Frozen truth |
| `candidates` | List[`SynthesisCandidateRecord`] | Yes | Exactly requested budget entries |
| `unique_candidate_count` | Integer | Yes | Number of unique canonical candidates |
| `qualified_candidate_ids` | List[String] | Yes | Subset of candidates |
| `status` | Enum | Yes | `QUALIFIED_SUCCESS`, `COMPLETED_NO_SUCCESS`, `REQUEST_ERROR`, `EXECUTION_ERROR` |
| `duration_ms` | Number | Yes | `>= 0` |

## 2. Readiness and Run Governance

### `ReadinessThreshold`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `gate_id` | String | Yes | Unique within policy |
| `metric` | String | Yes | Stable metric key |
| `comparator` | Enum | Yes | `EQ`, `GE`, `LE` |
| `threshold` | Number | Yes | Finite |
| `unit` | String | Yes | Non-empty |
| `source` | String | Yes | Constitution/spec/policy reference |
| `non_relaxable` | Boolean | Yes | Constitutional thresholds are true |

### `ReadinessPolicy`

Contains `schema_version`, deterministic `policy_id`, policy name, all thresholds, required experiment IDs, required artifact contracts, and native-backend requirement.

### `ReadinessGateResult`

Contains gate ID, measured value, threshold snapshot, pass/fail, evidence artifact references, evaluation timestamp, and diagnostics.

### `OverrideRecord`

Contains immutable override ID, operator, UTC timestamp, reason, diagnostic intent, failed gate IDs, policy ID, and resulting unqualified status. Empty reasons and implicit overrides are invalid.

### `ReadinessReport`

Contains report ID, run ID, policy, gate results, aggregate `passed`, optional override, and resulting qualification state.

### Run State Machine

```text
INITIALIZED
    -> PREFLIGHT
        -> BLOCKED
        -> AUTHORIZED
            -> RUNNING
                -> COMPLETED_QUALIFIED
                -> COMPLETED_UNQUALIFIED
                -> FAILED
                -> INTERRUPTED
        -> OVERRIDDEN_UNQUALIFIED
            -> RUNNING_UNQUALIFIED
                -> COMPLETED_UNQUALIFIED
                -> FAILED
                -> INTERRUPTED
```

**Rules**:

- Only `AUTHORIZED` can lead to `COMPLETED_QUALIFIED`.
- An override is irreversible for that run ID.
- Blocked or unqualified runs cannot update graduation or best-run records.

## 3. Controlled Experiments and Curriculum

### `ExperimentManifest`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `schema_version` | String | Yes | Contract version |
| `experiment_id` | String | Yes | Stable unique identifier |
| `experiment_type` | Enum | Yes | `INFERENCE_ABLATION` or `TRAINING_ABLATION` |
| `status` | Enum | Yes | `PLANNED`, `RUNNING`, `PARTIAL`, `COMPLETE`, `FAILED` |
| `checkpoint_sha256` | String | Yes | Common starting checkpoint |
| `benchmark_manifest_sha256` | String | Yes | Frozen cohort |
| `seeds` | List[Integer] | Yes | At least three unique seeds |
| `variants` | List[`ExperimentVariant`] | Yes | At least two |
| `invariants` | Object | Yes | Settings held equal across variants |
| `decision_schedule` | List[Integer] | Training only | Exactly `0,100,200,300,400,500` for v1 |
| `max_trial_hours` | Number | Yes | `<= 4` |
| `max_total_hours` | Number | Yes | `<= 24` |
| `outcomes` | List[Object] | Yes | Empty while planned; per seed/variant afterward |

### `ExperimentVariant`

Contains variant ID, one declared changed factor, factor value, candidate source/cache identity, active rollout budget, replay budget, and all protocol overrides. Exactly one factor may differ within a paired comparison.

### Experiment State Machine

```text
PLANNED -> RUNNING -> COMPLETE
                   -> PARTIAL
                   -> FAILED
```

Only `COMPLETE` manifests with every seed/variant pair contribute to promotion decisions.

### `TaskTrainingState`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `oeis_id` | String | Yes | Valid OEIS ID |
| `pass_history` | List[0 or 1] | Yes | Maximum policy window |
| `competence` | Number | Yes | `[0,1]` |
| `competence_slope` | Number | Yes | `[-1,1]` |
| `bandit_weight` | Number | Yes | `>0` |
| `selection_probability` | Number | Yes | `(0,1]`; sums to one across tasks |
| `allocated_rollouts` | Integer | Yes | `8..16` when active |
| `last_active_step` | Integer | Yes | `>=0` |
| `last_replay_step` | Integer | Yes | `>=0` |
| `has_verified_elite` | Boolean | Yes | Derived from buffer |
| `retention_status` | Enum | Yes | `UNKNOWN`, `RETAINED`, `REGRESSED` |

### Curriculum Events

Append-only event variants are `TASK_SELECTED`, `ROLLOUT_ALLOCATED`, `CANDIDATE_EVALUATED`, `BANDIT_FEEDBACK_APPLIED`, `ELITE_ADDED`, `REPLAY_SELECTED`, and `RETENTION_EVALUATED`. Every event carries run ID, global step, sequence ID, timestamp, and event-specific evidence.

## 4. Recurrence Representation

### `ResultProfile`

| Value | WAT result | Exact host range |
| :--- | :--- | :--- |
| `i64_scalar_v1` | One `i64` | $[-2^{63},2^{63}-1]$ |
| `i256x4_v1` | Four `i64` limbs, least significant first | $[-2^{255},2^{255}-1]$ |

### `RecurrenceFrame`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `state_locals` | Ordered list[String] | Yes | At least one current-state local |
| `next_locals` | Ordered list[String] | Yes | One per state local |
| `progress_local` | String | Yes | Declared integer local |
| `required_commits` | Set[String] | Yes | Initially all state locals |
| `completed_commits` | Set[String] | Yes | Subset of required commits |
| `progress_advanced` | Boolean | Yes | False until a recognized bounded increment |
| `phase` | Enum | Yes | `GUARD`, `COMPUTE_NEXT`, `COMMIT_ALL`, `ADVANCE`, `BACKEDGE_READY` |

### Recurrence Transition

```text
GUARD
  -> COMPUTE_NEXT        when a bounded exit guard is established
  -> COMMIT_ALL          when all next-state temporaries are assigned
  -> ADVANCE             when every current-state local is committed once
  -> BACKEDGE_READY      when the progress local advances
  -> GUARD               after a legal loop backedge
```

Writes to current-state locals during `COMPUTE_NEXT`, repeated or missing commits, missing progress, and early backedges are invalid grammar transitions.

## 5. Discovery Evidence

### `SequenceRef`

A complete operand identity: OEIS ID, integer `index_scale`, and integer `index_shift`. Two references to the same ID with different transforms are distinct operands.

### `CanonicalRelation`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `relation_type` | String | Yes | `POINTWISE_INTEGER_LINEAR_V1` initially |
| `operands` | Ordered List[`SequenceRef`] | Yes | Lexicographically sorted, unique, at least two |
| `coefficients` | List[decimal string] | Yes | Same length, all nonzero, primitive GCD 1, first positive |
| `canonical_expression` | String | Yes | Deterministic display form |
| `claim_id` | String | Yes | SHA-256 of canonical relation JSON |

### `LatentEvidence`

Contains checkpoint and vocabulary digests, embedding layer/version, candidate source tuple, vector distance, search parameters, seed, and backend/version. It proposes a claim but cannot verify one.

### `NumericalEvidence`

Contains source and cohort digests, search/validation/unseen index sets, coefficient proposal method, PSLQ precision and coefficient bound, primitive coefficients, exact residual arrays, matrix rank/nullity, minimal-support result, first counterexample, and outcome.

Outcomes are `VERIFIED`, `COUNTEREXAMPLE`, `TRIVIAL`, or `INSUFFICIENT_EVIDENCE`.

### `SymbolicDefinition`

| Field | Type | Required | Validation |
| :--- | :--- | :---: | :--- |
| `definition_id` | String | Yes | Deterministic ID |
| `sequence_ref` | `SequenceRef` | Yes | Exact operand |
| `kind` | Enum | Yes | `CLOSED_FORM` initially |
| `source` | Object | Yes | URI/reference, revision, and content hash |
| `expression` | String | Yes | Exact reviewed expression |
| `domain` | Object | Yes | Integer lower/upper bounds or unbounded upper range |
| `assumptions` | List[String] | Yes | Explicit symbolic assumptions |
| `parser_version` | String | Yes | Registry parser contract version |

### `SymbolicProofEvidence`

Contains definition IDs and hashes, normalized identity, proof method, reduced expression, common domain, verifier version, outcome, timestamp, and diagnostic. Outcomes are `PROVEN`, `COUNTEREXAMPLE`, `UNSUPPORTED_DEFINITION`, `MISSING_DEFINITION`, `DOMAIN_MISMATCH`, or `INCONCLUSIVE`.

### `DiscoveryClaim`

Contains schema version, canonical relation, status, lists of latent evidence, one numerical evidence record, optional symbolic evidence, optional rejection, and append-only status history.

### Discovery State Machine

```text
LATENT_CANDIDATE
    -> NUMERICALLY_VERIFIED_CONJECTURE
        -> SYMBOLICALLY_PROVEN_IDENTITY
        -> REJECTED                 (symbolic counterexample only)
    -> REJECTED                     (triviality or numerical counterexample)
    -> INSUFFICIENT_EVIDENCE
```

Missing or unsupported symbolic definitions leave a claim `NUMERICALLY_VERIFIED_CONJECTURE`; they never produce a proof or automatic rejection.

## 6. Relationships

```text
CheckpointProvenance ─┐
BenchmarkCohort ───────┼─> EvaluationProtocol ─> SynthesisEvaluationResult
ReadinessPolicy ───────┘                              │
                                                      ├─> SynthesisCandidateRecord[]
                                                      └─> ReadinessGateResult evidence

ExperimentManifest ─> ExperimentVariant[] ─> EvaluationProtocol / TaskTrainingState
TaskTrainingState ─> CurriculumEvent[] ─> ReadinessReport

BenchmarkTarget ─> SequenceRef ─> CanonicalRelation ─> DiscoveryClaim
                                           │              ├─> LatentEvidence[]
SymbolicDefinitionRegistry ─────────────────┘              ├─> NumericalEvidence
                                                          └─> SymbolicProofEvidence
```

## 7. Data Retention and Authority

- Benchmark manifests, evaluation protocols, synthesis results, readiness reports, experiment manifests/results, definition registries, and discovery reports are immutable JSON evidence.
- Run metadata references immutable artifact IDs and digests; it does not duplicate or overwrite their content.
- Markdown reports are regenerable views and carry no independent status authority.
- Mutable training state, caches, and DuckDB indexes are never accepted as qualification evidence without export to a contract-valid immutable artifact.
