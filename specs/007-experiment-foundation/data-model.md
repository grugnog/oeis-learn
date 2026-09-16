# Data model: foundation/v1

All names below are planned types in `src/oeis_learn/experiments/models.py`. Keep legacy types in `data/models.py` readable for diagnostics; do not widen old schemas silently. The external shapes of the three critical trust-boundary records are also specified in [artifacts.schema.json](contracts/artifacts.schema.json); other records follow the tables and behavioral contracts here.

## Common value rules

- `schema_version` is exactly `foundation/v1`; unknown versions and unknown fields are rejected.
- A `Digest` is `sha256:` followed by exactly 64 lowercase hexadecimal characters. Object identity hashes canonical UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=False`, and no NaN/Infinity. Unless a narrower identity core is explicitly defined below, omit only the object's own identity field, not other provenance fields. References must form an acyclic graph. File identity hashes final bytes; do not embed a self-referential file digest.
- An `IntegerText` matches `^(0|-?[1-9][0-9]*)$`. Semantic validation converts with exact integer parsing and enforces the record's numerical profile. Floats, scientific notation, booleans, `-0` and leading-zero spellings are rejected.
- `index`, counts and nanosecond durations are nonnegative integers unless specifically described as original source indices; booleans are not accepted as integers by the Python validators.
- Artifact paths are relative to an immutable run/cohort root and cannot contain `..`, absolute paths or symlink escapes. References include content digests; existence and hash are checked before consumption.
- Deterministic IDs are digests; run IDs are generated UUIDs with immutable manifests. A change of profile/configuration/pool/runtime creates a new run, not a mutable manifest under the same identity.

## ExperimentContract

| Fields | Constraints / relationship |
| --- | --- |
| schema_version, contract_id | Common version and digest rules. |
| language_profile, numerical_profile, entrypoint_profile, resource_profile, codec_profile | Required content-addressed profile references; all artifacts in the run match these references. |
| source_revision, dependency_lock, base_image_digest, final_image_id, environment_report | Exact commit and artifacts captured after image resolution/build, before a qualifying run; unresolved tags are insufficient. |
| track, initialization, objective | Exactly `strict_generic`, `random`, `sft` for 007. Legacy/RL/assisted variants require different profiles. |
| observed_terms, total_terms, index_policy | Exactly 20, 100, `prefix_rebased_zero`. |
| dataset_manifest, benchmark_manifest, evaluation_protocol, effective_config | Required hashes; missing pool never triggers legacy generation. |
| enabled_subsystems | Allowlisted; solvers, regex optimizer, family scaffolds, online OEIS replay and legacy proof promotion are disabled. Inactive subsystem options are rejected, not ignored. |

## SequenceRecord, SplitGroup and views

| Entity | Fields and constraints |
| --- | --- |
| SequenceRecord (private) | source-qualified record ID, original signed index, exact consecutive indices, exactly 100 IntegerText values, source hash, offset provenance, eligibility and exclusion reason. Complete records have all values in signed-256 range. Source name/tags/formulas are not features. |
| SplitGroup (private) | group_id, sorted unique member IDs, exact edge evidence, representative ID, partition (`development`, `final`, `reserved`); representative is smallest source-qualified ID. Every eligible record belongs to exactly one component/partition. |
| CohortManifest (private) | source snapshot hashes, grouping profile, seed, requested/actual counts, all group references, disjoint representatives, exclusion counts/reasons, private truth artifact hash. Count shortfall is an error. |
| VisiblePrompt | kind=`visible_prompt`, schema_version, request_nonce, language_profile, observed_terms. **observed_terms contains exactly 20 IntegerText values**. No other field is permitted. Nonce is random opaque routing data and excluded from model features/seeds. |
| AdmissionDecision | program_id, status (`admitted`, `rejected`), reason, independent/production evidence references and reserved-prefix check result. Rejection cannot expose a matched heldout ID or continuation. |
| TrainerExample | program_id, codec_profile, exactly 20 conditioning values and body token IDs; loader resolves and verifies admitted pool identity. No OEIS identity, metadata or evaluation continuation. |

Grouping and split algorithms are normative in [evaluation.md](contracts/evaluation.md). Values beyond the first 100 remain source provenance only; they cannot silently enlarge the qualification horizon.

## ProgramRecord and pool

Fields: `program_id`, `generator_revision`, `generator_config_hash`, `sample_seed`, `sample_counter`, `language_profile`, `codec_profile`, `canonical_source`, `source_sha256`, `body_tokens`, `tokens_sha256`, `visible_terms`, `outputs_ref`, `production_evidence_ref`, `reference_evidence_ref`, `origin`. The pool manifest separately links `admission_decision_ref`; it is not inside the hashed program identity.

`program_id` hashes only the immutable core `(generator_revision, generator_config_hash, sample_seed, sample_counter, language_profile, codec_profile, source_sha256, tokens_sha256, origin)`. Independent outputs/evidence are separate append-only records referencing this ID; AdmissionDecision references the program, and the pool references both, so no identity depends on its own admission decision. `origin` is exactly `generic_sample` for admitted 007 records. `body_tokens` is nonempty, contains EOS exactly once at the end, contains no BOS/padding/unknown tokens, and is at most 1,024 tokens. Exactly 100 independently verified synthetic outputs are archived; the trainer sees only the first 20. Decode then encode must reproduce the same canonical source/token sequence. Canonical source is at most 64 KiB. Deduplicate exact source hashes and exact 100-output fingerprints; for a pool's duplicate output class keep shortest token body then smallest source hash, chosen before finalizing the pool.

Pool manifest stores generator parameters/seeds, all attempted/accepted/rejected counts, rejection reasons, ordered admitted program IDs, the reserved-prefix set identity, and validation/runtime identities. A frozen pool is read-only; no asynchronous online generation is part of 007 SFT. Pool construction may use workers but publishes decisions in ascending sample-counter order. No family label or target-driven routing is allowed.

## CandidateResult

The machine-readable schema in [artifacts.schema.json](contracts/artifacts.schema.json) defines required fields, outcome enum and prohibited extras. Relationships enforced semantically:

- `stage` is `prepare`, `prefix` or `full`; matches require the corresponding stage and exactly 20/100 outputs.
- `attempt_index` is unique within a sealed target evaluation. The combination of run, checkpoint, prompt, attempt and stage is an idempotency key; duplicate messages cannot change an existing result.
- `source_sha256` hashes the final candidate, never an earlier skeleton. `checkpoint_sha256` is required for model evaluation and is null only for labeled conformance fixtures; null results cannot enter primary metrics.
- `selected` is frozen before full-stage verification. `proof_status` is exactly `not_claimed`.
- `reason` is null for matches and a nonempty string for failures. Costs record measured values or an explicit unavailable reason; no fabricated zeros.
- `verified_terms` cannot exceed output length or 100. Full match requires exact equality with all truth values and independent evidence; a schema-valid shape alone is insufficient.

## CheckpointManifest and training payload

`CheckpointManifest` external schema identifies final blob bytes, contract/pool/codec/runtime/config identities, completed update, run lineage and the budget ledger position. The binary payload contains model, optimizer, scheduler (if active), scaler (if active), Python/NumPy/CPU/all-active-GPU RNG states, named RNG stream states, epoch/permutation/cursor, next sample IDs, gradient-accumulation boundary, and every active counter. Disabled scheduler/scaler states are explicit null, not silently missing. 007 checkpoints occur after a completed optimizer step, with no partially accumulated gradients.

The frozen pool means no online replay/reservoir state exists. Pool construction has its own durable sample counter, dedup state and immutable chunks if resumed; these are not implicitly restored by a training checkpoint. Future online generation must add its active state in a versioned contract before being called resumable.

Checkpoint state transition: write temporary blob → flush/fsync → compute final blob hash → atomically rename to immutable final blob filename and fsync directory → atomically write/fsync manifest → atomically replace latest pointer → fsync directory. Never hash a blob, embed its hash and rewrite the blob. A recovery scan verifies newest to oldest, uses the newest complete matching manifest/blob pair, and records any rollback. Partial/corrupt files are quarantined. Preserve two valid generations and a separately pinned final checkpoint within quota.

## RunLedger, metric and finalization

Ledger events contain run ID, strictly increasing sequence number, UTC event time, monotonic interval duration when applicable, event kind, artifact references and typed metrics. One controller is the writer; workers send bounded messages. Metric values have `state` (`measured`, `disabled`, `unavailable`), `value`/unit/source, and reason for nonmeasurement. A measured zero is valid; disabled/unavailable values are null.

States: `created → preflight → ready → running → paused|completed|failed`; `paused → running` requires exact resume validation. `hardware_not_ready` is a preflight failure, not CPU readiness. All fixture/override/legacy runs remain `diagnostic`, not qualified. Qualification requires every acceptance gate and an adopted constitution profile; it is separate from run lifecycle.

Budget accounting separates measured active wall time from conservatively charged crash gaps. The external controller durably records a start and one-second heartbeats independently of blocked learner/runtime calls. Clean pauses exclude downtime; after an unclean exit, charge the entire unobserved interval from the last durable heartbeat through recovery, including uncertain downtime, and label it estimated. Use same-boot monotonic time, or recorded UTC across boots; backward/inconsistent clock evidence blocks qualifying resume instead of refunding time. An external launcher enforces the remaining-arm deadline even if the learner/controller stops responding. Image download/build is separately recorded and excluded; bootstrap, validation, updates, development evaluation, checkpointing and active recovery count once the arm starts. Clock reset on resume is prohibited. Recovery merges the durable ledger high-water mark with checkpoint state: model/data may roll back, but spent budget cannot. Abandoned updates are marked uncommitted and never double-counted. Deterministic tests use an injected clock and a checkpoint-boundary interruption; they do not assert identical real elapsed times.

`FinalizationManifest` is an immutable decision lock fixing checkpoint hash, run/contract, cohort/protocol, selector, stopping decision, ledger high-water mark and timestamp; it does not yet contain candidates. Final evaluation first receives only visible final prompts, generates and durably writes an immutable `CandidateSeal` per target linked to that decision lock, then permits scoring of that sealed target. A seal contains attempts/order, final sources, prefix evidence and selected candidate. A target without a seal cannot be scored. A committed seal is reused on retry; generation interrupted before sealing may resume only while no hidden feedback has been exposed for that target. If feedback was exposed without a valid seal, invalidate final evaluation rather than regenerate. The final report references the decision lock and all target seals. Evaluation progress is durable by target/attempt, independently of training checkpoints.
