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
| track, initialization, objective | Exactly `strict_generic`, `random`, `sft` for the foundation/v1 smoke. Qualified extension runs use the explicit profile/state rules below; assisted data remains prohibited. |
| observed_terms, total_terms, index_policy | Exactly 20, 100, `prefix_rebased_zero`. |
| dataset_manifest, benchmark_manifest, evaluation_protocol, effective_config | Required hashes; missing pool never triggers legacy generation. |
| enabled_subsystems | Allowlisted per profile; the initial smoke disables solvers/optimization/RL. Only repaired and qualified extensions may enable them; family scaffolds and legacy proof promotion remain prohibited. Inactive subsystem options are rejected, not ignored. |

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

`program_id` hashes only the immutable core `(generator_revision, generator_config_hash, sample_seed, sample_counter, language_profile, codec_profile, source_sha256, tokens_sha256, origin)`. Independent outputs/evidence are separate append-only records referencing this ID; AdmissionDecision references the program, and the pool references both, so no identity depends on its own admission decision. `origin` is exactly `generic_sample` for foundation/v1 smoke records. The extension archive additionally supports generic-model and permitted-prefix hypotheses with distinct evidence and training eligibility, defined below. `body_tokens` is nonempty, contains EOS exactly once at the end, contains no BOS/padding/unknown tokens, and is at most 1,024 tokens. Exactly 100 independently verified synthetic outputs are archived; the trainer sees only the first 20. Decode then encode must reproduce the same canonical source/token sequence. Canonical source is at most 64 KiB. Deduplicate exact source hashes and exact 100-output fingerprints; for a pool's duplicate output class keep shortest token body then smallest source hash, chosen before finalizing the pool.

Pool manifest stores generator parameters/seeds, all attempted/accepted/rejected counts, rejection reasons, ordered admitted program IDs, the reserved-prefix set identity, and validation/runtime identities. A frozen baseline pool is read-only. The extension archive/replay has separate versioned state and deterministic admission ordering. Pool construction may use workers but publishes decisions in ascending sample-counter order. No family label or target-driven routing is allowed.

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

The frozen baseline pool has no online replay/reservoir state. Pool construction has its own durable sample counter, dedup state and immutable chunks if resumed; these are not implicitly restored by a training checkpoint. US6 online generation must persist the additional versioned state enumerated below before it is called resumable.

Checkpoint state transition: write temporary blob → flush/fsync → compute final blob hash → atomically rename to immutable final blob filename and fsync directory → atomically write/fsync manifest → atomically replace latest pointer → fsync directory. Never hash a blob, embed its hash and rewrite the blob. A recovery scan verifies newest to oldest, uses the newest complete matching manifest/blob pair, and records any rollback. Partial/corrupt files are quarantined. Preserve two valid generations and a separately pinned final checkpoint within quota.

## RunLedger, metric and finalization

Ledger events contain run ID, strictly increasing sequence number, UTC event time, monotonic interval duration when applicable, event kind, artifact references and typed metrics. One controller is the writer; workers send bounded messages. Metric values have `state` (`measured`, `disabled`, `unavailable`), `value`/unit/source, and reason for nonmeasurement. A measured zero is valid; disabled/unavailable values are null.

States: `created → preflight → ready → running → paused|completed|failed`; `paused → running` requires exact resume validation. `hardware_not_ready` is a preflight failure, not CPU readiness. All fixture/override/legacy runs remain `diagnostic`, not qualified. Qualification requires every acceptance gate and an adopted constitution profile; it is separate from run lifecycle.

Budget accounting separates measured active wall time from conservatively charged crash gaps. The external controller durably records a start and one-second heartbeats independently of blocked learner/runtime calls. Clean pauses exclude downtime; after an unclean exit, charge the entire unobserved interval from the last durable heartbeat through recovery, including uncertain downtime, and label it estimated. Use same-boot monotonic time, or recorded UTC across boots; backward/inconsistent clock evidence blocks qualifying resume instead of refunding time. An external launcher enforces the remaining-arm deadline even if the learner/controller stops responding. Image download/build is separately recorded and excluded; bootstrap, validation, updates, development evaluation, checkpointing and active recovery count once the arm starts. Clock reset on resume is prohibited. Recovery merges the durable ledger high-water mark with checkpoint state: model/data may roll back, but spent budget cannot. Abandoned updates are marked uncommitted and never double-counted. Deterministic tests use an injected clock and a checkpoint-boundary interruption; they do not assert identical real elapsed times.

`FinalizationManifest` is an immutable decision lock fixing checkpoint hash, run/contract, cohort/protocol, selector, stopping decision, ledger high-water mark and timestamp; it does not yet contain candidates. Final evaluation first receives only visible final prompts, generates and durably writes an immutable `CandidateSeal` per target linked to that decision lock, then permits scoring of that sealed target. A seal contains attempts/order, final sources, prefix evidence and selected candidate. A target without a seal cannot be scored. A committed seal is reused on retry; generation interrupted before sealing may resume only while no hidden feedback has been exposed for that target. If feedback was exposed without a valid seal, invalidate final evaluation rather than regenerate. The final report references the decision lock and all target seals. Evaluation progress is durable by target/attempt, independently of training checkpoints.

## Qualified extensions: qualified/v1

The preceding foundation/v1 records are the fixed SFT control. Extension runs use separate discriminated records in `experiments/qualified_models.py`; unknown fields/versions fail, and the common digest, exact-integer, path and identity rules still apply. All have `schema_version=qualified/v1`, `kind`, immutable content identity and source/config/profile references. No implicit conversion of legacy artifacts is allowed.

| Entity | Required fields and constraints |
| --- | --- |
| QualifiedContract | Base contract reference; qualified language/runtime/codec/model/objective/sampling/search/admission profile digests; gate-report digests; track=`strict_generic`, initialization=`random` or `strict_parent` with a complete parent chain rooted at random. objective=`sft`, `reinforce` or `grpo`. Solver/optimization activation requires G7; neural/replay/search activation G8; initial formula-identity certificates with an independent checker G7, richer program-class/compiler-link certificates and extended runtime G9. Settings for inactive modes fail. |
| GroundingResult | Final source/parameter map (if any), exact fit indices and visible prompt digest, solver method/version, coefficient bounds, elapsed metric, outcome=`verified_solution`, `proved_unsat_in_scope`, `unknown`, `timeout` or `unsupported`; proof/checker evidence required for scoped UNSAT, exact final execution evidence for a solution. Incomplete searches cannot be UNSAT. |
| ProofEvidence | Statement/expression/program digests, domain with excluded points, assumptions, index transforms, numeric semantics, evidence kind, checker/version, budget and result. Kinds=`formula_identity`, `bounded_machine`, `polynomial_program`, `linear_recurrence`, `finite_state`, `loop_invariant`. Result=`verified`, `counterexample`, `unknown`, `unsupported`, `timeout`, `empty_domain`. Verified requires the typed certificate plus successful independent checker report; counterexample requires an exact defined in-domain witness. No universal PROVEN or automatic novelty field. |
| ArchiveEntry | Source/IR/codec/runtime hashes, origin, generation parent/model/sample identity, exact conditioning values, available outputs, fit/independent-validation index manifests, finite evidence, training_eligible plus reason, token counts and costs. Origins=`generic_sample`, `generic_model`, `training_prefix_hypothesis`; imported/teacher origins never train. Up to four Pareto implementations per output identity in the training view; all valid distinct sources may remain in the quota-bounded immutable archive. Over-context sources are retained with training_eligible=false and never truncated. Every origin and macro-expanded program must pass the same reserved structure/parameter/composition and OEIS-prefix membership checks before training/retrieval eligibility. |
| RolloutRecord | Immutable behavior encoder+decoder hash, visible prompt hash, action IDs, per-action support/temperature/log-probability, forced/PAD/EOS masks, RNG identity, reward components and execution evidence. In 007 policy-gradient modes top_p=1, no top-k and scoring/rollout dropout off. Every sampled action has one stored finite behavior log-probability. |
| VisitObservation | Task/prompt training identity, successes, attempts, elapsed time, visit/step, selection probability and sampling rule; 0<=successes<=attempts, attempts>0. A visit is one aggregate update, not ordered success/failure pseudo-history. |
| QualifiedCheckpoint | Complete base state plus active reference encoder/decoder, replay/archive root and cursor, frozen self-training round, curriculum sufficient statistics/RNG, search frontier/cursors, sampler/prefetch state and active scaler/scheduler. Inactive state is explicit null. Streaming snapshots record only committed outputs and restorable pre-call state. Publication, integrity and nonrefundable ledger rules match foundation/v1. |
| RelationProposal | Ordered source/program references, transforms and exact coefficients, structural origin, fit/validation/final index manifests, exact rank/nullity when computed, residual evidence, attempt/cost counters and immutable seal. Each index set is nonempty and pairwise disjoint for confirmatory finite validation; combining duplicate operands is canonical before scoring. |
| MacroDictionary | Content-addressed permitted source archive root, typed expanded bodies, stable internal IDs, acyclic dependency closure, side conditions, equivalence evidence, expansion costs and description-length benefit. No OEIS solution calls or unpinned external dependencies. |
| ExperimentDecision | Every manifest field and decision status enumerated in experiments.md; active/charged times, paired seeds/results, stop reason and evidence-backed trigger disposition. No final-test score or hidden-derived feature enters selection. |

Implement these validators and positive/negative contract fixtures as part of US5–US8; the existing JSON schema covers the three baseline boundary records only and is not claimed to validate these new records. Kind-specific certificates are replayed by a separate process without model weights. A formula identity does not prove its imported definition is true of an OEIS sequence.

Training-only prefixes and continuations are distinct source views with their own immutable split; development/final truth is never mounted into search, replay or admission. For observed OEIS training prefixes, candidates remain hypotheses until a separate training-only continuation check succeeds. Passing a prefix alone cannot be relabeled verified independent generalization. The generic-only SFT control never consumes these extensions silently.
