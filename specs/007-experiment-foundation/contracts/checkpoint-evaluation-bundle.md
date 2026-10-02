# Phase 4 checkpoint and finalization interfaces

This is the concrete reader interface for T023–T029, and the producer handoff
for T041/T042. It does not implement the later training/checkpoint writer,
pool admission, container isolation, or run qualification gates.

## Checkpoint bundle

The CLI takes an external `checkpoint_manifest`, not a legacy weights file.
Its parent directory contains `config.yaml`, `effective-config.json`,
`contract.json`, `pool.json`, `runtime.json`, and the relative `blob_path`.
Paths use the phase-2 traversal/symlink checks. Sidecar identities use canonical
JSON; the model blob identity uses the final file bytes. Hashing and restricted
`torch.load(weights_only=True)` use the same open file descriptor, followed by a
second hash. Legacy v2 loading/conversion remains a separate API.

`contract.json` has exactly these fields:

- `schema_version: foundation/v1`, `kind: experiment_contract`, UUID `run_id`,
  `purpose: diagnostic | training`, `contract_id` (canonical digest excluding
  only itself).
- `profiles`: the exact adopted language, codec, resource and evaluation
  digests. The language inventory includes the numerical and entrypoint
  profiles; evaluation includes the 20/100 horizon. These bundled references
  encode the conceptual profile fields in `data-model.md`.
- `constitution_version: 2.0.0`, `track: strict_generic`,
  `initialization: random`, `objective: sft`, `enabled_subsystems: [sft]`.
- `observed_terms: 20`, `total_terms: 100`, `index_policy: prefix_rebased_zero`.
- `run_provenance`: exactly `source_revision`, `dependency_lock`,
  `base_image_digest`, `final_image_id`, `environment_report`,
  `dataset_manifest`, `benchmark_manifest`, `evaluation_protocol`,
  `effective_config`. Source revision is a full Git SHA; other populated pins
  are SHA-256 identities. Pool/config pins must match the manifest. Evaluation
  checks populated cohort/protocol pins against the chosen inputs. A training
  bundle requires every pin; diagnostic fixtures explicitly use null for
  unavailable production provenance. Null is never qualifying evidence.

The manifest's `contract_sha256` hashes the complete sidecar, including its
validated logical `contract_id`. No model file contains its own byte digest.
The future producer must also resolve and validate the referenced provenance
artifacts; hash-shaped strings alone do not establish qualification.

The blob contains exactly the eleven schema-defined state keys. The reader
checks both model state dictionaries, constructor/dtype/shape/finite-value
compatibility, exact loaded tensors, AdamW state, explicit null inactive
scheduler/scaler, Python/NumPy/CPU/GPU/named RNG state, permutation/cursor/next
examples, a completed update boundary, and manifest/counter agreement.
Evaluation consumes the actual reconstructed encoder and decoder weights.
This reader validates complete continuation material; restoring a training
session remains T041/T044 work.

`tests/helpers/foundation_fixture.py` constructs a small real encoder/Transformer
and real optimizer state. Deliberately wired token transitions make its outputs
depend on checkpoint parameters. This is a diagnostic fixture, not evidence of
learning or a substitute for a generic admitted training pool.

## Evaluation and finalization

`freeze-cohort` accepts a directory containing `records.json`. Each record has
exactly `record_id` (source-qualified), signed `first_index`, `indices`, canonical
integer-text `values`, and `metadata`. Source order does not choose the split.
Its YAML config has exactly `schema_version`, `profile`, `seed`, `dev_count`,
`final_count`; see the corrected fixtures for a runnable example.

`evaluate` uses explicit `--device cpu|cuda`; the PyTorch `cuda` device API also
addresses HIP. It never falls back from unavailable GPU to CPU. Missing/corrupt
inputs are not zero-accuracy runs. Candidate failures are scored against the
fixed denominator; infrastructure failures leave an incomplete run without a
new completed report. The model process receives a checkpoint path and only
the narrow VisiblePrompt plus public sampling controls. It is spawned, not
forked from the evaluator's private truth objects. Actual filesystem/device/
network isolation belongs to T046 and must not be inferred from this boundary.

Phase budgets persist monotonic time plus boot identity and UTC recovery time;
crash gaps are charged. Model loading precedes the prefix budget. Generation,
prefix execution/verification and sealing consume the prefix phase; hidden
truth loading and full verification consume the full phase. Execution intent
records preserve worker request identity and original limits across retries;
an independent enclosing deadline can only shorten execution.

`finalize` reads a paused/completed `run_dir/ledger.json` snapshot containing
exactly `run_id`, nonnegative integer `sequence`, `charged_budget_ns`, and
`status: paused|completed`. Its high-water values cannot precede the checkpoint.
T042 should export this snapshot from the durable ledger; this is not its
heartbeat/event-log storage format.

The stopping record has exactly `decision: stop`, a nonempty `reason`,
`checkpoint_sha256`, `protocol_sha256`, and `cohort_id`. The lock binds all
inputs, runtime/evaluator implementation, selector, stopping record and ledger.
Its canonical anchor is `run_dir/finalization.json`; the requested output file
is an immutable copy. `finalize` returns the required `evaluation_output` path.
Final evaluation accepts the lock and forbids checkpoint/cohort/protocol
overrides. The final report lives at that path's `reports/evaluation.json`.

Per-target seals freeze ordered attempts, actual sources/tokens, prefix
evidence and deterministic selection before hidden decoding. Duplicate programs
consume attempts. Full verification starts fresh at indices 0–99 and requires
production/reference agreement. A separate exposure anchor outside the final
evaluation directory prevents deleting the local seal from enabling fresh
generation after feedback. These artifacts protect supported controller
operations and crash recovery; they are not an adversarial signature service
against an operator rewriting every run artifact.

## Qualification and diagnostics

Reports always retain purpose, checkpoint, protocol, cohort, seal and score
identities. Diagnostic fixtures remain unqualified. Training evidence remains
`pending_run_gates` until later hardware/provenance/isolation qualification is
implemented. Neither finite success nor the `G2` software label is a proof or
novelty claim.

The legacy canary command emits `reference-kernel-diagnostic/v1`, validated by
the new adjacent schema. It explicitly reports no evaluated checkpoint, no
model synthesis and no qualification. Execution status/error and the chosen
fuel budget are preserved; an unexecuted scalar-overflow boundary is null.
The historical 006 schema and reports remain historical artifacts.
