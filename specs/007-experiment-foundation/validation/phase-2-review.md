# Spec 007 phase 2: code review and remediation

Reviewed 2026-10-02 against main `4ead096ad9b3b802fdb0c52063ffb5338607259d`.
Scope: T003–T008 and their already-listed convergence fixes T106–T108.
Later phases were not implemented or marked complete. The root Dockerfile and
shared-filesystem `lab-gpu` workflow are accepted as the phase-1 approach.

This improves the existing encoder, decoder, grammar masker, sampler and CLI.
`experiments/` holds the strict artifact/configuration boundary; it is not a
second training codebase. The new codec is explicitly selected on the existing
`WatProgramSampler`; legacy full-module sampling remains available for diagnostics.

## Findings and fixes

| Priority | Finding | Remediation and evidence | Tasks |
| --- | --- | --- | --- |
| Critical | `validate_body` rejected its own EOS; encode/decode admitted ill-typed bodies and non-WAT integer terminators. Tests enshrined invalid examples. | Canonical WAT source uses ordinary integers. Token streams use `<int>`, optional sign, digits and `</int>`. Exact signed bounds, source/token limits, one final EOS, no BOS/PAD in bodies, no UNK fallback. Replaced invalid positive fixtures with valid four-limb bodies and rejection cases. | T006–T007 |
| Critical | The purported incremental mask did not check operand types, locals, control nesting, branch scope or the return signature. | Incremental typed control stack, block/loop/if result signatures, polymorphic unreachable handling, all 41 declared locals, numeric branch depths, nesting caps and partial integer validation. Deterministically sampled native-only bodies are independently compiled by Wasmtime. This establishes native static-type agreement, not wide arithmetic correctness. | T006–T007 |
| High | The codec was disconnected from the actual sampler. Legacy finalization silently completed truncated programs; masks could discard high IDs. | Explicit codec selection in existing grammar/sampler modules; full-vocabulary masks reject a mismatched width; seeded single-candidate generation requires masking and rejects caps/incomplete bodies. Body IDs exclude BOS/PAD and retain EOS. Decoder state carries the exact codec digest and rejects legacy/tampered identity even with non-strict loading. | T007 |
| Critical | Storage called semantic checks without JSON shape validation. Missing fields could escape or raise `KeyError`; integral floats and bad UUIDs were accepted. | The public validator now runs the packaged schema with strict integer types and format checking before semantic checks. Additional range, stage/horizon, failure-index, metric-unit and expected-profile checks. Missing/unknown fields are exercised at every object level. Duplicate JSON members/non-JSON constants are rejected on ingestion. | T003–T004, T008 |
| High | Configuration accepted missing fields, duplicate YAML keys, unresolved vocabulary size zero and invalid constructor dimensions. Effective dropout remained `10`, not `0.1`; the encoder ignored the declared modulus list. | Required-key validation, explicit smoke settings, content-digest profile locks, dimensions/context/prime/modulus checks, exact constructor kwargs, effective numeric conversion, defensive copies and durable immutable effective-config publication. S2 now receives explicit base moduli; omitted base moduli preserve legacy behavior. Real CPU encoder/decoder forward/backward verifies wiring. | T005 |
| High | The frozen profile omitted most inventory details and did not bind configuration to content hashes. | Complete serializable opcode/local/signature/numeric/codec/resource/evaluation inventory, immutable exposed constants, explicit special-token IDs, computed profile digests and exact profile-file validation. Runtime/helper identities still belong to later preparation/run contracts. | T004–T005 |
| Critical | Registry could construct unsafe paths/kinds, rewrote checkpoint metadata into `.pt` files, omitted fsync, and relied on an in-memory lookup index. | Exact final UTF-8 bytes are hashed and atomically published without replacing existing content; flush/fsync and failure cleanup. Checkpoint manifests remain JSON and verify existing blobs. Path containment is rechecked, reads verify hashes, lookup survives restart, and reference graphs verify existence/hash/acyclicity. Run storage requires matching expected identities and complete references. Diagnostic fixtures are explicitly isolated and cannot enter run storage. | T008 |
| High | Three CLI implementations disagreed; conformance could pass an empty input or only check shape. “Freeze” and “build pool” merely wrote lists, bypassing required later pipelines. Preflight delegation used a nonexistent signature. | One argparse dispatch boundary, diagnostic-only conformance using the shared validator, nonzero failures for malformed/empty input and unsupported future commands. Preflight forwards its real config/output arguments and exit status. Heavy training dependencies load only for their owning commands. | T008, T108 |
| Medium | JSON Schema was only a development dependency and its contract depended on a repository-relative path. | Added the existing pinned jsonschema package to runtime dependencies; package the schema and test equality with the normative contract. Built/installed wheel validates from outside the checkout. No package upgrades or Docker layout changes. | T004, T008 |
| Medium | Documentation validator assumed all tasks were unchecked and omitted existing T106–T108; adoption RFC link was broken. `data/` ignore pattern also hid source-package additions. | Accept explicit implementation statuses while retaining unique task/coverage/dependency checks; map existing convergence tasks; fix relative RFC link; anchor the data ignore pattern at repository root. | Phase-2 review support |

## Validation performed

Initial baseline: the old 102 phase-2 tests passed, while seven new public-boundary
regressions failed. This confirmed that passing the original tests was insufficient.

Final focused/affected suite: **262 passed, 1 skipped** in 22.05 seconds.
The skip is the AMD/HIP device case. Six PyTorch warnings concern the existing
`norm_first`/nested-tensor configuration; no test failures remain.

Review environment: Python 3.12.14, PyTorch 2.9.1+cpu, Wasmtime 48.0.0,
jsonschema 4.26.0, pytest 9.1.1. Dependencies were installed in an isolated
scratch venv; no host system packages or repository Docker setup were changed.
This was not a Docker build or execution of a frozen training run.

CPU command (inside the project development environment):

```bash
OMP_NUM_THREADS=2 python -m pytest \
  tests/unit/test_foundation_codec.py \
  tests/unit/test_foundation_config.py \
  tests/unit/test_foundation_sampler.py \
  tests/unit/test_foundation_storage.py \
  tests/unit/test_foundation_phase2_regressions.py \
  tests/contract/test_foundation_artifacts.py \
  tests/unit/test_sampler_determinism.py \
  tests/unit/test_grammar_masker.py \
  tests/unit/test_environment_tracker.py \
  tests/unit/test_tri_stream_encoder.py \
  tests/contract/test_wat_grammar_contract.py \
  tests/contract/test_cli_contract.py -q -ra
```

Also passed:

- Spec Kit `check-prerequisites.sh --json --require-tasks --include-tasks`, with `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation`.
- Project `validate_artifacts.py`: all 108 tasks, 52 requirements, 68 review/decision items, coverage/dependency graph, local links and positive/negative schema fixtures. Document checks do not certify runtime gates.
- `uv lock --offline --check` using project-pinned uv 0.8.22. The lock change adds only jsonschema runtime edges; existing versions and platform markers are preserved.
- Wheel build and installation; schema validation outside the source checkout.
- Ruff `E4,E7,E9,F` on changed Python files, and `git diff --check`. Broader legacy style modernization was not part of this review.

## Outstanding device check and phase handoff

T003/T004/T006/T008 are reviewed complete for this phase. T005/T007 are marked
`[/]`: their CPU implementation is complete, but the AMD/HIP integration case
has not run. No `lab-gpu`, Docker or GPU device was available in this review
workspace. On the Ryzen machine, run from the shared project checkout:

```bash
lab-gpu image ensure
lab-gpu exec -- python -m pytest tests/unit/test_foundation_sampler.py -q -ra
```

The `cuda` parameter case must execute, not skip. It checks the effective
constructor wiring, HIP tensor residency and finite forward/backward on a small
FP32 model. It does **not** replace the full-context, three-update GPU smoke,
resource containment, recovery or throughput gates in phase 4. Those tasks
remain unchecked. Successful device output can close the phase-2 partial marks;
retain the command output and actual GPU/runtime identity with that evidence.

The body codec identity/vocabulary has changed relative to the broken initial
implementation. Existing output weights cannot be reused. `wat_profile.yaml`
and `wat_smoke.yaml` are updated together. This is compatible with the explicit
permission to restart training, and checkpoint loads fail on incompatible identity.

Next review is phase 3: checked wide arithmetic, parsing/lowering, independent
reference execution and sandbox limits. Native compilation in this PR is not
G1 execution conformance, a proof claim or an OEIS success result.
