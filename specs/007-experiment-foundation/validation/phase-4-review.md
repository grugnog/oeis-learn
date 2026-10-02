# Phase 4 review and remediation — 2026-10-02

Scope: T020–T029 (US2), based on merged phase-3 PR #4, main commit
`27ec560ac1f3066cebd0113b3142737da0709452`. This improves the existing encoder,
decoder, codec, checkpoint loader, evaluation CLI and phase-3 runtime. New
foundation modules implement the strict evaluation protocol around those
components; there is no second model implementation or training stack.

## Findings and fixes

| Priority | Finding | Remediation and evidence |
| --- | --- | --- |
| Critical | The cohort command inspected artifact directories rather than building the indexed first-100 cohort. Some source fixtures were incomplete or did not represent their stated relationships. | Implement indexed exact-integer census, eligibility, duplicate/conflicting-index checks, signed offsets, equal100/equal20/shift1–20 connected components, lexicographic representatives and seeded digest-order splits. Missing requested groups are errors. Replace the fixtures and test boundaries, ambiguous prefixes, false 80-term matches and a 500-record duplicate bucket. |
| Critical | No foundation evaluation path demonstrated dependence on the actual checkpoint parameters. Legacy conversion accepted a different artifact contract. | Add a separate complete-manifest loader with final-blob integrity, sidecar/config/profile/contract checks, exact loaded tensors and all eleven continuation state keys. Reuse the existing TriStreamEncoder and WatTransformerDecoder. Actual zero-producing versus one-producing weights change generated programs and scores; ignored/missing/corrupt weights are rejected. |
| Critical | No strict visible/private evaluator enforced public-only generation inputs and seeds. | A spawned model process receives exactly the narrow prompt and public sampling controls. No sequence ID, offset, family, stage, metadata or continuation is passed. Seeds exclude private protocol identity. Perturbing hidden terms, names, offsets and nonce preserves proposals. The strict sampler has no canonical-program fallback; tests retain duplicates and token-limit failures. |
| Critical | Top-1 selection, finite verification and the denominator could not be audited through immutable evidence. | Persist ordered attempts, prefix CandidateResults and candidate seals. Freeze shortest-body/hash selection before hidden decoding, execute fresh indices 0–99, and require production/reference agreement. Preserve all frozen groups in N, including complete failures. Tests exercise length/hash ties and reject changed sealed choices. Missing truth/checkpoints produce no new completed score report. |
| Critical | Finalization lacked durable decision locks, crash-safe seals and a rule preventing regeneration after feedback. | Bind checkpoint, protocol, cohort, selector, stopping decision, ledger and implementation before final access. Persist external exposure anchors and immutable reports. Inject failures after attempt, seal, exposure and score; resume without post-seal generation. Reject missing seals, new final output locations and changed inputs. |
| High | Retried execution could receive a new remaining deadline under the same request identity, invalidating or repeating completed evidence. | Persist execution intents with original limits. Extend the existing worker pool with a separate inherited monotonic deadline that shortens execution without changing the request fingerprint. Verify cached retry and forced deadline behavior; repeat the complete G1 corpus. |
| High | The legacy canary command claimed a checkpoint was evaluated while running reference kernels. Threshold-only readiness could authorize qualification, and synthesis reported invented memory. | Version new output as `reference-kernel-diagnostic/v1`, with null evaluated checkpoint, explicit purpose, no qualification/proof and execution status/error. Block threshold-only qualification; retain diagnostic/pending-run-gates status for foundation evaluation. Report unavailable memory explicitly. |
| Medium | A reference-canary fuel failure could still claim overflow prevention; some overflow boundaries are outside the tested horizon. | Use null when a boundary was not executed. Test all six kernels with an explicit 100,000 per-call diagnostic fuel allowance, and fuel exhaustion with allowance 1. The default allowance and strict profile limits are unchanged. This does not qualify learned synthesis. |
| Medium | Freeze/evaluate/finalize were missing or incompatible with the contracted CLI. | Implement exact inputs and exit codes, explicit device selection, final lock-only invocation and machine-readable reports. Add CLI tests for freezing, real checkpoint evaluation, missing inputs, no GPU fallback and finalization. |

Grouping stores a spanning forest of exact witnesses rather than every pair.
Shift matching uses hashed 80-term indexing followed by full `100-d` overlap
verification, including an overlap key to avoid quadratic false-positive
buckets. Equal-prefix membership covers every heldout component member, not
only representatives. The evaluator checks all referenced truth/source/
membership file hashes before reporting, but decodes each private continuation
only after its seal. The model process never receives these artifacts.

## Validation

The final CPU suite passed **398 tests**, with **two explicit AMD/HIP skips**
and no failures. The result and diagnostic evaluation examples are in
[phase-4-software-report.json](phase-4-software-report.json). The examples include
real-weight dependence and a final lock/report/seal. Fixture checkpoints are
recreated by `tests/helpers/foundation_fixture.py`; their deterministic token
transitions are deliberately hand wired, so accuracy is **not** a training
result. UUIDs, file paths and timings are observations from the review run.

Final regression command (from the repository root):

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src:. python -m pytest -q \
  tests/unit/test_foundation_*.py tests/contract/test_foundation_*.py \
  tests/integration/test_foundation_*.py tests/unit/test_readiness_policy.py \
  tests/integration/test_canary_qualification.py \
  tests/contract/test_canary_benchmark_contract.py \
  tests/contract/test_readiness_report_contract.py \
  tests/unit/test_checkpoint*.py tests/contract/test_cli_contract.py -ra
```

The shared worker-pool deadline change was also checked with the complete
`foundation conformance --profile configs/foundation/wat_profile.yaml` corpus:
**10,268 cases, 35,612 requested terms, zero disagreements, zero interruptions**.
The [machine G1 report](phase-4-conformance.json) records the changed runtime
identity and a 46.22-second run; it does not replace the historical phase-3
report. Arithmetic uses seed 7001, structured programs seed 7002.

Environment: Python 3.12.14, Wasmtime 48.0.0 (project lock), PyTorch 2.9.1+cpu,
pytest 9.1.1 and jsonschema 4.26.0. Local process/socket IPC required execution
outside the tool's restricted sandbox. No system dependency was installed.
The validation scope is the impacted foundation/legacy integration suite,
not all historical training/analysis tests.

Additional checks: Spec Kit prerequisites with `--require-tasks --include-tasks`,
the project-specific `validate_artifacts.py`, `uv lock --offline --check`, wheel
build and installed-package import, Ruff `E4,E7,E9,F` on the changed strict
modules/tests, and `git diff --check`. Requirements/design checklists remain
unchanged and complete. Document validation is separate from execution gates.

## GPU follow-up and phase handoff

T020–T022 and T025–T029 are complete for the phase's CPU software scope.
T023/T024 are left `[/]` for actual AMD/HIP loading and generation evidence.
The earlier phase-2 GPU skip remains partial too. Neither Docker nor `lab-gpu`
was available in the review environment. On the Ryzen host, run:

```bash
lab-gpu exec -- python -m pytest -q -ra \
  tests/integration/test_foundation_evaluation.py \
  -k gpu_checkpoint_loading_and_prefix_only_generation
```

Require a **pass, not a skip**, real HIP availability, CUDA-device residency
(PyTorch's HIP API), and actual generation including extreme signed-256 input
terms. This test does not replace the later training/resume/resource gates.

The [checkpoint/evaluation bundle interface](../contracts/checkpoint-evaluation-bundle.md)
specifies fixed sidecars, provenance pins and the ledger snapshot consumed by
finalization. T041/T042 own production writers and continuation/ledger recovery;
T046 owns Docker mount/network/device/resource containment; T049 owns run
qualification. The current process boundary has narrow serialized messages,
not a claimed filesystem security sandbox. Production provenance is required
for training-purpose bundles; diagnostic nulls never confer qualification.
No training, LODA comparison, solver promotion or infinite-sequence proof is
claimed. Later phases and the accepted root Dockerfile workflow are unchanged.
