# Phase 3 code review and remediation

Reviewed against spec 007 US1, execution/CLI contracts and T009–T019, from main
`698de386fd322691ffb54a4ef1fa20b32a6cc3fe`. Scope is this phase only. The existing
root Dockerfile and shared-filesystem `lab-gpu` approach are retained.

All eleven phase-3 tasks are implemented and CPU-validated. The execution service
extends the existing `oeis_learn.sandbox` package and repairs its lowering,
preamble and compatibility runners. It is not a second application or training
stack. `FoundationRunner` supplies the strict body-only entry point; the old
full-module interface remains explicitly diagnostic.

## Findings and fixes

| Severity | Finding on the baseline | Remediation and evidence |
| --- | --- | --- |
| Critical | `1 * -1` produced an unsigned-looking result; wide constants were emitted as invalid i64 literals; add/sub/multiply silently wrapped logical overflow. | Seven new regressions failed before repair. Emit four signed limb literals; detect signed add/sub overflow; calculate a 320-bit scalar product with both sign corrections and high-limb range check. Native carries still wrap normally. Checked overflow sets a private marker immediately before trapping. Hand vectors and 10,000 independently calculated random vectors pass. |
| Critical | No frozen AST/reference/shared execution implementation existed. | Added body-only AST validation using the complete codec; independent Python-integer interpreter; final-source parse/lower/compile and separate execute/verify stages. Tests cover native opcode semantics, scope/types, result-bearing branches, loops, locals, traps, resource limits and fresh state. Reference arithmetic never calls production helpers, lowering or Wasmtime. Parser independence is not claimed. |
| Critical | No externally enforced candidate deadline or bounded execution service existed. | Persistent spawned workers with fresh Store/Instance per term, fuel and reference counters at both term/candidate scopes, external two-second deadline covering compile and both engines, kill/reap within another two seconds, replacement, 1 GiB address-space cap, one native execution thread, at most eight workers, 32 outstanding requests and 1 MiB messages. Real fault tests block compilation/reference, kill workers and saturate the queue. Docker isolation is explicitly T046, not claimed here. |
| High | Legacy fallback coerced multi-value results with `int(list)`; malformed limbs silently became zero; batch fallback ran unlowered source. | Shared strict four-limb decoder; repaired legacy single/batch lowering and actual fuel accounting; fresh legacy instances per term. Strict `FoundationRunner` requires an explicit Python/Wasmtime backend and shares the worker service for both paths. Rust/auto are rejected, and the legacy runner cannot accept the strict profile. |
| High | There was no trustworthy final-source/finite-evidence verification boundary. | Recheck proposed source after grounding/transformation; reject C=4 for C*C=4 and undeclared local.tee transformations. Require complete exact outputs, matching independent records/digests, per-term and aggregate resource evidence and runtime/profile identities. Raw execution cannot preclaim a match or PROVEN. Existing model/archive validation rejects proof labels; finite matches retain `proof_status: not_claimed`. |
| High | The previous conformance command only inspected artifact files; regression fixtures themselves were invalid or had wrong expectations. | Implemented `foundation conformance --profile FILE --output DIR --json`, corrected and packaged fixtures, deterministic arithmetic/structured generators and immutable per-attempt records/report. Artifact-only invocation remains explicitly diagnostic. No artifact-only pass is promoted into G1. |
| Medium | Preamble fuel costs were invented constants; packaged wheels omitted the WAT resource. | Static costs are unavailable rather than measurements. Actual `Store.get_fuel()` consumption and independent AST steps are persisted separately. Package WAT and conformance JSON; verify installed-wheel execution outside the checkout. |
| Medium | A new cache/idempotency layer could reuse evidence after runtime changes. | Bounded LRU stores only self-compiled serialized modules; keys bind canonical source, helper, profile, runtime/engine flags, architecture and implementation hashes. Durable attempt IDs bind source/indices/limits/runtime and reject conflicting reuse. Hashes cover the parser, reference, lowering, controller and profiles. |

## Validation

Final affected suite: **393 passed, 1 skipped** in 25.97 seconds. The skip is the
existing phase-2 AMD/HIP sampler test, not a phase-3 test. Four existing PyTorch
warnings concern `norm_first`/nested tensors. No phase-3 test needs a GPU.

```bash
OMP_NUM_THREADS=2 python -m pytest -q -ra \
  tests/unit/test_foundation_*.py \
  tests/contract/test_foundation_*.py \
  tests/integration/test_foundation_*.py \
  tests/unit/test_multi_limb_arithmetic.py \
  tests/unit/test_wasm_sandbox.py \
  tests/contract/test_macro_instruction_lowering.py \
  tests/contract/test_preamble_contract.py \
  tests/contract/test_cli_contract.py
```

Full G1 command:

```bash
python -m oeis_learn.cli.main foundation conformance \
  --profile configs/foundation/wat_profile.yaml \
  --output /workspace/phase3-conformance --json
```

Use a new, empty output directory. Conformance runs CPU workers, requires local
Unix-socket IPC/process control, and does not require `lab-gpu`. It records all
attempts under `evidence/`, including complete reference outputs/steps/failures,
actual fuel, hashes and resource failures. The result directory is owned by one
controller; restart/re-delivery of an attempt is idempotent, but a new generation
attempt must receive a new identity. Synchronous `Runtime` primitives are for
worker internals and diagnostics; admission/evaluation callers must use the
supervised runner, never bypass it to claim process containment.

The final locked-runtime run completed **10,268 cases / 35,612 requested terms**
in **42.42 seconds**: 12 known regressions, 10,000 seed-7001 arithmetic vectors,
and 256 seed-7002 structured programs, each evaluated at indices 0–99. There were
**zero unexplained disagreements and zero interruptions**. Overflow rejections
are expected cases and must match exact-integer arithmetic. The structured
corpus has independently calculated expected values; it uses task-independent
bounded instruction/control combinations, not OEIS sequence-family templates.

The [machine report](phase-3-conformance.json) preserves original review-machine
output paths and exact identities. Those paths are not repository dependencies.
[Three complete evidence samples](phase-3-evidence-samples.json) cover negative
multiplication, a genuine overflow and a 100-term structured program. Re-running
the command generates the complete per-attempt evidence corpus locally.

Environment: Python 3.12.14, Wasmtime **48.0.0** (project lock), PyTorch 2.9.1+cpu,
pytest 9.1.1, jsonschema 4.26.0. The first exploratory G1 run also passed on
Wasmtime 49.0.0; the attached final report and final suite use locked 48.0.0.
Only an isolated venv was changed; no new host/system dependency was installed.

Additional checks passed:

- Spec Kit prerequisites with `SPECIFY_FEATURE_DIRECTORY=specs/007-experiment-foundation`, `--require-tasks --include-tasks`; both requirements checklists are fully checked.
- `validate_artifacts.py`: 108 tasks, 52 requirements, 68 review/decision inputs, dependency graph, local links and schema examples. This remains document validation, separate from G1.
- `uv lock --offline --check` using uv 0.8.22; no dependency/lock change.
- Wheel build/install and supervised exact execution outside the source tree; WAT and regression JSON are present.
- Ruff `E4,E7,E9,F` on changed sandbox/test modules and `cli/foundation.py`, plus `git diff --check`.

## Phase handoff and limits

T009–T019 are complete. Phase-2 T005/T007 remain partial for their outstanding
AMD/HIP case; later phases are unchanged. Rust parity, Docker device/network/
mount/seccomp containment, learner resource orchestration, admission-pool building
and training remain their own later-phase gates. G1 establishes finite execution
conformance, not GPU readiness, formal correctness, OEIS coverage or infinite
sequence proofs. No solver, regex optimizer or legacy proof hook is enabled in
the strict path.
