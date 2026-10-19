# Phase 2 US1 checkpoint (T011–T025)

Status: **historical US1 checkpoint; current review is [phase-2.md](phase-2.md)**.
The following original checkpoint and CI repairs are retained as historical evidence.
The merged PR #54 native checks passed; PR #55 now implements the common bounded
core driver and routes every runtime differential case through it. T025 remains
open only for execution of that changed bridge; Phase 3 is still closed.
Production flags and the framed public service remain disabled.

See [the native build repair and cache boundaries](build-cache.md) for the later
formatting failure, early formatting gate, independent dependency-cache export
and source-scoped native-wheel stages.

The local suite passed **202 tests** with **29 explicit skips**: 25 native
compiler/runtime tests, two installed native/artifact checks, and two actual-HIP
checks. Another 16 disposition/traceability tests passed. Python F/E9 lint,
workflow YAML parsing and build-script shell syntax passed. The local environment
was Python 3.12.14 and Torch 2.9.1+cpu; Python Wasmtime 20.0.0 is used only for
retained local tests. The production native engine stays pinned to 20.0.2.
See [local evidence and source hashes](algorithm-runtime-local.json).
That evidence records the original US1 implementation at `29f7fa8`; the CI
follow-up below records subsequent changes separately.

## CI regression fixes after the US1 push

Completed review run [39165212003](https://github.com/grugnog/oeis-learn/actions/runs/39165212003)
reported two failed jobs: `cpu-regressions` omitted `input_profile` from the
explicit encoder constructor record, and `cpu-acceptance (training-integration)`
could not deep-copy a decoder containing a Python codec module. All three failing
test cases were reproduced locally before the fixes.

Foundation constructor records now explicitly select the historical
`input_profile=None`. Decoders store a codec module name and resolve the module
through a property, so frozen training references can copy the model. A regression
test covers both explicit codecs, matching outputs and independent copied weights.
The rolling 007 source identity is rebound for traceability only.

Local verification: **88 passed, one HIP-only skip**, including the complete
training-integration job, foundation sampler/configuration/cache/handoff tests,
and bigint decoding tests. Python F/E9 lint and `git diff --check` passed.
The environment remains Python 3.12.14 / Torch 2.9.1+cpu; the declared Python
`z3-solver` dependency was installed (5.3.1.0) for the retained training tests.
No native or HIP qualification is claimed. The still-running container build was
not monitored, and T025 remains open.

```sh
OMP_NUM_THREADS=2 PYTHONPATH=src .venv-phase2/bin/python -m pytest -q \
  tests/unit/test_foundation_sampler.py tests/unit/test_foundation_config.py \
  tests/integration/test_egca_training_step.py tests/integration/test_bootstrap_admission.py \
  tests/integration/test_bigint_decoding.py tests/unit/test_foundation_cache.py \
  tests/unit/test_foundation_handoff.py --junitxml=tmp-validation/phase2-ci-fixes.xml
```

## Original US1 implementation evidence

The independent tests were written first: the reference/value suites initially
failed collection because the semantic modules did not exist. Encoder tests
likewise failed on missing implementations. Native tests cannot execute in this
workspace: neither Docker/Podman nor the pinned Rust toolchain is installed.
They are required, rather than skipped, in the unprivileged shared-image CI
partition through `OEIS_REQUIRE_BIGINT_BUILD=1`.

| Task group | Implemented evidence | Remaining acceptance |
| --- | --- | --- |
| T011–T015 language/values | Canonical decimal and minimal signed magnitude; immutable resource/profile records; Rust typed parser, canonical parameter renaming and AST identity; source/node/depth/type rejection tests | T014/T015 remain unchecked until native parser/profile tests pass |
| T012/T016/T017 runtime | Immutable guest arena, sticky statuses, full multiplication, bounded host transfer, typed lowering, explicit seven-function linking, fresh stores, fuel/epoch/memory limits, differential/containment tests | T016/T017 and T025 remain unchecked until real guest/core execution passes |
| T013/T018 reference | Separately parsed source and Python-int execution; factorial0…99, signed offsets, limit-before-cancellation, eager select/lazy branches, aggregate steps and deadlines; injected wrong outputs never agree | Public admission and evidence identities belong to T027–T035; this arithmetic comparison grants no admission |
| T019 retained adapter | Named bare macro-body const/zero/add/sub adapter using the original checked preamble/native evaluator; explicit unsupported scalar/wrapping/bitwise/control/streaming forms | Native min/max/overflow regression pending; T019 remains unchecked |
| T020 codec | Distinct BOS/PAD/EOS and literal delimiters, first-occurrence parameters, typed constrained generation, token cap, no repairs, incompatible old checkpoint rejection | Native typed-parser cross-check is part of T025; no production routing |
| T021–T024 inputs/decoder | Finite exact-residue/difference/log features; shared eight-byte/d64/four-head/ff256 chunk layer; masked pooling, growing positions, d256 byte-gate fusion; CPU teacher-forcing/cache/full parity and old-profile regressions | Actual HIP update/cache tests pending; real conditioning-policy parity remains explicitly at T052 |

The exact frozen fixture manifest is unchanged. Native tests cover its arithmetic
cases directly, including all factorial indices, large cancellation, two-large
multiplication, borderline and zero products, and actual signed source indices.
Resource tests cover source/literal/type limits, fuel, aggregate allowance, arena,
memory limiter denial, control/loop bounds, deadline and unknown-trap classification.
Native guest tests reject invalid handles and reset attempts. A failure produces
no numeric values; successful coverage before failure remains metadata.

The core locks reuse the existing exact Wasmtime/wat/wasmparser dependency tuple.
`dashu-base` 0.6.1, already in all Dashu locks, is now a direct dependency to bring
its `BitTest` trait into scope. No package version or solver tuple was upgraded.
The original preamble is now explicitly in the binding's recursive source identity.
The rolling 007 disposition record is rebound to current Python source bytes for
traceability; historical evidence is not promoted to this implementation.

## Native gate to review after CI

The PR workflow builds the real guest, tests the guest/core/retained adapter, and
packages `/opt/oeis-build-info/runtime-probe` beside its hashed WASM artifact.
The contract partition runs the reference, language, input, decoder and real
native differential suites without network, as UID/GID 12345, with read-only
storage. It requires packaged native artifacts rather than silently skipping.

On a machine with the qualified image, the equivalent direct checks are:

```sh
docker build --target qualification -t oeis-008-phase2 .
docker run --rm --network none --read-only --tmpfs /tmp --user 12345:12345 \
  -e HOME=/tmp -e OEIS_REQUIRE_BIGINT_BUILD=1 \
  -e PYTHONPATH=/opt/oeis-build/src:/opt/oeis-build oeis-008-phase2 \
  python -m pytest -q -p no:cacheprovider \
    tests/contract/test_bigint_build.py tests/contract/test_bigint_contracts.py \
    tests/contract/test_bigint_language.py tests/unit/test_bigint_reference.py \
    tests/integration/test_bigint_runtime.py tests/unit/test_bigint_encoder.py \
    tests/integration/test_bigint_decoding.py
```

Use the equivalent Podman command if preferred. Native parser/compiler process
RSS and kill/reap supervision remain obligations of the Phase 3 worker, which is
not yet implemented; the diagnostic probe is not a production worker. Engine
compilation time is charged conservatively to each evaluation. The runtime API
requires mutable access so request-local epoch interruption cannot race another
request on the same engine.

Actual HIP checks are `tests/integration/test_bigint_gpu.py`, to run in the GPU
image with the user's usual explicit device mounts. No GPU run or result is
claimed here. The future `validate --suite bigint-gpu` CLI is still a Phase 3 task.

At the time of this original checkpoint, T026–T058 were untouched pending T025.
This historical statement is superseded by the PR #55 completion review; The spec explicitly orders these after the runtime checkpoint; test
presence or CPU-only reference agreement cannot close that gate. The continuation implements all unblocked algorithms at the user's request;
current native CI is left unmonitored and its execution gates remain open.
