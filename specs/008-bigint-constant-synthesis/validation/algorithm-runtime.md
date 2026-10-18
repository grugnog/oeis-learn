# Phase 2 US1 checkpoint (T011–T025)

Status: **implementation milestone pushed; T025 native qualification pending**.
This is not the complete Phase 2 exit and does not open fitting or Phase 3.
Production flags and the framed public service remain disabled.

The local suite passed **202 tests** with **29 explicit skips**: 25 native
compiler/runtime tests, two installed native/artifact checks, and two actual-HIP
checks. Another 16 disposition/traceability tests passed. Python F/E9 lint,
workflow YAML parsing and build-script shell syntax passed. The local environment
was Python 3.12.14 and Torch 2.9.1+cpu; Python Wasmtime 20.0.0 is used only for
retained local tests. The production native engine stays pinned to 20.0.2.
See [local evidence and source hashes](algorithm-runtime-local.json).

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

T026–T058 are intentionally untouched until T025's required native evidence is
available. The spec explicitly orders these after the runtime checkpoint; test
presence or CPU-only reference agreement cannot close that gate. Per the user,
CI is left running without monitoring. On continuation, inspect the completed
checks, repair any failures, record exact artifact identities, then close T025
and proceed to native fitting.
