# Phase 1 gate / Phase 2 handoff (T010)

Status: **not yet open**. Phase 1 source preparation is implemented; pending native/container execution evidence must pass before T010 or Phase 2 can be declared complete/started. No compiler, fitting, encoder, data or learning-policy algorithm has been implemented by this phase.

| Task/gate | Evidence / state |
| --- | --- |
| T001 retirement inventory | `retirement.md`, machine-readable path inventory and reproducible inventory script; old implementations remain for Phase 4 migration |
| T002 boundaries | Separately locked guest/core; binding → core; core-owned execution trait, no back-edge |
| T003 dependency qualification | `dependencies.md`; exact locks committed; initial GCC12 failure recorded; corrected compiler build pending |
| T004 packaging/identity | Both final targets consume shared dependency layer; recursive source-closure fingerprint, runtime/lock report and arbitrary-UID test implemented; image execution pending |
| T005 shape stubs | Strict Python/Rust request/record shapes; unknown fields/profile rejected; no operation admits; Python tests pass, native test execution pending |
| T006 fixtures | `tests/fixtures/bigint/manifest.json`: independent arithmetic/index/signed-256/false-fit expectations and hashes |
| T007 actual dependency smoke | Guest/native/PyO3 tests and artifact inspection implemented; final corrected native/WASM binding run pending |
| T008 CI | Shared-image contract/native partitions; lock/source build triggers; GPU tests remain explicit; existing tests retained |
| T009 compatibility | `compatibility-baseline.md`; available CPU model/codec/arithmetic tests, with native/image/GPU evidence separately identified |
| T010 handoff | Blocked until pinned smoke builds/import inspection/readability pass; no automatic Phase 2 start |

## Reproducible commands

From a machine with a usable container engine (host Python is standard library only):

```sh
python3 scripts/foundation_run.py --engine docker image ensure
# Direct build-only CPU/ROCm checks, no training or actual GPU execution:
docker build --target qualification -t oeis-008-qualification .
docker build --target gpu -t oeis-008-gpu .
# The equivalent podman build targets use identical Dockerfile/scripts.
```

For each built image, inspect the packaged smoke report and run the required artifact tests:

```sh
docker run --rm --network none --read-only --tmpfs /tmp --user 12345:12345 \
  --workdir /opt/oeis-build -e HOME=/tmp -e OEIS_REQUIRE_BIGINT_BUILD=1 \
  -e PYTHONPATH=/opt/oeis-build/src:/opt/oeis-build oeis-008-qualification \
  python -m pytest -q -p no:cacheprovider \
  tests/contract/test_bigint_build.py tests/contract/test_bigint_contracts.py
```

Replace the image with `oeis-008-gpu` for its separate packaging/readability gate. A GPU image build is not real HIP update evidence. Do not run the new Phase 3 quickstart suites or enable new profiles from this preparation PR.

## Phase 2 owner handoff

Read every spec document again, especially `algorithm-phase.md`. Preserve the pinned toolchain/locks and frozen fixture manifest. The current request/record schemas are explicitly **disabled shape stubs** (`synthesis-record-stub/v1`), not semantic implementations of the final records. Opaque hashes do not validate referenced evidence. Replace/extend these during Phase 2; no stub can grant admission. `synthesis_dependency_smoke()` is a fixed diagnostic without candidate inputs, not a fitting service.

Finish every mandatory runtime/reference/solver/encoder/policy exit gate in that one algorithm phase. Keep original signed-256 profile semantics; changed binary/source identities are real and must be recorded. Do not remove independent certificate/reference checks with the old solver: the inventory identifies the additional loop-certificate Python Z3 dependency. No user hardware, training, assisted manifests or checkpoint weights are assumed available.
