# Phase 1 gate / Phase 2 handoff (T010)

Status: **PASS — T010 complete; Phase 2 entry open**. At the user's request on 2026-10-18, close Phase 1 on the passing CPU/native and portable-engine checks for commit `dbaea14795332225c234f33c7658f8416e1b5e2f`. The separate `podman-build` job was still running at acceptance and was explicitly designated non-blocking. No Phase 2 implementation or training is started by this record.

Evidence: [container/native run 39157377529](https://github.com/grugnog/oeis-learn/actions/runs/39157377529), [passing foundation regression run 39157377456](https://github.com/grugnog/oeis-learn/actions/runs/39157377456), and [immutable image/artifact/dependency identities](phase-1-evidence.json). The shared qualification image built successfully. Native qualification, contract and native bigint partitions, and both Docker/Podman portable-engine suites passed; the unprivileged contract partition passed all 28 tests without skips.

Scope: both final targets are wired to the same pinned artifacts; execution evidence here comes from the CPU qualification image. A separate ROCm image build and actual HIP updates are not claimed. These remain distinct later qualification work. The user accepts this Phase 1 handoff scope; production bigint profiles stay disabled and all Phase 2 algorithm gates still apply.

| Task/gate | Evidence / state |
| --- | --- |
| T001 retirement inventory | `retirement.md`, machine-readable path inventory and reproducible inventory script; old implementations remain for Phase 4 migration |
| T002 boundaries | Separately locked guest/core; binding → core; core-owned execution trait, no back-edge |
| T003 dependency qualification | `dependencies.md` and `phase-1-evidence.json`; locked Rust 1.90.0 standalone guest/native builds and corrected C++ ABI linkage passed |
| T004 packaging/identity | Both final targets consume the shared dependency layer; source/runtime/lock identities verified; CPU image inspection passed as UID/GID 12345 with read-only storage/no network; separate ROCm image execution not claimed |
| T005 shape stubs | Strict Python/Rust request/record shapes; unknown fields/profile rejected; no operation admits; Python and native shape/rejection tests passed |
| T006 fixtures | `tests/fixtures/bigint/manifest.json`: independent arithmetic/index/signed-256/false-fit expectations and hashes |
| T007 actual dependency smoke | Standalone WASM instantiation, zero imports, disabled production status and exact >i64 PyO3 binding result passed; report records exports/features/hashes |
| T008 CI | Shared-image contract/native partitions; lock/source build triggers; GPU tests remain explicit; existing tests retained |
| T009 compatibility | `compatibility-baseline.md`; retained CPU model/codec/arithmetic regressions plus passing native qualification; actual GPU evidence remains separate |
| T010 handoff | Complete on the passing evidence and user-accepted scope above; Phase 2 handoff ready, no automatic Phase 2 start |

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
