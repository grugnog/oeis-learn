# Phase 1 dependency qualification (T003)

Status: locks resolved in a pinned Rust 1.90.0 container; native/guest smoke qualification is tracked in [phase-1.md](phase-1.md). Resolution is not successful compilation or arithmetic qualification.

| Closure | Exact dependency | Source lock |
| --- | --- | --- |
| Standalone guest | dashu-int 0.6.2, default features disabled | `crates/oeis_bigint_runtime/Cargo.lock` |
| Native core and binding | dashu-int 0.6.2; z3 0.21.1; z3-sys 0.13.1; z3-src 501.0.1 (Z3 5.1.0 distribution) | Each crate's committed `Cargo.lock`, including registry source/checksum |
| Retained binding | Wasmtime 20.0.2; PyO3 0.20.3 | `crates/oeis_wasm_evaluator/Cargo.lock` |

The guest has a separate dependency closure without Z3, PyO3, Wasmtime, WASI or JS dependencies. Native dependency direction is binding → core, with an execution-backend trait owned by core. No core → binding dependency exists. Production bigint execution and every synthesis operation remain disabled.

Resolution evidence: [run 39152193338](https://github.com/grugnog/oeis-learn/actions/runs/39152193338), with Rust 1.90.0 installed in root Dockerfile's explicit `dependency-locks` maintenance target. [Run 39152833400](https://github.com/grugnog/oeis-learn/actions/runs/39152833400) resolved the final SHA-256 transport dependency and formatted sources using Rust 1.90.0. Production targets require `--locked`; they never inherit the lock-resolution stage. The maintenance workflow is manual only and never commits automatically.

## Compiler correction discovered by compilation

[Build 39152838621](https://github.com/grugnog/oeis-learn/actions/runs/39152838621) compiled the standalone guest but failed native Z3 at `src/ast/array_decl_plugin.cpp`: `fatal error: format: No such file or directory`. The existing Debian Bookworm GCC 12 standard library does not provide the C++ `<format>` surface required by vendored Z3 5.1.0.

Keep the researched Rust/Z3 pins and existing Python/base/Wasmtime versions. Use container-installed Clang 19 with libc++19 (`CXX=clang++-19`, `CXXFLAGS=-stdlib=libc++`, `CXXSTDLIB=c++`). Copy the shared builder's actual libc++/libc++abi/libunwind runtime bytes to both final targets; record hashes in `bigint-build.json`. Install the matching compiler major in both targets for reproducible subsequent native tests. This is a build-only compiler qualification adjustment; no host dependency or fallback solver is introduced. The binding wheel explicitly uses `--compatibility linux`: it is qualified in these images with their packaged C++ runtime, not advertised as a standalone manylinux wheel. Native build/cache flags are declared in the Dockerfile and scripts; actual image, compiler, lock, binary and runtime hashes belong to the generated report.

Upstream sources: [z3-sys 0.13.1 manifest](https://github.com/prove-rs/z3.rs/blob/z3-sys-v0.13.1/z3-sys/Cargo.toml), [build/linking policy](https://github.com/prove-rs/z3.rs/blob/z3-sys-v0.13.1/z3-sys/build.rs). Its vendored dependency is z3-src 501, and `CXXSTDLIB` explicitly selects the C++ library. No source patch, system Z3 or runtime download is used.

## Build and inspection procedure

`docker/dev/build-synthesis.sh` builds the guest for `wasm32-unknown-unknown` (release, panic abort, LTO, one codegen unit, size optimization), executes a fixed >i64 native model numeral smoke and runs the native shape/rejection tests. The dependency-only layer is keyed by manifests/locks, so source edits do not rebuild Z3. Both final targets use the same resulting artifacts/cache.

`docker/dev/build-native.sh` builds the retained binding wheel with `--locked`, then `scripts/inspect_bigint_build.py` instantiates the actual WASM with metered fuel, requires zero imports, inspects exports/declared target features, checks disabled production status, and round-trips the fixed exact native numeral through PyO3. It writes `/opt/oeis-build-info/bigint-build.json` with artifact bytes/hashes, locks and native identity. These are dependency smoke tests, not runtime/solver arithmetic qualification.

The shared-image contract partition repeats inspection under UID/GID 12345 with a read-only image and no network. Native tests run in a separate partition. Actual HIP tests remain separate and are not satisfied by building a ROCm image.
