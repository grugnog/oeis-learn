# Phase 2 native build repair and cache boundaries

The [failed run 39166274739](https://github.com/grugnog/oeis-learn/actions/runs/39166274739/job/117789821868)
completed the dependency compilation in 17m20s and the guest/core build-and-test
script in 21s. The final native packaging step then failed on
`cargo fmt --check` for `synthesis_adapter.rs`, before compiling the binding.
The adapter now incorporates the exact formatting changes emitted by that
pinned formatter. No native arithmetic behavior was changed by the repair.

The previous build exported its cache only after the entire image succeeded.
A late formatting/test error therefore discarded a successful expensive dependency
build on the ephemeral runner. Earlier incremental pushes also cancelled that
work before completion. This change establishes these boundaries:

| Stage | Inputs that can invalidate expensive work |
| --- | --- |
| `native-format-check` | Pinned toolchain, formatter configuration and Rust checkout; runs before compilation in CI |
| `synthesis-dependencies` | Pinned toolchain and core Cargo manifest/lock; existing expensive RUN instruction retained |
| `native-dependencies` | Above plus binding Cargo manifest/lock; warms the binding's feature-unified graph without production source/build.rs |
| `qualification-dependencies` / `gpu-dependencies` | Stable toolchain, install script, Python project metadata and uv lock; independent of Rust source builds |
| `qualification-native` / `gpu-native` | Dependency stages and exact native source/build identity closure, including the checked preamble/schema; wheel compiled before full checkout COPY |
| Final `qualification` / `gpu` | Current checkout, guest/core test artifacts, Python packaging and native identity inspection |

The dependency seed packages are explicitly cleaned after compilation so their
empty cache-seed libraries cannot stand in for actual production sources. All
production build commands remain locked. Both image variants use two Cargo/CMake
build jobs. The final package installs the prebuilt native wheel and validates its
embedded identity against the current source closure; it cannot silently accept
a stale wheel. The wheel script and formatter configuration are included in that
closure and in source distributions.

CI exports `native-dependencies` in a separate cache-only build before source
compilation/testing, using a distinct GHA scope. Final-image cache exports cannot
replace it. Main-branch runs may also refresh the dedicated registry dependency
cache; PRs retain the existing restriction against updating shared registry
caches. Fork/Dependabot builds retain archive transport and read-only registry
access. A cancelled dependency build still cannot save unfinished work, but a
subsequent source/test failure no longer discards a completed dependency cache.

Local validation passed **95 tests with two installed-artifact/native skips**,
plus **16 disposition/traceability tests**. Cache-boundary tests follow the actual
Docker stage dependency graph and check Python/docs/fixture, Rust source,
manifest/lock, preamble and uv-lock changes for both image targets. Source identity
tests detect a modified wheel recipe. Shell syntax, Python F/E9 lint, workflow
YAML and whitespace checks passed. With the pinned uv 0.8.22, an offline locked
export succeeded from a directory containing only `pyproject.toml` and `uv.lock`;
README/setup/source files are unnecessary inputs to the dependency installer.
An actual source distribution was built and extracted; its native source/build
closure matches the checkout byte-for-byte, including the new wheel script and
formatter configuration.

There is no local container engine or Rust toolchain. The formatted adapter was
reconstructed from CI's exact rustfmt diff; the new early CI check must confirm it.
The new Docker builds and cache-hit timings remain unexecuted locally. This
repair does not close T025 or grant native/HIP qualification. CI is left running
after the push, per the user's instruction.
