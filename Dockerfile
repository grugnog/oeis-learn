# Public, pinned images only. Default build is the ROCm GPU target.
ARG GPU_BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191

# Explicit maintenance target for resolving proposed dependency locks. Production
# targets below require committed locks; they never generate or update them.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS synthesis-toolchain
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential cmake ninja-build pkg-config git clang-19 libc++-19-dev libc++abi-19-dev && rm -rf /var/lib/apt/lists/*
RUN curl --fail --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/rustup-init.sh && bash /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.90.0
ENV CC=clang-19 CXX=clang++-19 CXXFLAGS=-stdlib=libc++ CXXSTDLIB=c++ \
    LIBRARY_PATH=/opt/oeis-cxx/lib LD_LIBRARY_PATH=/opt/oeis-cxx/lib
# Keep the native smoke and both installed images on the same C++ runtime bytes.
# libc++.so is a linker script: Rust's -nodefaultlibs link also needs the ABI
# and unwind libraries explicitly, not just a symlink to libc++.so.1.
RUN mkdir -p /opt/oeis-cxx/lib && cp -L /usr/lib/llvm-19/lib/libc++.so.1 /usr/lib/llvm-19/lib/libc++abi.so.1 /usr/lib/llvm-19/lib/libunwind.so.1 /opt/oeis-cxx/lib/ && printf '%s\n' 'INPUT ( /opt/oeis-cxx/lib/libc++.so.1 /opt/oeis-cxx/lib/libc++abi.so.1 /opt/oeis-cxx/lib/libunwind.so.1 )' > /opt/oeis-cxx/lib/libc++.so
# Exercise formatting, allocation and exception ABI symbols through Rust's
# linker before spending time compiling the vendored solver.
RUN printf '%s\n' \
      '#include <format>' \
      '#include <stdexcept>' \
      '#include <string>' \
      'extern "C" int oeis_cxx_smoke() {' \
      '  try { throw std::runtime_error(std::format("{}", 42)); }' \
      '  catch (const std::exception& e) { return std::string(e.what()) == "42" ? 42 : -1; }' \
      '}' > /tmp/oeis_cxx_smoke.cpp \
    && "$CXX" -std=c++20 -stdlib=libc++ -c /tmp/oeis_cxx_smoke.cpp -o /tmp/oeis_cxx_smoke.o \
    && ar rcs /tmp/liboeis_cxx_smoke.a /tmp/oeis_cxx_smoke.o \
    && printf '%s\n' 'extern "C" { fn oeis_cxx_smoke() -> i32; }' 'fn main() { assert_eq!(unsafe { oeis_cxx_smoke() }, 42); }' > /tmp/oeis_cxx_smoke.rs \
    && rustc --edition=2021 /tmp/oeis_cxx_smoke.rs -L native=/tmp -l static=oeis_cxx_smoke -L native=/opt/oeis-cxx/lib -l dylib=c++ -o /tmp/oeis-cxx-link-smoke \
    && /tmp/oeis-cxx-link-smoke \
    && rm /tmp/oeis_cxx_smoke.cpp /tmp/oeis_cxx_smoke.o /tmp/liboeis_cxx_smoke.a /tmp/oeis_cxx_smoke.rs /tmp/oeis-cxx-link-smoke
FROM synthesis-toolchain AS dependency-locks
COPY crates /locks/crates
RUN for crate in oeis_bigint_runtime oeis_synthesis_core oeis_wasm_evaluator; do cargo metadata --format-version 1 --manifest-path /locks/crates/$crate/Cargo.toml > /dev/null; done

# A cheap CI target: fail formatting before compiling vendored Z3/Wasmtime.
# It deliberately has no dependency on the compilation stages below.
FROM synthesis-toolchain AS native-format-check
RUN rustup component add rustfmt --toolchain 1.90.0
COPY rustfmt.toml /opt/oeis-build/rustfmt.toml
COPY crates /opt/oeis-build/crates
RUN cargo fmt --manifest-path /opt/oeis-build/crates/oeis_wasm_evaluator/Cargo.toml -- --check

# Cache native third-party compilation by manifests/locks, independent of source
# edits. Both image targets reuse this layer, including the vendored Z3 build.
FROM synthesis-toolchain AS synthesis-dependencies
ENV CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_TARGET_DIR=/opt/oeis-native-target
COPY crates/oeis_synthesis_core/Cargo.toml crates/oeis_synthesis_core/Cargo.lock /opt/oeis-build/crates/oeis_synthesis_core/
RUN mkdir -p /opt/oeis-build/crates/oeis_synthesis_core/src && echo '// Dependency cache seed only; replaced below.' > /opt/oeis-build/crates/oeis_synthesis_core/src/lib.rs && cargo build --release --locked --lib --manifest-path /opt/oeis-build/crates/oeis_synthesis_core/Cargo.toml && cargo clean --release --manifest-path /opt/oeis-build/crates/oeis_synthesis_core/Cargo.toml -p oeis_synthesis_core

# Warm the binding's feature-unified dependency graph too. Its manifests/locks,
# rather than any project source or build.rs, own this long-running cache layer.
# Preserve the core-only graph above for its qualification driver/tests.
FROM synthesis-dependencies AS native-dependencies
COPY crates/oeis_wasm_evaluator/Cargo.toml crates/oeis_wasm_evaluator/Cargo.lock /opt/oeis-build/crates/oeis_wasm_evaluator/
RUN mkdir -p /opt/oeis-build/crates/oeis_wasm_evaluator/src && echo '// Dependency cache seed only; replaced below.' > /opt/oeis-build/crates/oeis_wasm_evaluator/src/lib.rs && cargo build --release --locked --lib --manifest-path /opt/oeis-build/crates/oeis_wasm_evaluator/Cargo.toml && cargo clean --release --manifest-path /opt/oeis-build/crates/oeis_wasm_evaluator/Cargo.toml -p oeis_wasm_evaluator -p oeis_synthesis_core

# Real sources replace the inert cache seed and are always compiled/tested.
FROM native-dependencies AS synthesis-build
COPY crates /opt/oeis-build/crates
COPY src/oeis_learn/synthesis/records.schema.json /opt/oeis-build/src/oeis_learn/synthesis/records.schema.json
COPY tests/fixtures/bigint /opt/oeis-build/tests/fixtures/bigint
COPY docker/dev/build-synthesis.sh /opt/oeis-setup/build-synthesis.sh
RUN mkdir -p /opt/oeis-build-info && bash /opt/oeis-setup/build-synthesis.sh

# Portable CPU qualification target. Uses the same installer/native build as
# the development/GPU targets; no compiler or Docker dependency on the host.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS qualification-dependencies
USER root
COPY --from=synthesis-toolchain /opt/oeis-cxx /opt/oeis-cxx
ENV CC=clang-19 CXX=clang++-19 CXXFLAGS=-stdlib=libc++ CXXSTDLIB=c++ \
    LIBRARY_PATH=/opt/oeis-cxx/lib LD_LIBRARY_PATH=/opt/oeis-cxx/lib \
    CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
COPY --from=synthesis-toolchain /opt/rustup /opt/rustup
COPY --from=synthesis-toolchain /opt/cargo /opt/cargo
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    UV_PYTHON_INSTALL_DIR=/opt/oeis-python PATH=/opt/oeis-venv/bin:/opt/cargo/bin:${PATH}
RUN pip install --no-cache-dir uv==0.8.22
COPY docker/dev/install.sh /opt/oeis-setup/install.sh
COPY pyproject.toml uv.lock /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh cpu
ENV VIRTUAL_ENV=/opt/oeis-venv

# Compile the extension before the full checkout COPY. Only its exact native
# source/provenance closure can invalidate it; ordinary Python/docs edits cannot.
FROM qualification-dependencies AS qualification-native
COPY --from=native-dependencies /opt/cargo /opt/cargo
COPY --from=native-dependencies /opt/oeis-native-target /opt/oeis-native-target
COPY crates /opt/oeis-build/crates
COPY Dockerfile rustfmt.toml /opt/oeis-build/
COPY docker/dev/build-native.sh docker/dev/build-native-wheel.sh docker/dev/build-synthesis.sh docker/dev/install.sh /opt/oeis-build/docker/dev/
COPY src/oeis_learn/synthesis/records.schema.json /opt/oeis-build/src/oeis_learn/synthesis/records.schema.json
COPY src/oeis_learn/sandbox/preamble.wat /opt/oeis-build/src/oeis_learn/sandbox/preamble.wat
RUN bash /opt/oeis-build/docker/dev/build-native-wheel.sh

FROM qualification-native AS qualification
COPY --from=synthesis-build /opt/oeis-build-info /opt/oeis-build-info
COPY . /opt/oeis-build
RUN bash /opt/oeis-build/docker/dev/build-native.sh
ENV CARGO_TARGET_DIR=/opt/oeis-native-target \
    PYTHONPATH=/opt/oeis-build/src OMP_NUM_THREADS=2
WORKDIR /opt/oeis-build
CMD ["python", "-m", "oeis_learn.evaluation.repair_gate", "--output", "/evidence"]


FROM ${GPU_BASE_IMAGE} AS gpu-dependencies
USER root
COPY --from=synthesis-toolchain /opt/oeis-cxx /opt/oeis-cxx
ENV CC=clang-19 CXX=clang++-19 CXXFLAGS=-stdlib=libc++ CXXSTDLIB=c++ \
    LIBRARY_PATH=/opt/oeis-cxx/lib LD_LIBRARY_PATH=/opt/oeis-cxx/lib:${LD_LIBRARY_PATH} \
    CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
COPY --from=synthesis-toolchain /opt/rustup /opt/rustup
COPY --from=synthesis-toolchain /opt/cargo /opt/cargo
ENV PATH=/opt/venv/bin:/opt/cargo/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
COPY docker/dev/install.sh /opt/oeis-setup/install.sh
COPY pyproject.toml uv.lock /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh gpu

FROM gpu-dependencies AS gpu-native
COPY --from=native-dependencies /opt/cargo /opt/cargo
COPY --from=native-dependencies /opt/oeis-native-target /opt/oeis-native-target
COPY crates /opt/oeis-build/crates
COPY Dockerfile rustfmt.toml /opt/oeis-build/
COPY docker/dev/build-native.sh docker/dev/build-native-wheel.sh docker/dev/build-synthesis.sh docker/dev/install.sh /opt/oeis-build/docker/dev/
COPY src/oeis_learn/synthesis/records.schema.json /opt/oeis-build/src/oeis_learn/synthesis/records.schema.json
COPY src/oeis_learn/sandbox/preamble.wat /opt/oeis-build/src/oeis_learn/sandbox/preamble.wat
RUN bash /opt/oeis-build/docker/dev/build-native-wheel.sh

FROM gpu-native AS gpu
COPY --from=synthesis-build /opt/oeis-build-info /opt/oeis-build-info
COPY . /opt/oeis-build
RUN bash /opt/oeis-build/docker/dev/build-native.sh
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
# The host wrapper supplies each command and its explicit mounts.
CMD ["bash"]

