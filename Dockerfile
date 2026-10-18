# Public, pinned images only. Default build is the ROCm GPU target.
ARG GPU_BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191

# Explicit maintenance target for resolving proposed dependency locks. Production
# targets below require committed locks; they never generate or update them.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS synthesis-toolchain
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential cmake ninja-build pkg-config git && rm -rf /var/lib/apt/lists/*
RUN curl --fail --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/rustup-init.sh && bash /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.90.0
FROM synthesis-toolchain AS dependency-locks
COPY crates /locks/crates
RUN for crate in oeis_bigint_runtime oeis_synthesis_core oeis_wasm_evaluator; do cargo metadata --format-version 1 --manifest-path /locks/crates/$crate/Cargo.toml > /dev/null; done

# Native Z3 and the separate guest are built once, before Python/ROCm layers.
# Both final targets inherit identical dependency smoke artifacts and caches.
FROM synthesis-toolchain AS synthesis-build
COPY crates /opt/oeis-build/crates
COPY src/oeis_learn/synthesis /opt/oeis-build/src/oeis_learn/synthesis
COPY tests/fixtures/bigint /opt/oeis-build/tests/fixtures/bigint
COPY docker/dev/build-synthesis.sh /opt/oeis-setup/build-synthesis.sh
RUN mkdir -p /opt/oeis-build-info && bash /opt/oeis-setup/build-synthesis.sh

# Portable CPU qualification target. Uses the same installer/native build as
# the development/GPU targets; no compiler or Docker dependency on the host.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS qualification
USER root
COPY --from=synthesis-build /opt/rustup /opt/rustup
COPY --from=synthesis-build /opt/cargo /opt/cargo
COPY --from=synthesis-build /opt/oeis-native-target /opt/oeis-native-target
COPY --from=synthesis-build /opt/oeis-build-info /opt/oeis-build-info
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    UV_PYTHON_INSTALL_DIR=/opt/oeis-python PATH=/opt/oeis-venv/bin:/opt/cargo/bin:${PATH}
RUN pip install --no-cache-dir uv==0.8.22
COPY docker/dev /opt/oeis-setup
COPY pyproject.toml uv.lock README.md setup.py /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh cpu
ENV VIRTUAL_ENV=/opt/oeis-venv
COPY . /opt/oeis-build
RUN bash /opt/oeis-setup/build-native.sh
ENV CARGO_TARGET_DIR=/opt/oeis-native-target \
    PYTHONPATH=/opt/oeis-build/src OMP_NUM_THREADS=2
WORKDIR /opt/oeis-build
CMD ["python", "-m", "oeis_learn.evaluation.repair_gate", "--output", "/evidence"]


FROM ${GPU_BASE_IMAGE} AS gpu
USER root
COPY --from=synthesis-build /opt/rustup /opt/rustup
COPY --from=synthesis-build /opt/cargo /opt/cargo
COPY --from=synthesis-build /opt/oeis-native-target /opt/oeis-native-target
COPY --from=synthesis-build /opt/oeis-build-info /opt/oeis-build-info
ENV PATH=/opt/venv/bin:/opt/cargo/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
COPY docker/dev /opt/oeis-setup
COPY pyproject.toml uv.lock README.md setup.py /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh gpu
COPY . /opt/oeis-build
RUN bash /opt/oeis-setup/build-native.sh
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
# The host wrapper supplies each command and its explicit mounts.
CMD ["bash"]

