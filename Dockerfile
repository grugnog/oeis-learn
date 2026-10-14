# Public, pinned images only. Default build is the ROCm GPU target.
ARG GPU_BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191

# Portable CPU qualification target. Uses the same installer/native build as
# the development/GPU targets; no compiler or Docker dependency on the host.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS qualification
USER root
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    UV_PYTHON_INSTALL_DIR=/opt/oeis-python PATH=/opt/oeis-venv/bin:/opt/cargo/bin:${PATH}
RUN pip install --no-cache-dir uv==0.8.22
COPY docker/dev /opt/oeis-setup
COPY pyproject.toml uv.lock README.md setup.py /opt/oeis-build/
RUN bash /opt/oeis-setup/install.sh cpu
ENV VIRTUAL_ENV=/opt/oeis-venv
COPY . /opt/oeis-build
RUN bash /opt/oeis-setup/build-native.sh
ENV CARGO_TARGET_DIR=/opt/oeis-build/crates/oeis_wasm_evaluator/target \
    PYTHONPATH=/opt/oeis-build/src OMP_NUM_THREADS=2
WORKDIR /opt/oeis-build
CMD ["python", "-m", "oeis_learn.evaluation.repair_gate", "--output", "/evidence"]


FROM ${GPU_BASE_IMAGE} AS gpu
USER root
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

